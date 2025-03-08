//===- Multilib.cpp - Multilib Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Multilib.h"
#include "clang/Basic/LLVM.h"
#include "clang/Driver/Driver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <string>

using namespace clang;
using namespace driver;
using namespace llvm::sys;

Multilib::Multilib(StringRef GCCSuffix, StringRef OSSuffix,
                   StringRef IncludeSuffix, const flags_list &Flags,
                   StringRef ExclusiveGroup, std::optional<StringRef> Error)
    : GCCSuffix(GCCSuffix), OSSuffix(OSSuffix), IncludeSuffix(IncludeSuffix),
      Flags(Flags), ExclusiveGroup(ExclusiveGroup), Error(Error) {
  assert(GCCSuffix.empty() ||
         (StringRef(GCCSuffix).front() == '/' && GCCSuffix.size() > 1));
  assert(OSSuffix.empty() ||
         (StringRef(OSSuffix).front() == '/' && OSSuffix.size() > 1));
  assert(IncludeSuffix.empty() ||
         (StringRef(IncludeSuffix).front() == '/' && IncludeSuffix.size() > 1));
}

LLVM_DUMP_METHOD void Multilib::dump() const {
  print(llvm::errs());
}

void Multilib::print(raw_ostream &OS) const {
  if (GCCSuffix.empty())
    OS << ".";
  else {
    OS << StringRef(GCCSuffix).drop_front();
  }
  OS << ";";
  for (StringRef Flag : Flags) {
    if (Flag.front() == '-')
      OS << "@" << Flag.substr(1);
  }
}

bool Multilib::operator==(const Multilib &Other) const {
  // Check whether the flags sets match
  // allowing for the match to be order invariant
  llvm::StringSet<> MyFlags;
  for (const auto &Flag : Flags)
    MyFlags.insert(Flag);

  for (const auto &Flag : Other.Flags)
    if (!MyFlags.contains(Flag))
      return false;

  if (osSuffix() != Other.osSuffix())
    return false;

  if (gccSuffix() != Other.gccSuffix())
    return false;

  if (includeSuffix() != Other.includeSuffix())
    return false;

  return true;
}

raw_ostream &clang::driver::operator<<(raw_ostream &OS, const Multilib &M) {
  M.print(OS);
  return OS;
}

MultilibSet &MultilibSet::FilterOut(FilterCallback F) {
  llvm::erase_if(Multilibs, F);
  return *this;
}

void MultilibSet::push_back(const Multilib &M) { Multilibs.push_back(M); }

static void DiagnoseUnclaimedMultilibCustomFlags(
    const Driver &D, const SmallVector<StringRef> &UnclaimedCustomFlagValues,
    const SmallVector<custom_flag::Declaration> &CustomFlagDecls) {
  struct EditDistanceInfo {
    StringRef FlagValue;
    unsigned EditDistance;
  };
  const unsigned MaxEditDistance = 5;

  for (StringRef Unclaimed : UnclaimedCustomFlagValues) {
    std::optional<EditDistanceInfo> BestCandidate;
    for (const auto &Decl : CustomFlagDecls) {
      for (const auto &Value : Decl.ValueList) {
        const std::string &FlagValueName = Value.Name;
        unsigned EditDistance =
            Unclaimed.edit_distance(FlagValueName, /*AllowReplacements=*/true,
                                    /*MaxEditDistance=*/MaxEditDistance);
        if (!BestCandidate || (EditDistance <= MaxEditDistance &&
                               EditDistance < BestCandidate->EditDistance)) {
          BestCandidate = {FlagValueName, EditDistance};
        }
      }
    }
    if (!BestCandidate)
      D.Diag(clang::diag::err_drv_unsupported_opt)
          << (custom_flag::Prefix + Unclaimed).str();
    else
      D.Diag(clang::diag::err_drv_unsupported_opt_with_suggestion)
          << (custom_flag::Prefix + Unclaimed).str()
          << (custom_flag::Prefix + BestCandidate->FlagValue).str();
  }
}

namespace clang::driver::custom_flag {
// Map implemented using linear searches as the expected size is too small for
// the overhead of a search tree or a hash table.
class ValueNameToDetailMap {
  SmallVector<std::pair<StringRef, const ValueDetail *>> Mapping;

public:
  template <typename It>
  ValueNameToDetailMap(It FlagDeclsBegin, It FlagDeclsEnd) {
    for (auto DeclIt = FlagDeclsBegin; DeclIt != FlagDeclsEnd; ++DeclIt) {
      const Declaration &Decl = *DeclIt;
      for (const auto &Value : Decl.ValueList)
        Mapping.emplace_back(Value.Name, &Value);
    }
  }

  const ValueDetail *get(StringRef Key) const {
    auto Iter = llvm::find_if(
        Mapping, [&](const auto &Pair) { return Pair.first == Key; });
    return Iter != Mapping.end() ? Iter->second : nullptr;
  }
};
} // namespace clang::driver::custom_flag

std::pair<Multilib::flags_list, SmallVector<StringRef>>
MultilibSet::processCustomFlags(const Driver &D,
                                const Multilib::flags_list &Flags) const {
  Multilib::flags_list Result;
  SmallVector<StringRef> MacroDefines;

  // Custom flag values detected in the flags list
  SmallVector<const custom_flag::ValueDetail *> ClaimedCustomFlagValues;

  // Arguments to -fmultilib-flag=<arg> that don't correspond to any valid
  // custom flag value. An error will be printed out for each of these.
  SmallVector<StringRef> UnclaimedCustomFlagValueStrs;

  const auto ValueNameToValueDetail = custom_flag::ValueNameToDetailMap(
      CustomFlagDecls.begin(), CustomFlagDecls.end());

  for (StringRef Flag : Flags) {
    if (!Flag.starts_with(custom_flag::Prefix)) {
      Result.push_back(Flag.str());
      continue;
    }

    StringRef CustomFlagValueStr = Flag.substr(custom_flag::Prefix.size());
    const custom_flag::ValueDetail *Detail =
        ValueNameToValueDetail.get(CustomFlagValueStr);
    if (Detail)
      ClaimedCustomFlagValues.push_back(Detail);
    else
      UnclaimedCustomFlagValueStrs.push_back(CustomFlagValueStr);
  }

  // Set of custom flag declarations for which a value was passed in the flags
  // list. This is used to, firstly, detect multiple values for the same flag
  // declaration (in this case, the last one wins), and secondly, to detect
  // which declarations had no value passed in (in this case, the default value
  // is selected).
  llvm::SmallPtrSet<custom_flag::Declaration *, 32> TriggeredCustomFlagDecls;

  // Detect multiple values for the same flag declaration. Last one wins.
  for (auto *CustomFlagValue : llvm::reverse(ClaimedCustomFlagValues)) {
    if (!TriggeredCustomFlagDecls.insert(CustomFlagValue->Decl).second)
      continue;
    Result.push_back(std::string(custom_flag::Prefix) + CustomFlagValue->Name);
    if (CustomFlagValue->MacroDefines)
      MacroDefines.append(CustomFlagValue->MacroDefines->begin(),
                          CustomFlagValue->MacroDefines->end());
  }

  // Detect flag declarations with no value passed in. Select default value.
  for (const auto &Decl : CustomFlagDecls) {
    if (TriggeredCustomFlagDecls.contains(&Decl))
      continue;
    const custom_flag::ValueDetail &CustomFlagValue =
        Decl.ValueList[*Decl.DefaultValueIdx];
    Result.push_back(std::string(custom_flag::Prefix) + CustomFlagValue.Name);
    if (CustomFlagValue.MacroDefines)
      MacroDefines.append(CustomFlagValue.MacroDefines->begin(),
                          CustomFlagValue.MacroDefines->end());
  }

  DiagnoseUnclaimedMultilibCustomFlags(D, UnclaimedCustomFlagValueStrs,
                                       CustomFlagDecls);

  return {Result, MacroDefines};
}

bool MultilibSet::select(
    const Driver &D, const Multilib::flags_list &Flags,
    llvm::SmallVectorImpl<Multilib> &Selected,
    llvm::SmallVector<StringRef> *CustomFlagMacroDefines) const {
  auto [FlagsWithCustom, CFMacroDefines] = processCustomFlags(D, Flags);
  llvm::StringSet<> FlagSet(expandFlags(FlagsWithCustom));
  Selected.clear();
  bool AnyErrors = false;

  // Determining the list of macro defines depends only on the custom flags
  // passed in. The library variants actually selected are not relevant in
  // this. Therefore this assignment can take place before the selection
  // happens.
  if (CustomFlagMacroDefines)
    *CustomFlagMacroDefines = std::move(CFMacroDefines);

  // Decide which multilibs we're going to select at all.
  llvm::DenseSet<StringRef> ExclusiveGroupsSelected;
  for (const Multilib &M : llvm::reverse(Multilibs)) {
    // If this multilib doesn't match all our flags, don't select it.
    if (!llvm::all_of(M.flags(), [&FlagSet](const std::string &F) {
          return FlagSet.contains(F);
        }))
      continue;

    const std::string &group = M.exclusiveGroup();
    if (!group.empty()) {
      // If this multilib has the same ExclusiveGroup as one we've already
      // selected, skip it. We're iterating in reverse order, so the group
      // member we've selected already is preferred.
      //
      // Otherwise, add the group name to the set of groups we've already
      // selected a member of.
      auto [It, Inserted] = ExclusiveGroupsSelected.insert(group);
      if (!Inserted)
        continue;
    }

    // If this multilib is actually a placeholder containing an error message
    // written by the multilib.yaml author, then set a flag that will cause a
    // failure return. Our caller will display the error message.
    if (M.isError())
      AnyErrors = true;

    // Select this multilib.
    Selected.push_back(M);
  }

  // We iterated in reverse order, so now put Selected back the right way
  // round.
  std::reverse(Selected.begin(), Selected.end());

  return !AnyErrors && !Selected.empty();
}

llvm::StringSet<>
MultilibSet::expandFlags(const Multilib::flags_list &InFlags) const {
  llvm::StringSet<> Result;
  for (const auto &F : InFlags)
    Result.insert(F);
  for (const FlagMatcher &M : FlagMatchers) {
    std::string RegexString(M.Match);

    // Make the regular expression match the whole string.
    if (!StringRef(M.Match).starts_with("^"))
      RegexString.insert(RegexString.begin(), '^');
    if (!StringRef(M.Match).ends_with("$"))
      RegexString.push_back('$');

    const llvm::Regex Regex(RegexString);
    assert(Regex.isValid());
    if (llvm::any_of(InFlags,
                     [&Regex](StringRef F) { return Regex.match(F); })) {
      Result.insert(M.Flags.begin(), M.Flags.end());
    }
  }
  return Result;
}

namespace {

// When updating this also update MULTILIB_VERSION in MultilibTest.cpp
static const VersionTuple MultilibVersionCurrent(1, 0);

struct MultilibSerialization {
  std::string Dir;        // if this record successfully selects a library dir
  std::string Error;      // if this record reports a fatal error message
  std::vector<std::string> Flags;
  std::string Group;
};

enum class MultilibGroupType {
  /*
   * The only group type currently supported is 'Exclusive', which indicates a
   * group of multilibs of which at most one may be selected.
   */
  Exclusive,

  /*
   * Future possibility: a second group type indicating a set of library
   * directories that are mutually _dependent_ rather than mutually exclusive:
   * if you include one you must include them all.
   *
   * It might also be useful to allow groups to be members of other groups, so
   * that a mutually exclusive group could contain a mutually dependent set of
   * library directories, or vice versa.
   *
   * These additional features would need changes in the implementation, but
   * the YAML schema is set up so they can be added without requiring changes
   * in existing users' multilib.yaml files.
   */
};

struct MultilibGroupSerialization {
  std::string Name;
  MultilibGroupType Type;
};

struct MultilibSetSerialization {
  llvm::VersionTuple MultilibVersion;
  SmallVector<MultilibGroupSerialization> Groups;
  SmallVector<MultilibSerialization> Multilibs;
  SmallVector<MultilibSet::FlagMatcher> FlagMatchers;
  SmallVector<custom_flag::Declaration> CustomFlagDeclarations;
};

} // end anonymous namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(MultilibSerialization)
LLVM_YAML_IS_SEQUENCE_VECTOR(MultilibGroupSerialization)
LLVM_YAML_IS_SEQUENCE_VECTOR(MultilibSet::FlagMatcher)
LLVM_YAML_IS_SEQUENCE_VECTOR(custom_flag::ValueDetail)
LLVM_YAML_IS_SEQUENCE_VECTOR(custom_flag::Declaration)

template <> struct llvm::yaml::MappingTraits<MultilibSerialization> {
  static void mapping(llvm::yaml::IO &io, MultilibSerialization &V) {
    io.mapOptional("Dir", V.Dir);
    io.mapOptional("Error", V.Error);
    io.mapRequired("Flags", V.Flags);
    io.mapOptional("Group", V.Group);
  }
  static std::string validate(IO &io, MultilibSerialization &V) {
    if (V.Dir.empty() && V.Error.empty())
      return "one of the 'Dir' and 'Error' keys must be specified";
    if (!V.Dir.empty() && !V.Error.empty())
      return "the 'Dir' and 'Error' keys may not both be specified";
    if (StringRef(V.Dir).starts_with("/"))
      return "paths must be relative but \"" + V.Dir + "\" starts with \"/\"";
    return std::string{};
  }
};

template <> struct llvm::yaml::ScalarEnumerationTraits<MultilibGroupType> {
  static void enumeration(IO &io, MultilibGroupType &Val) {
    io.enumCase(Val, "Exclusive", MultilibGroupType::Exclusive);
  }
};

template <> struct llvm::yaml::MappingTraits<MultilibGroupSerialization> {
  static void mapping(llvm::yaml::IO &io, MultilibGroupSerialization &V) {
    io.mapRequired("Name", V.Name);
    io.mapRequired("Type", V.Type);
  }
};

template <> struct llvm::yaml::MappingTraits<MultilibSet::FlagMatcher> {
  static void mapping(llvm::yaml::IO &io, MultilibSet::FlagMatcher &M) {
    io.mapRequired("Match", M.Match);
    io.mapRequired("Flags", M.Flags);
  }
  static std::string validate(IO &io, MultilibSet::FlagMatcher &M) {
    llvm::Regex Regex(M.Match);
    std::string RegexError;
    if (!Regex.isValid(RegexError))
      return RegexError;
    if (M.Flags.empty())
      return "value required for 'Flags'";
    return std::string{};
  }
};

template <>
struct llvm::yaml::MappingContextTraits<custom_flag::ValueDetail,
                                        llvm::SmallSet<std::string, 32>> {
  static void mapping(llvm::yaml::IO &io, custom_flag::ValueDetail &V,
                      llvm::SmallSet<std::string, 32> &) {
    io.mapRequired("Name", V.Name);
    io.mapOptional("MacroDefines", V.MacroDefines);
  }
  static std::string validate(IO &io, custom_flag::ValueDetail &V,
                              llvm::SmallSet<std::string, 32> &NameSet) {
    if (V.Name.empty())
      return "custom flag value requires a name";
    if (!NameSet.insert(V.Name).second)
      return "duplicate custom flag value name: \"" + V.Name + "\"";
    return {};
  }
};

template <>
struct llvm::yaml::MappingContextTraits<custom_flag::Declaration,
                                        llvm::SmallSet<std::string, 32>> {
  static void mapping(llvm::yaml::IO &io, custom_flag::Declaration &V,
                      llvm::SmallSet<std::string, 32> &NameSet) {
    io.mapRequired("Name", V.Name);
    io.mapRequired("Values", V.ValueList, NameSet);
    std::string DefaultValueName;
    io.mapRequired("Default", DefaultValueName);

    for (auto [Idx, Value] : llvm::enumerate(V.ValueList)) {
      Value.Decl = &V;
      if (Value.Name == DefaultValueName) {
        assert(!V.DefaultValueIdx);
        V.DefaultValueIdx = Idx;
      }
    }
  }
  static std::string validate(IO &io, custom_flag::Declaration &V,
                              llvm::SmallSet<std::string, 32> &) {
    if (V.Name.empty())
      return "custom flag requires a name";
    if (V.ValueList.empty())
      return "custom flag must have at least one value";
    if (!V.DefaultValueIdx)
      return "custom flag must have a default value";
    return {};
  }
};

template <> struct llvm::yaml::MappingTraits<MultilibSetSerialization> {
  static void mapping(llvm::yaml::IO &io, MultilibSetSerialization &M) {
    io.mapRequired("MultilibVersion", M.MultilibVersion);
    io.mapRequired("Variants", M.Multilibs);
    io.mapOptional("Groups", M.Groups);
    llvm::SmallSet<std::string, 32> NameSet;
    io.mapOptionalWithContext("Flags", M.CustomFlagDeclarations, NameSet);
    io.mapOptional("Mappings", M.FlagMatchers);
  }
  static std::string validate(IO &io, MultilibSetSerialization &M) {
    if (M.MultilibVersion.empty())
      return "missing required key 'MultilibVersion'";
    if (M.MultilibVersion.getMajor() != MultilibVersionCurrent.getMajor())
      return "multilib version " + M.MultilibVersion.getAsString() +
             " is unsupported";
    if (M.MultilibVersion.getMinor() > MultilibVersionCurrent.getMinor())
      return "multilib version " + M.MultilibVersion.getAsString() +
             " is unsupported";
    for (const MultilibSerialization &Lib : M.Multilibs) {
      if (!Lib.Group.empty()) {
        bool Found = false;
        for (const MultilibGroupSerialization &Group : M.Groups)
          if (Group.Name == Lib.Group) {
            Found = true;
            break;
          }
        if (!Found)
          return "multilib \"" + Lib.Dir +
                 "\" specifies undefined group name \"" + Lib.Group + "\"";
      }
    }
    return std::string{};
  }
};

llvm::ErrorOr<MultilibSet>
MultilibSet::parseYaml(llvm::MemoryBufferRef Input,
                       llvm::SourceMgr::DiagHandlerTy DiagHandler,
                       void *DiagHandlerCtxt) {
  MultilibSetSerialization MS;
  llvm::yaml::Input YamlInput(Input, nullptr, DiagHandler, DiagHandlerCtxt);
  YamlInput >> MS;
  if (YamlInput.error())
    return YamlInput.error();

  multilib_list Multilibs;
  Multilibs.reserve(MS.Multilibs.size());
  for (const auto &M : MS.Multilibs) {
    if (!M.Error.empty()) {
      Multilibs.emplace_back("", "", "", M.Flags, M.Group, M.Error);
    } else {
      std::string Dir;
      if (M.Dir != ".")
        Dir = "/" + M.Dir;
      // We transfer M.Group straight into the ExclusiveGroup parameter for the
      // Multilib constructor. If we later support more than one type of group,
      // we'll have to look up the group name in MS.Groups, check its type, and
      // decide what to do here.
      Multilibs.emplace_back(Dir, Dir, Dir, M.Flags, M.Group);
    }
  }

  return MultilibSet(std::move(Multilibs), std::move(MS.FlagMatchers),
                     std::move(MS.CustomFlagDeclarations));
}

LLVM_DUMP_METHOD void MultilibSet::dump() const {
  print(llvm::errs());
}

void MultilibSet::print(raw_ostream &OS) const {
  for (const auto &M : *this)
    OS << M << "\n";
}

raw_ostream &clang::driver::operator<<(raw_ostream &OS, const MultilibSet &MS) {
  MS.print(OS);
  return OS;
}

namespace clang::driver::custom_flag {
Declaration::Declaration(const Declaration &Other)
    : Name(Other.Name), ValueList(Other.ValueList),
      DefaultValueIdx(Other.DefaultValueIdx) {
  for (ValueDetail &Detail : ValueList)
    Detail.Decl = this;
}

Declaration::Declaration(Declaration &&Other)
    : Name(std::move(Other.Name)), ValueList(std::move(Other.ValueList)),
      DefaultValueIdx(std::move(Other.DefaultValueIdx)) {
  for (ValueDetail &Detail : ValueList)
    Detail.Decl = this;
}

Declaration &Declaration::operator=(const Declaration &Other) {
  if (this == &Other)
    return *this;
  Name = Other.Name;
  ValueList = Other.ValueList;
  DefaultValueIdx = Other.DefaultValueIdx;
  for (ValueDetail &Detail : ValueList)
    Detail.Decl = this;
  return *this;
}

Declaration &Declaration::operator=(Declaration &&Other) {
  if (this == &Other)
    return *this;
  Name = std::move(Other.Name);
  ValueList = std::move(Other.ValueList);
  DefaultValueIdx = std::move(Other.DefaultValueIdx);
  for (ValueDetail &Detail : ValueList)
    Detail.Decl = this;
  return *this;
}
} // namespace clang::driver::custom_flag
