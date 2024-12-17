//===--- Builtins.cpp - Builtin function implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements various things for builtin functions.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Builtins.h"
#include "BuiltinTargetFeatures.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/StringRef.h"
using namespace clang;

const char *HeaderDesc::getName() const {
  switch (ID) {
#define HEADER(ID, NAME)                                                       \
  case ID:                                                                     \
    return NAME;
#include "clang/Basic/BuiltinHeaders.def"
#undef HEADER
  };
  llvm_unreachable("Unknown HeaderDesc::HeaderID enum");
}

static constexpr unsigned NumBuiltins = Builtin::FirstTSBuiltin;

#define GET_BUILTIN_STR_TABLE
#include "clang/Basic/Builtins.inc"
#undef GET_BUILTIN_STR_TABLE

static constexpr Builtin::Info BuiltinInfos[] = {
    Builtin::Info{}, // No-builtin info entry.
#define GET_BUILTIN_INFOS
#include "clang/Basic/Builtins.inc"
#undef GET_BUILTIN_INFOS
};
static_assert(std::size(BuiltinInfos) == NumBuiltins);

std::pair<const Builtin::InfosShard &, const Builtin::Info &>
Builtin::Context::getShardAndInfo(unsigned ID) const {
  assert((ID < (Builtin::FirstTSBuiltin + NumTargetBuiltins +
                NumAuxTargetBuiltins)) &&
         "Invalid builtin ID!");

  ArrayRef<InfosShard> Shards = BuiltinShards;
  if (isAuxBuiltinID(ID)) {
    Shards = AuxTargetShards;
    ID = getAuxBuiltinID(ID) - Builtin::FirstTSBuiltin;
  } else if (ID >= Builtin::FirstTSBuiltin) {
    Shards = TargetShards;
    ID -= Builtin::FirstTSBuiltin;
  }

  // Loop over the shards to find the one matching this ID. We don't expect to
  // have many shards and so its better to search linearly than with a binary
  // search.
  for (const auto &Shard : Shards) {
    if (ID < Shard.Infos.size()) {
      return {Shard, Shard.Infos[ID]};
    }

    ID -= Shard.Infos.size();
  }
  llvm_unreachable("Invalid target builtin shard structure!");
}

std::string Builtin::Info::getName(const Builtin::InfosShard &Shard) const {
  return (Twine(Shard.NamePrefix) + (*Shard.Strings)[Offsets.Name]).str();
}

/// Return the identifier name for the specified builtin,
/// e.g. "__builtin_abs".
std::string Builtin::Context::getName(unsigned ID) const {
  const auto &[Shard, I] = getShardAndInfo(ID);
  return I.getName(Shard);
}

std::string Builtin::Context::getQuotedName(unsigned ID) const {
  const auto &[Shard, I] = getShardAndInfo(ID);
  return (Twine("'") + Shard.NamePrefix + (*Shard.Strings)[I.Offsets.Name] +
          "'")
      .str();
}

const char *Builtin::Context::getTypeString(unsigned ID) const {
  const auto &[Shard, I] = getShardAndInfo(ID);
  return (*Shard.Strings)[I.Offsets.Type].data();
}

const char *Builtin::Context::getAttributesString(unsigned ID) const {
  const auto &[Shard, I] = getShardAndInfo(ID);
  return (*Shard.Strings)[I.Offsets.Attributes].data();
}

const char *Builtin::Context::getRequiredFeatures(unsigned ID) const {
  const auto &[Shard, I] = getShardAndInfo(ID);
  return (*Shard.Strings)[I.Offsets.Features].data();
}

Builtin::Context::Context() : BuiltinShards{{&BuiltinStrings, BuiltinInfos}} {}

void Builtin::Context::InitializeTarget(const TargetInfo &Target,
                                        const TargetInfo *AuxTarget) {
  assert(TargetShards.empty() && "Already initialized target?");
  assert(NumTargetBuiltins == 0 && "Already initialized target?");
  TargetShards = Target.getTargetBuiltins();
  for (const auto &Shard : TargetShards)
    NumTargetBuiltins += Shard.Infos.size();
  if (AuxTarget) {
    AuxTargetShards = AuxTarget->getTargetBuiltins();
    for (const auto &Shard : AuxTargetShards)
      NumAuxTargetBuiltins += Shard.Infos.size();
  }
}

bool Builtin::Context::isBuiltinFunc(llvm::StringRef FuncName) {
  bool InStdNamespace = FuncName.consume_front("std-");
  for (const auto &Shard : {InfosShard{&BuiltinStrings, BuiltinInfos}})
    if (llvm::StringRef FuncNameSuffix = FuncName;
        FuncNameSuffix.consume_front(Shard.NamePrefix))
      for (const auto &I : Shard.Infos)
        if (FuncNameSuffix == (*Shard.Strings)[I.Offsets.Name] &&
            (bool)strchr((*Shard.Strings)[I.Offsets.Attributes].data(), 'z') ==
                InStdNamespace)
          return strchr((*Shard.Strings)[I.Offsets.Attributes].data(), 'f') !=
                 nullptr;

  return false;
}

/// Is this builtin supported according to the given language options?
static bool builtinIsSupported(const llvm::StringTable &Strings,
                               const Builtin::Info &BuiltinInfo,
                               const LangOptions &LangOpts) {
  auto AttributesStr = Strings[BuiltinInfo.Offsets.Attributes];

  /* Builtins Unsupported */
  if (LangOpts.NoBuiltin && strchr(AttributesStr.data(), 'f') != nullptr)
    return false;
  /* CorBuiltins Unsupported */
  if (!LangOpts.Coroutines && (BuiltinInfo.Langs & COR_LANG))
    return false;
  /* MathBuiltins Unsupported */
  if (LangOpts.NoMathBuiltin && BuiltinInfo.Header.ID == HeaderDesc::MATH_H)
    return false;
  /* GnuMode Unsupported */
  if (!LangOpts.GNUMode && (BuiltinInfo.Langs & GNU_LANG))
    return false;
  /* MSMode Unsupported */
  if (!LangOpts.MicrosoftExt && (BuiltinInfo.Langs & MS_LANG))
    return false;
  /* HLSLMode Unsupported */
  if (!LangOpts.HLSL && (BuiltinInfo.Langs & HLSL_LANG))
    return false;
  /* ObjC Unsupported */
  if (!LangOpts.ObjC && BuiltinInfo.Langs == OBJC_LANG)
    return false;
  /* OpenCLC Unsupported */
  if (!LangOpts.OpenCL && (BuiltinInfo.Langs & ALL_OCL_LANGUAGES))
    return false;
  /* OopenCL GAS Unsupported */
  if (!LangOpts.OpenCLGenericAddressSpace && (BuiltinInfo.Langs & OCL_GAS))
    return false;
  /* OpenCL Pipe Unsupported */
  if (!LangOpts.OpenCLPipes && (BuiltinInfo.Langs & OCL_PIPE))
    return false;

  // Device side enqueue is not supported until OpenCL 2.0. In 2.0 and higher
  // support is indicated with language option for blocks.

  /* OpenCL DSE Unsupported */
  if ((LangOpts.getOpenCLCompatibleVersion() < 200 || !LangOpts.Blocks) &&
      (BuiltinInfo.Langs & OCL_DSE))
    return false;
  /* OpenMP Unsupported */
  if (!LangOpts.OpenMP && BuiltinInfo.Langs == OMP_LANG)
    return false;
  /* CUDA Unsupported */
  if (!LangOpts.CUDA && BuiltinInfo.Langs == CUDA_LANG)
    return false;
  /* CPlusPlus Unsupported */
  if (!LangOpts.CPlusPlus && BuiltinInfo.Langs == CXX_LANG)
    return false;
  /* consteval Unsupported */
  if (!LangOpts.CPlusPlus20 && strchr(AttributesStr.data(), 'G') != nullptr)
    return false;
  return true;
}

/// initializeBuiltins - Mark the identifiers for all the builtins with their
/// appropriate builtin ID # and mark any non-portable builtin identifiers as
/// such.
void Builtin::Context::initializeBuiltins(IdentifierTable &Table,
                                          const LangOptions &LangOpts) {
  {
    unsigned ID = 0;
    // Step #1: mark all target-independent builtins with their ID's.
    for (const auto &Shard : BuiltinShards)
      for (const auto &I : Shard.Infos) {
        // If this is a real builtin (ID != 0) and is supported, add it.
        if (ID != 0 && builtinIsSupported(*Shard.Strings, I, LangOpts))
          Table.get(I.getName(Shard)).setBuiltinID(ID);
        ++ID;
      }
    assert(ID == FirstTSBuiltin && "Should have added all non-target IDs!");

    // Step #2: Register target-specific builtins.
    for (const auto &Shard : TargetShards)
      for (const auto &I : Shard.Infos) {
        if (builtinIsSupported(*Shard.Strings, I, LangOpts))
          Table.get(I.getName(Shard)).setBuiltinID(ID);
        ++ID;
      }

    // Step #3: Register target-specific builtins for AuxTarget.
    for (const auto &Shard : AuxTargetShards)
      for (const auto &I : Shard.Infos) {
        Table.get(I.getName(Shard)).setBuiltinID(ID);
        ++ID;
      }
  }

  // Step #4: Unregister any builtins specified by -fno-builtin-foo.
  for (llvm::StringRef Name : LangOpts.NoBuiltinFuncs) {
    bool InStdNamespace = Name.consume_front("std-");
    auto NameIt = Table.find(Name);
    if (NameIt != Table.end()) {
      unsigned ID = NameIt->second->getBuiltinID();
      if (ID != Builtin::NotBuiltin && isPredefinedLibFunction(ID) &&
          isInStdNamespace(ID) == InStdNamespace) {
        NameIt->second->clearBuiltinID();
      }
    }
  }
}

unsigned Builtin::Context::getRequiredVectorWidth(unsigned ID) const {
  const char *WidthPos = ::strchr(getAttributesString(ID), 'V');
  if (!WidthPos)
    return 0;

  ++WidthPos;
  assert(*WidthPos == ':' &&
         "Vector width specifier must be followed by a ':'");
  ++WidthPos;

  char *EndPos;
  unsigned Width = ::strtol(WidthPos, &EndPos, 10);
  assert(*EndPos == ':' && "Vector width specific must end with a ':'");
  return Width;
}

bool Builtin::Context::isLike(unsigned ID, unsigned &FormatIdx,
                              bool &HasVAListArg, const char *Fmt) const {
  assert(Fmt && "Not passed a format string");
  assert(::strlen(Fmt) == 2 &&
         "Format string needs to be two characters long");
  assert(::toupper(Fmt[0]) == Fmt[1] &&
         "Format string is not in the form \"xX\"");

  const char *Like = ::strpbrk(getAttributesString(ID), Fmt);
  if (!Like)
    return false;

  HasVAListArg = (*Like == Fmt[1]);

  ++Like;
  assert(*Like == ':' && "Format specifier must be followed by a ':'");
  ++Like;

  assert(::strchr(Like, ':') && "Format specifier must end with a ':'");
  FormatIdx = ::strtol(Like, nullptr, 10);
  return true;
}

bool Builtin::Context::isPrintfLike(unsigned ID, unsigned &FormatIdx,
                                    bool &HasVAListArg) {
  return isLike(ID, FormatIdx, HasVAListArg, "pP");
}

bool Builtin::Context::isScanfLike(unsigned ID, unsigned &FormatIdx,
                                   bool &HasVAListArg) {
  return isLike(ID, FormatIdx, HasVAListArg, "sS");
}

bool Builtin::Context::performsCallback(unsigned ID,
                                        SmallVectorImpl<int> &Encoding) const {
  const char *CalleePos = ::strchr(getAttributesString(ID), 'C');
  if (!CalleePos)
    return false;

  ++CalleePos;
  assert(*CalleePos == '<' &&
         "Callback callee specifier must be followed by a '<'");
  ++CalleePos;

  char *EndPos;
  int CalleeIdx = ::strtol(CalleePos, &EndPos, 10);
  assert(CalleeIdx >= 0 && "Callee index is supposed to be positive!");
  Encoding.push_back(CalleeIdx);

  while (*EndPos == ',') {
    const char *PayloadPos = EndPos + 1;

    int PayloadIdx = ::strtol(PayloadPos, &EndPos, 10);
    Encoding.push_back(PayloadIdx);
  }

  assert(*EndPos == '>' && "Callback callee specifier must end with a '>'");
  return true;
}

bool Builtin::Context::canBeRedeclared(unsigned ID) const {
  return ID == Builtin::NotBuiltin || ID == Builtin::BI__va_start ||
         ID == Builtin::BI__builtin_assume_aligned ||
         (!hasReferenceArgsOrResult(ID) && !hasCustomTypechecking(ID)) ||
         isInStdNamespace(ID);
}

bool Builtin::evaluateRequiredTargetFeatures(
    StringRef RequiredFeatures, const llvm::StringMap<bool> &TargetFetureMap) {
  // Return true if the builtin doesn't have any required features.
  if (RequiredFeatures.empty())
    return true;
  assert(!RequiredFeatures.contains(' ') && "Space in feature list");

  TargetFeatures TF(TargetFetureMap);
  return TF.hasRequiredFeatures(RequiredFeatures);
}
