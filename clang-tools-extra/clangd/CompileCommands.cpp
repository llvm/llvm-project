//===--- CompileCommands.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CompileCommands.h"
#include "Config.h"
#include "support/Logger.h"
#include "support/Trace.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include <iterator>
#include <string>
#include <vector>

namespace clang {
namespace clangd {
namespace {

// Query apple's `xcrun` launcher, which is the source of truth for "how should"
// clang be invoked on this system.
llvm::Optional<std::string> queryXcrun(llvm::ArrayRef<llvm::StringRef> Argv) {
  auto Xcrun = llvm::sys::findProgramByName("xcrun");
  if (!Xcrun) {
    log("Couldn't find xcrun. Hopefully you have a non-apple toolchain...");
    return llvm::None;
  }
  llvm::SmallString<64> OutFile;
  llvm::sys::fs::createTemporaryFile("clangd-xcrun", "", OutFile);
  llvm::FileRemover OutRemover(OutFile);
  llvm::Optional<llvm::StringRef> Redirects[3] = {
      /*stdin=*/{""}, /*stdout=*/{OutFile.str()}, /*stderr=*/{""}};
  vlog("Invoking {0} to find clang installation", *Xcrun);
  int Ret = llvm::sys::ExecuteAndWait(*Xcrun, Argv,
                                      /*Env=*/llvm::None, Redirects,
                                      /*SecondsToWait=*/10);
  if (Ret != 0) {
    log("xcrun exists but failed with code {0}. "
        "If you have a non-apple toolchain, this is OK. "
        "Otherwise, try xcode-select --install.",
        Ret);
    return llvm::None;
  }

  auto Buf = llvm::MemoryBuffer::getFile(OutFile);
  if (!Buf) {
    log("Can't read xcrun output: {0}", Buf.getError().message());
    return llvm::None;
  }
  StringRef Path = Buf->get()->getBuffer().trim();
  if (Path.empty()) {
    log("xcrun produced no output");
    return llvm::None;
  }
  return Path.str();
}

// Resolve symlinks if possible.
std::string resolve(std::string Path) {
  llvm::SmallString<128> Resolved;
  if (llvm::sys::fs::real_path(Path, Resolved)) {
    log("Failed to resolve possible symlink {0}", Path);
    return Path;
  }
  return std::string(Resolved.str());
}

// Get a plausible full `clang` path.
// This is used in the fallback compile command, or when the CDB returns a
// generic driver with no path.
std::string detectClangPath() {
  // The driver and/or cc1 sometimes depend on the binary name to compute
  // useful things like the standard library location.
  // We need to emulate what clang on this system is likely to see.
  // cc1 in particular looks at the "real path" of the running process, and
  // so if /usr/bin/clang is a symlink, it sees the resolved path.
  // clangd doesn't have that luxury, so we resolve symlinks ourselves.

  // On Mac, `which clang` is /usr/bin/clang. It runs `xcrun clang`, which knows
  // where the real clang is kept. We need to do the same thing,
  // because cc1 (not the driver!) will find libc++ relative to argv[0].
#ifdef __APPLE__
  if (auto MacClang = queryXcrun({"xcrun", "--find", "clang"}))
    return resolve(std::move(*MacClang));
#endif
  // On other platforms, just look for compilers on the PATH.
  for (const char *Name : {"clang", "gcc", "cc"})
    if (auto PathCC = llvm::sys::findProgramByName(Name))
      return resolve(std::move(*PathCC));
  // Fallback: a nonexistent 'clang' binary next to clangd.
  static int StaticForMainAddr;
  std::string ClangdExecutable =
      llvm::sys::fs::getMainExecutable("clangd", (void *)&StaticForMainAddr);
  SmallString<128> ClangPath;
  ClangPath = llvm::sys::path::parent_path(ClangdExecutable);
  llvm::sys::path::append(ClangPath, "clang");
  return std::string(ClangPath.str());
}

// On mac, /usr/bin/clang sets SDKROOT and then invokes the real clang.
// The effect of this is to set -isysroot correctly. We do the same.
const llvm::Optional<std::string> detectSysroot() {
#ifndef __APPLE__
  return llvm::None;
#endif

  // SDKROOT overridden in environment, respect it. Driver will set isysroot.
  if (::getenv("SDKROOT"))
    return llvm::None;
  return queryXcrun({"xcrun", "--show-sdk-path"});
}

std::string detectStandardResourceDir() {
  static int StaticForMainAddr; // Just an address in this process.
  return CompilerInvocation::GetResourcesPath("clangd",
                                              (void *)&StaticForMainAddr);
}

// The path passed to argv[0] is important:
//  - its parent directory is Driver::Dir, used for library discovery
//  - its basename affects CLI parsing (clang-cl) and other settings
// Where possible it should be an absolute path with sensible directory, but
// with the original basename.
static std::string resolveDriver(llvm::StringRef Driver, bool FollowSymlink,
                                 llvm::Optional<std::string> ClangPath) {
  auto SiblingOf = [&](llvm::StringRef AbsPath) {
    llvm::SmallString<128> Result = llvm::sys::path::parent_path(AbsPath);
    llvm::sys::path::append(Result, llvm::sys::path::filename(Driver));
    return Result.str().str();
  };

  // First, eliminate relative paths.
  std::string Storage;
  if (!llvm::sys::path::is_absolute(Driver)) {
    // If it's working-dir relative like bin/clang, we can't resolve it.
    // FIXME: we could if we had the working directory here.
    // Let's hope it's not a symlink.
    if (llvm::any_of(Driver,
                     [](char C) { return llvm::sys::path::is_separator(C); }))
      return Driver.str();
    // If the driver is a generic like "g++" with no path, add clang dir.
    if (ClangPath &&
        (Driver == "clang" || Driver == "clang++" || Driver == "gcc" ||
         Driver == "g++" || Driver == "cc" || Driver == "c++")) {
      return SiblingOf(*ClangPath);
    }
    // Otherwise try to look it up on PATH. This won't change basename.
    auto Absolute = llvm::sys::findProgramByName(Driver);
    if (Absolute && llvm::sys::path::is_absolute(*Absolute))
      Driver = Storage = std::move(*Absolute);
    else if (ClangPath) // If we don't find it, use clang dir again.
      return SiblingOf(*ClangPath);
    else // Nothing to do: can't find the command and no detected dir.
      return Driver.str();
  }

  // Now we have an absolute path, but it may be a symlink.
  assert(llvm::sys::path::is_absolute(Driver));
  if (FollowSymlink) {
    llvm::SmallString<256> Resolved;
    if (!llvm::sys::fs::real_path(Driver, Resolved))
      return SiblingOf(Resolved);
  }
  return Driver.str();
}

} // namespace

CommandMangler CommandMangler::detect() {
  CommandMangler Result;
  Result.ClangPath = detectClangPath();
  Result.ResourceDir = detectStandardResourceDir();
  Result.Sysroot = detectSysroot();
  return Result;
}

CommandMangler CommandMangler::forTests() { return CommandMangler(); }

void CommandMangler::adjust(std::vector<std::string> &Cmd,
                            llvm::StringRef File) const {
  trace::Span S("AdjustCompileFlags");
  // Most of the modifications below assumes the Cmd starts with a driver name.
  // We might consider injecting a generic driver name like "cc" or "c++", but
  // a Cmd missing the driver is probably rare enough in practice and errnous.
  if (Cmd.empty())
    return;
  auto &OptTable = clang::driver::getDriverOptTable();
  // OriginalArgs needs to outlive ArgList.
  llvm::SmallVector<const char *, 16> OriginalArgs;
  OriginalArgs.reserve(Cmd.size());
  for (const auto &S : Cmd)
    OriginalArgs.push_back(S.c_str());
  bool IsCLMode = driver::IsClangCL(driver::getDriverMode(
      OriginalArgs[0], llvm::makeArrayRef(OriginalArgs).slice(1)));
  // ParseArgs propagates missig arg/opt counts on error, but preserves
  // everything it could parse in ArgList. So we just ignore those counts.
  unsigned IgnoredCount;
  // Drop the executable name, as ParseArgs doesn't expect it. This means
  // indices are actually of by one between ArgList and OriginalArgs.
  llvm::opt::InputArgList ArgList;
  ArgList = OptTable.ParseArgs(
      llvm::makeArrayRef(OriginalArgs).drop_front(), IgnoredCount, IgnoredCount,
      /*FlagsToInclude=*/
      IsCLMode ? (driver::options::CLOption | driver::options::CoreOption)
               : /*everything*/ 0,
      /*FlagsToExclude=*/driver::options::NoDriverOption |
          (IsCLMode ? 0 : driver::options::CLOption));

  llvm::SmallVector<unsigned, 1> IndicesToDrop;
  // Having multiple architecture options (e.g. when building fat binaries)
  // results in multiple compiler jobs, which clangd cannot handle. In such
  // cases strip all the `-arch` options and fallback to default architecture.
  // As there are no signals to figure out which one user actually wants. They
  // can explicitly specify one through `CompileFlags.Add` if need be.
  unsigned ArchOptCount = 0;
  for (auto *Input : ArgList.filtered(driver::options::OPT_arch)) {
    ++ArchOptCount;
    for (auto I = 0U; I <= Input->getNumValues(); ++I)
      IndicesToDrop.push_back(Input->getIndex() + I);
  }
  // If there is a single `-arch` option, keep it.
  if (ArchOptCount < 2)
    IndicesToDrop.clear();

  // In some cases people may try to reuse the command from another file, e.g.
  //   { File: "foo.h", CommandLine: "clang foo.cpp" }.
  // We assume the intent is to parse foo.h the same way as foo.cpp, or as if
  // it were being included from foo.cpp.
  //
  // We're going to rewrite the command to refer to foo.h, and this may change
  // its semantics (e.g. by parsing the file as C). If we do this, we should
  // use transferCompileCommand to adjust the argv.
  // In practice only the extension of the file matters, so do this only when
  // it differs.
  llvm::StringRef FileExtension = llvm::sys::path::extension(File);
  llvm::Optional<std::string> TransferFrom;
  auto SawInput = [&](llvm::StringRef Input) {
    if (llvm::sys::path::extension(Input) != FileExtension)
      TransferFrom.emplace(Input);
  };

  // Strip all the inputs and `--`. We'll put the input for the requested file
  // explicitly at the end of the flags. This ensures modifications done in the
  // following steps apply in more cases (like setting -x, which only affects
  // inputs that come after it).
  for (auto *Input : ArgList.filtered(driver::options::OPT_INPUT)) {
    SawInput(Input->getValue(0));
    IndicesToDrop.push_back(Input->getIndex());
  }
  // Anything after `--` is also treated as input, drop them as well.
  if (auto *DashDash =
          ArgList.getLastArgNoClaim(driver::options::OPT__DASH_DASH)) {
    auto DashDashIndex = DashDash->getIndex() + 1; // +1 accounts for Cmd[0]
    for (unsigned I = DashDashIndex; I < Cmd.size(); ++I)
      SawInput(Cmd[I]);
    Cmd.resize(DashDashIndex);
  }
  llvm::sort(IndicesToDrop);
  llvm::for_each(llvm::reverse(IndicesToDrop),
                 // +1 to account for the executable name in Cmd[0] that
                 // doesn't exist in ArgList.
                 [&Cmd](unsigned Idx) { Cmd.erase(Cmd.begin() + Idx + 1); });
  // All the inputs are stripped, append the name for the requested file. Rest
  // of the modifications should respect `--`.
  Cmd.push_back("--");
  Cmd.push_back(File.str());

  if (TransferFrom) {
    tooling::CompileCommand TransferCmd;
    TransferCmd.Filename = std::move(*TransferFrom);
    TransferCmd.CommandLine = std::move(Cmd);
    TransferCmd = transferCompileCommand(std::move(TransferCmd), File);
    Cmd = std::move(TransferCmd.CommandLine);
    assert(Cmd.size() >= 2 && Cmd.back() == File &&
           Cmd[Cmd.size() - 2] == "--" &&
           "TransferCommand should produce a command ending in -- filename");
  }

  for (auto &Edit : Config::current().CompileFlags.Edits)
    Edit(Cmd);

  // Check whether the flag exists, either as -flag or -flag=*
  auto Has = [&](llvm::StringRef Flag) {
    for (llvm::StringRef Arg : Cmd) {
      if (Arg.consume_front(Flag) && (Arg.empty() || Arg[0] == '='))
        return true;
    }
    return false;
  };

  llvm::erase_if(Cmd, [](llvm::StringRef Elem) {
    return Elem.startswith("--save-temps") || Elem.startswith("-save-temps");
  });

  std::vector<std::string> ToAppend;
  if (ResourceDir && !Has("-resource-dir"))
    ToAppend.push_back(("-resource-dir=" + *ResourceDir));

  // Don't set `-isysroot` if it is already set or if `--sysroot` is set.
  // `--sysroot` is a superset of the `-isysroot` argument.
  if (Sysroot && !Has("-isysroot") && !Has("--sysroot")) {
    ToAppend.push_back("-isysroot");
    ToAppend.push_back(*Sysroot);
  }

  if (!ToAppend.empty()) {
    Cmd.insert(llvm::find(Cmd, "--"), std::make_move_iterator(ToAppend.begin()),
               std::make_move_iterator(ToAppend.end()));
  }

  if (!Cmd.empty()) {
    bool FollowSymlink = !Has("-no-canonical-prefixes");
    Cmd.front() =
        (FollowSymlink ? ResolvedDrivers : ResolvedDriversNoFollow)
            .get(Cmd.front(), [&, this] {
              return resolveDriver(Cmd.front(), FollowSymlink, ClangPath);
            });
  }
}

CommandMangler::operator clang::tooling::ArgumentsAdjuster() && {
  // ArgumentsAdjuster is a std::function and so must be copyable.
  return [Mangler = std::make_shared<CommandMangler>(std::move(*this))](
             const std::vector<std::string> &Args, llvm::StringRef File) {
    auto Result = Args;
    Mangler->adjust(Result, File);
    return Result;
  };
}

// ArgStripper implementation
namespace {

// Determine total number of args consumed by this option.
// Return answers for {Exact, Prefix} match. 0 means not allowed.
std::pair<unsigned, unsigned> getArgCount(const llvm::opt::Option &Opt) {
  constexpr static unsigned Rest = 10000; // Should be all the rest!
  // Reference is llvm::opt::Option::acceptInternal()
  using llvm::opt::Option;
  switch (Opt.getKind()) {
  case Option::FlagClass:
    return {1, 0};
  case Option::JoinedClass:
  case Option::CommaJoinedClass:
    return {1, 1};
  case Option::GroupClass:
  case Option::InputClass:
  case Option::UnknownClass:
  case Option::ValuesClass:
    return {1, 0};
  case Option::JoinedAndSeparateClass:
    return {2, 2};
  case Option::SeparateClass:
    return {2, 0};
  case Option::MultiArgClass:
    return {1 + Opt.getNumArgs(), 0};
  case Option::JoinedOrSeparateClass:
    return {2, 1};
  case Option::RemainingArgsClass:
    return {Rest, 0};
  case Option::RemainingArgsJoinedClass:
    return {Rest, Rest};
  }
  llvm_unreachable("Unhandled option kind");
}

// Flag-parsing mode, which affects which flags are available.
enum DriverMode : unsigned char {
  DM_None = 0,
  DM_GCC = 1, // Default mode e.g. when invoked as 'clang'
  DM_CL = 2,  // MS CL.exe compatible mode e.g. when invoked as 'clang-cl'
  DM_CC1 = 4, // When invoked as 'clang -cc1' or after '-Xclang'
  DM_All = 7
};

// Examine args list to determine if we're in GCC, CL-compatible, or cc1 mode.
DriverMode getDriverMode(const std::vector<std::string> &Args) {
  DriverMode Mode = DM_GCC;
  llvm::StringRef Argv0 = Args.front();
  if (Argv0.endswith_insensitive(".exe"))
    Argv0 = Argv0.drop_back(strlen(".exe"));
  if (Argv0.endswith_insensitive("cl"))
    Mode = DM_CL;
  for (const llvm::StringRef Arg : Args) {
    if (Arg == "--driver-mode=cl") {
      Mode = DM_CL;
      break;
    }
    if (Arg == "-cc1") {
      Mode = DM_CC1;
      break;
    }
  }
  return Mode;
}

// Returns the set of DriverModes where an option may be used.
unsigned char getModes(const llvm::opt::Option &Opt) {
  // Why is this so complicated?!
  // Reference is clang::driver::Driver::getIncludeExcludeOptionFlagMasks()
  unsigned char Result = DM_None;
  if (Opt.hasFlag(driver::options::CC1Option))
    Result |= DM_CC1;
  if (!Opt.hasFlag(driver::options::NoDriverOption)) {
    if (Opt.hasFlag(driver::options::CLOption)) {
      Result |= DM_CL;
    } else {
      Result |= DM_GCC;
      if (Opt.hasFlag(driver::options::CoreOption)) {
        Result |= DM_CL;
      }
    }
  }
  return Result;
}

} // namespace

llvm::ArrayRef<ArgStripper::Rule> ArgStripper::rulesFor(llvm::StringRef Arg) {
  // All the hard work is done once in a static initializer.
  // We compute a table containing strings to look for and #args to skip.
  // e.g. "-x" => {-x 2 args, -x* 1 arg, --language 2 args, --language=* 1 arg}
  using TableTy =
      llvm::StringMap<llvm::SmallVector<Rule, 4>, llvm::BumpPtrAllocator>;
  static TableTy *Table = [] {
    auto &DriverTable = driver::getDriverOptTable();
    using DriverID = clang::driver::options::ID;

    // Collect sets of aliases, so we can treat -foo and -foo= as synonyms.
    // Conceptually a double-linked list: PrevAlias[I] -> I -> NextAlias[I].
    // If PrevAlias[I] is INVALID, then I is canonical.
    DriverID PrevAlias[DriverID::LastOption] = {DriverID::OPT_INVALID};
    DriverID NextAlias[DriverID::LastOption] = {DriverID::OPT_INVALID};
    auto AddAlias = [&](DriverID Self, DriverID T) {
      if (NextAlias[T]) {
        PrevAlias[NextAlias[T]] = Self;
        NextAlias[Self] = NextAlias[T];
      }
      PrevAlias[Self] = T;
      NextAlias[T] = Self;
    };
    // Also grab prefixes for each option, these are not fully exposed.
    const char *const *Prefixes[DriverID::LastOption] = {nullptr};
#define PREFIX(NAME, VALUE) static const char *const NAME[] = VALUE;
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELP, METAVAR, VALUES)                                          \
  Prefixes[DriverID::OPT_##ID] = PREFIX;
#include "clang/Driver/Options.inc"
#undef OPTION
#undef PREFIX

    struct {
      DriverID ID;
      DriverID AliasID;
      const void *AliasArgs;
    } AliasTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELP, METAVAR, VALUES)                                          \
  {DriverID::OPT_##ID, DriverID::OPT_##ALIAS, ALIASARGS},
#include "clang/Driver/Options.inc"
#undef OPTION
    };
    for (auto &E : AliasTable)
      if (E.AliasID != DriverID::OPT_INVALID && E.AliasArgs == nullptr)
        AddAlias(E.ID, E.AliasID);

    auto Result = std::make_unique<TableTy>();
    // Iterate over distinct options (represented by the canonical alias).
    // Every spelling of this option will get the same set of rules.
    for (unsigned ID = 1 /*Skip INVALID */; ID < DriverID::LastOption; ++ID) {
      if (PrevAlias[ID] || ID == DriverID::OPT_Xclang)
        continue; // Not canonical, or specially handled.
      llvm::SmallVector<Rule> Rules;
      // Iterate over each alias, to add rules for parsing it.
      for (unsigned A = ID; A != DriverID::OPT_INVALID; A = NextAlias[A]) {
        if (Prefixes[A] == nullptr) // option groups.
          continue;
        auto Opt = DriverTable.getOption(A);
        // Exclude - and -foo pseudo-options.
        if (Opt.getName().empty())
          continue;
        auto Modes = getModes(Opt);
        std::pair<unsigned, unsigned> ArgCount = getArgCount(Opt);
        // Iterate over each spelling of the alias, e.g. -foo vs --foo.
        for (auto *Prefix = Prefixes[A]; *Prefix != nullptr; ++Prefix) {
          llvm::SmallString<64> Buf(*Prefix);
          Buf.append(Opt.getName());
          llvm::StringRef Spelling = Result->try_emplace(Buf).first->getKey();
          Rules.emplace_back();
          Rule &R = Rules.back();
          R.Text = Spelling;
          R.Modes = Modes;
          R.ExactArgs = ArgCount.first;
          R.PrefixArgs = ArgCount.second;
          // Concrete priority is the index into the option table.
          // Effectively, earlier entries take priority over later ones.
          assert(ID < std::numeric_limits<decltype(R.Priority)>::max() &&
                 "Rules::Priority overflowed by options table");
          R.Priority = ID;
        }
      }
      // Register the set of rules under each possible name.
      for (const auto &R : Rules)
        Result->find(R.Text)->second.append(Rules.begin(), Rules.end());
    }
#ifndef NDEBUG
    // Dump the table and various measures of its size.
    unsigned RuleCount = 0;
    dlog("ArgStripper Option spelling table");
    for (const auto &Entry : *Result) {
      dlog("{0}", Entry.first());
      RuleCount += Entry.second.size();
      for (const auto &R : Entry.second)
        dlog("  {0} #={1} *={2} Mode={3}", R.Text, R.ExactArgs, R.PrefixArgs,
             int(R.Modes));
    }
    dlog("Table spellings={0} rules={1} string-bytes={2}", Result->size(),
         RuleCount, Result->getAllocator().getBytesAllocated());
#endif
    // The static table will never be destroyed.
    return Result.release();
  }();

  auto It = Table->find(Arg);
  return (It == Table->end()) ? llvm::ArrayRef<Rule>() : It->second;
}

void ArgStripper::strip(llvm::StringRef Arg) {
  auto OptionRules = rulesFor(Arg);
  if (OptionRules.empty()) {
    // Not a recognized flag. Strip it literally.
    Storage.emplace_back(Arg);
    Rules.emplace_back();
    Rules.back().Text = Storage.back();
    Rules.back().ExactArgs = 1;
    if (Rules.back().Text.consume_back("*"))
      Rules.back().PrefixArgs = 1;
    Rules.back().Modes = DM_All;
    Rules.back().Priority = -1; // Max unsigned = lowest priority.
  } else {
    Rules.append(OptionRules.begin(), OptionRules.end());
  }
}

const ArgStripper::Rule *ArgStripper::matchingRule(llvm::StringRef Arg,
                                                   unsigned Mode,
                                                   unsigned &ArgCount) const {
  const ArgStripper::Rule *BestRule = nullptr;
  for (const Rule &R : Rules) {
    // Rule can fail to match if...
    if (!(R.Modes & Mode))
      continue; // not applicable to current driver mode
    if (BestRule && BestRule->Priority < R.Priority)
      continue; // lower-priority than best candidate.
    if (!Arg.startswith(R.Text))
      continue; // current arg doesn't match the prefix string
    bool PrefixMatch = Arg.size() > R.Text.size();
    // Can rule apply as an exact/prefix match?
    if (unsigned Count = PrefixMatch ? R.PrefixArgs : R.ExactArgs) {
      BestRule = &R;
      ArgCount = Count;
    }
    // Continue in case we find a higher-priority rule.
  }
  return BestRule;
}

void ArgStripper::process(std::vector<std::string> &Args) const {
  if (Args.empty())
    return;

  // We're parsing the args list in some mode (e.g. gcc-compatible) but may
  // temporarily switch to another mode with the -Xclang flag.
  DriverMode MainMode = getDriverMode(Args);
  DriverMode CurrentMode = MainMode;

  // Read and write heads for in-place deletion.
  unsigned Read = 0, Write = 0;
  bool WasXclang = false;
  while (Read < Args.size()) {
    unsigned ArgCount = 0;
    if (matchingRule(Args[Read], CurrentMode, ArgCount)) {
      // Delete it and its args.
      if (WasXclang) {
        assert(Write > 0);
        --Write; // Drop previous -Xclang arg
        CurrentMode = MainMode;
        WasXclang = false;
      }
      // Advance to last arg. An arg may be foo or -Xclang foo.
      for (unsigned I = 1; Read < Args.size() && I < ArgCount; ++I) {
        ++Read;
        if (Read < Args.size() && Args[Read] == "-Xclang")
          ++Read;
      }
    } else {
      // No match, just copy the arg through.
      WasXclang = Args[Read] == "-Xclang";
      CurrentMode = WasXclang ? DM_CC1 : MainMode;
      if (Write != Read)
        Args[Write] = std::move(Args[Read]);
      ++Write;
    }
    ++Read;
  }
  Args.resize(Write);
}

std::string printArgv(llvm::ArrayRef<llvm::StringRef> Args) {
  std::string Buf;
  llvm::raw_string_ostream OS(Buf);
  bool Sep = false;
  for (llvm::StringRef Arg : Args) {
    if (Sep)
      OS << ' ';
    Sep = true;
    if (llvm::all_of(Arg, llvm::isPrint) &&
        Arg.find_first_of(" \t\n\"\\") == llvm::StringRef::npos) {
      OS << Arg;
      continue;
    }
    OS << '"';
    OS.write_escaped(Arg, /*UseHexEscapes=*/true);
    OS << '"';
  }
  return std::move(OS.str());
}

std::string printArgv(llvm::ArrayRef<std::string> Args) {
  std::vector<llvm::StringRef> Refs(Args.size());
  llvm::copy(Args, Refs.begin());
  return printArgv(Refs);
}

} // namespace clangd
} // namespace clang
