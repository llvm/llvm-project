//===--- TidyProvider.cpp - create options for running clang-tidy----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TidyProvider.h"
#include "Config.h"
#include "support/Logger.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <memory>

namespace clang {
namespace clangd {

static void mergeCheckList(llvm::Optional<std::string> &Checks,
                           llvm::StringRef List) {
  if (List.empty())
    return;
  if (!Checks || Checks->empty()) {
    Checks.emplace(List);
    return;
  }
  *Checks = llvm::join_items(",", *Checks, List);
}

static llvm::Optional<tidy::ClangTidyOptions>
tryReadConfigFile(llvm::vfs::FileSystem *FS, llvm::StringRef Directory) {
  assert(!Directory.empty());
  // We guaranteed that child directories of Directory exist, so this assert
  // should hopefully never fail.
  assert(FS->exists(Directory));

  llvm::SmallString<128> ConfigFile(Directory);
  llvm::sys::path::append(ConfigFile, ".clang-tidy");

  llvm::ErrorOr<llvm::vfs::Status> FileStatus = FS->status(ConfigFile);

  if (!FileStatus || !FileStatus->isRegularFile())
    return llvm::None;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Text =
      FS->getBufferForFile(ConfigFile);
  if (std::error_code EC = Text.getError()) {
    elog("Can't read '{0}': {1}", ConfigFile, EC.message());
    return llvm::None;
  }

  // Skip empty files, e.g. files opened for writing via shell output
  // redirection.
  if ((*Text)->getBuffer().empty())
    return llvm::None;
  llvm::ErrorOr<tidy::ClangTidyOptions> ParsedOptions =
      tidy::parseConfiguration((*Text)->getBuffer());
  if (!ParsedOptions) {
    if (ParsedOptions.getError())
      elog("Error parsing clang-tidy configuration in '{0}': {1}", ConfigFile,
           ParsedOptions.getError().message());
    return llvm::None;
  }
  return std::move(*ParsedOptions);
}

TidyProviderRef provideEnvironment() {
  static const llvm::Optional<std::string> User = [] {
    llvm::Optional<std::string> Ret = llvm::sys::Process::GetEnv("USER");
#ifdef _WIN32
    if (!Ret)
      return llvm::sys::Process::GetEnv("USERNAME");
#endif
    return Ret;
  }();

  if (User)
    return
        [](tidy::ClangTidyOptions &Opts, llvm::StringRef) { Opts.User = User; };
  // FIXME: Once function_ref and unique_function operator= operators handle
  // null values, this can return null.
  return [](tidy::ClangTidyOptions &, llvm::StringRef) {};
}

TidyProviderRef provideDefaultChecks() {
  // These default checks are chosen for:
  //  - low false-positive rate
  //  - providing a lot of value
  //  - being reasonably efficient
  static const std::string DefaultChecks = llvm::join_items(
      ",", "readability-misleading-indentation", "readability-deleted-default",
      "bugprone-integer-division", "bugprone-sizeof-expression",
      "bugprone-suspicious-missing-comma", "bugprone-unused-raii",
      "bugprone-unused-return-value", "misc-unused-using-decls",
      "misc-unused-alias-decls", "misc-definitions-in-headers");
  return [](tidy::ClangTidyOptions &Opts, llvm::StringRef) {
    if (!Opts.Checks || Opts.Checks->empty())
      Opts.Checks = DefaultChecks;
  };
}

TidyProvider addTidyChecks(llvm::StringRef Checks,
                           llvm::StringRef WarningsAsErrors) {
  return [Checks = std::string(Checks),
          WarningsAsErrors = std::string(WarningsAsErrors)](
             tidy::ClangTidyOptions &Opts, llvm::StringRef) {
    mergeCheckList(Opts.Checks, Checks);
    mergeCheckList(Opts.WarningsAsErrors, WarningsAsErrors);
  };
}

TidyProvider disableUnusableChecks(llvm::ArrayRef<std::string> ExtraBadChecks) {
  constexpr llvm::StringLiteral Seperator(",");
  static const std::string BadChecks =
      llvm::join_items(Seperator,
                       // We want this list to start with a seperator to
                       // simplify appending in the lambda. So including an
                       // empty string here will force that.
                       "",
                       // ----- False Positives -----

                       // Check relies on seeing ifndef/define/endif directives,
                       // clangd doesn't replay those when using a preamble.
                       "-llvm-header-guard",

                       // ----- Crashing Checks -----

                       // Check can choke on invalid (intermediate) c++
                       // code, which is often the case when clangd
                       // tries to build an AST.
                       "-bugprone-use-after-move");

  size_t Size = BadChecks.size();
  for (const std::string &Str : ExtraBadChecks) {
    if (Str.empty())
      continue;
    Size += Seperator.size();
    if (LLVM_LIKELY(Str.front() != '-'))
      ++Size;
    Size += Str.size();
  }
  std::string DisableGlob;
  DisableGlob.reserve(Size);
  DisableGlob += BadChecks;
  for (const std::string &Str : ExtraBadChecks) {
    if (Str.empty())
      continue;
    DisableGlob += Seperator;
    if (LLVM_LIKELY(Str.front() != '-'))
      DisableGlob.push_back('-');
    DisableGlob += Str;
  }

  return [DisableList(std::move(DisableGlob))](tidy::ClangTidyOptions &Opts,
                                               llvm::StringRef) {
    if (Opts.Checks && !Opts.Checks->empty())
      Opts.Checks->append(DisableList);
  };
}

TidyProviderRef provideClangdConfig() {
  return [](tidy::ClangTidyOptions &Opts, llvm::StringRef) {
    const auto &CurTidyConfig = Config::current().ClangTidy;
    if (!CurTidyConfig.Checks.empty())
      mergeCheckList(Opts.Checks, CurTidyConfig.Checks);

    for (const auto &CheckOption : CurTidyConfig.CheckOptions)
      Opts.CheckOptions.insert_or_assign(CheckOption.getKey(),
                                         tidy::ClangTidyOptions::ClangTidyValue(
                                             CheckOption.getValue(), 10000U));
  };
}

TidyProvider provideClangTidyFiles(ThreadsafeFS &TFS) {
  return [&TFS](tidy::ClangTidyOptions &Opts, llvm::StringRef Filename) {
    llvm::SmallVector<tidy::ClangTidyOptions, 4> OptionStack;
    auto FS(TFS.view(llvm::None));
    llvm::SmallString<256> AbsolutePath(Filename);

    assert(llvm::sys::path::is_absolute(AbsolutePath));

    llvm::sys::path::remove_dots(AbsolutePath, true);
    llvm::StringRef Directory = llvm::sys::path::parent_path(AbsolutePath);
    {
      auto Status = FS->status(Directory);

      if (!Status || !Status->isDirectory()) {
        elog("Error reading configuration from {0}: directory doesn't exist",
             Directory);
        return;
      }
    }

    // FIXME: Store options in a cache that validates itself against changes
    // during the clangd session.
    for (llvm::StringRef CurrentDirectory = Directory;
         !CurrentDirectory.empty();
         CurrentDirectory = llvm::sys::path::parent_path(CurrentDirectory)) {
      auto ConfigFile = tryReadConfigFile(FS.get(), CurrentDirectory);
      if (!ConfigFile)
        continue;
      OptionStack.push_back(std::move(*ConfigFile));
      // Should we search for a parent config to merge
      if (!OptionStack.back().InheritParentConfig.getValueOr(false))
        break;
    }
    unsigned Order = 1U;
    for (auto &Option : llvm::reverse(OptionStack))
      Opts.mergeWith(Option, Order++);
  };
}

TidyProvider combine(std::vector<TidyProvider> Providers) {
  // FIXME: Once function_ref and unique_function operator= operators handle
  // null values, we should filter out any Providers that are null. Right now we
  // have to ensure we dont pass any providers that are null.
  return [Providers(std::move(Providers))](tidy::ClangTidyOptions &Opts,
                                           llvm::StringRef Filename) {
    for (const auto &Provider : Providers)
      Provider(Opts, Filename);
  };
}

tidy::ClangTidyOptions getTidyOptionsForFile(TidyProviderRef Provider,
                                             llvm::StringRef Filename) {
  tidy::ClangTidyOptions Opts = tidy::ClangTidyOptions::getDefaults();
  Opts.Checks->clear();
  if (Provider)
    Provider(Opts, Filename);
  return Opts;
}
} // namespace clangd
} // namespace clang
