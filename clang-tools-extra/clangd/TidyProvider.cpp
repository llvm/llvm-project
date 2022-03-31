//===--- TidyProvider.cpp - create options for running clang-tidy----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TidyProvider.h"
#include "../clang-tidy/ClangTidyModuleRegistry.h"
#include "Config.h"
#include "support/FileCache.h"
#include "support/Logger.h"
#include "support/Path.h"
#include "support/ThreadsafeFS.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include <memory>

namespace clang {
namespace clangd {
namespace {

// Access to config from a .clang-tidy file, caching IO and parsing.
class DotClangTidyCache : private FileCache {
  // We cache and expose shared_ptr to avoid copying the value on every lookup
  // when we're ultimately just going to pass it to mergeWith.
  mutable std::shared_ptr<const tidy::ClangTidyOptions> Value;

public:
  DotClangTidyCache(PathRef Path) : FileCache(Path) {}

  std::shared_ptr<const tidy::ClangTidyOptions>
  get(const ThreadsafeFS &TFS,
      std::chrono::steady_clock::time_point FreshTime) const {
    std::shared_ptr<const tidy::ClangTidyOptions> Result;
    read(
        TFS, FreshTime,
        [this](llvm::Optional<llvm::StringRef> Data) {
          Value.reset();
          if (Data && !Data->empty()) {
            tidy::DiagCallback Diagnostics = [](const llvm::SMDiagnostic &D) {
              switch (D.getKind()) {
              case llvm::SourceMgr::DK_Error:
                elog("tidy-config error at {0}:{1}:{2}: {3}", D.getFilename(),
                     D.getLineNo(), D.getColumnNo(), D.getMessage());
                break;
              case llvm::SourceMgr::DK_Warning:
                log("tidy-config warning at {0}:{1}:{2}: {3}", D.getFilename(),
                    D.getLineNo(), D.getColumnNo(), D.getMessage());
                break;
              case llvm::SourceMgr::DK_Note:
              case llvm::SourceMgr::DK_Remark:
                vlog("tidy-config note at {0}:{1}:{2}: {3}", D.getFilename(),
                     D.getLineNo(), D.getColumnNo(), D.getMessage());
                break;
              }
            };
            if (auto Parsed = tidy::parseConfigurationWithDiags(
                    llvm::MemoryBufferRef(*Data, path()), Diagnostics))
              Value = std::make_shared<const tidy::ClangTidyOptions>(
                  std::move(*Parsed));
            else
              elog("Error parsing clang-tidy configuration in {0}: {1}", path(),
                   Parsed.getError().message());
          }
        },
        [&]() { Result = Value; });
    return Result;
  }
};

// Access to combined config from .clang-tidy files governing a source file.
// Each config file is cached and the caches are shared for affected sources.
//
// FIXME: largely duplicates config::Provider::fromAncestorRelativeYAMLFiles.
// Potentially useful for compile_commands.json too. Extract?
class DotClangTidyTree {
  const ThreadsafeFS &FS;
  std::string RelPath;
  std::chrono::steady_clock::duration MaxStaleness;

  mutable std::mutex Mu;
  // Keys are the ancestor directory, not the actual config path within it.
  // We only insert into this map, so pointers to values are stable forever.
  // Mutex guards the map itself, not the values (which are threadsafe).
  mutable llvm::StringMap<DotClangTidyCache> Cache;

public:
  DotClangTidyTree(const ThreadsafeFS &FS)
      : FS(FS), RelPath(".clang-tidy"), MaxStaleness(std::chrono::seconds(5)) {}

  void apply(tidy::ClangTidyOptions &Result, PathRef AbsPath) {
    namespace path = llvm::sys::path;
    assert(path::is_absolute(AbsPath));

    // Compute absolute paths to all ancestors (substrings of P.Path).
    // Ensure cache entries for each ancestor exist in the map.
    llvm::SmallVector<DotClangTidyCache *> Caches;
    {
      std::lock_guard<std::mutex> Lock(Mu);
      for (auto Ancestor = absoluteParent(AbsPath); !Ancestor.empty();
           Ancestor = absoluteParent(Ancestor)) {
        auto It = Cache.find(Ancestor);
        // Assemble the actual config file path only if needed.
        if (It == Cache.end()) {
          llvm::SmallString<256> ConfigPath = Ancestor;
          path::append(ConfigPath, RelPath);
          It = Cache.try_emplace(Ancestor, ConfigPath.str()).first;
        }
        Caches.push_back(&It->second);
      }
    }
    // Finally query each individual file.
    // This will take a (per-file) lock for each file that actually exists.
    std::chrono::steady_clock::time_point FreshTime =
        std::chrono::steady_clock::now() - MaxStaleness;
    llvm::SmallVector<std::shared_ptr<const tidy::ClangTidyOptions>>
        OptionStack;
    for (const DotClangTidyCache *Cache : Caches)
      if (auto Config = Cache->get(FS, FreshTime)) {
        OptionStack.push_back(std::move(Config));
        if (!OptionStack.back()->InheritParentConfig.getValueOr(false))
          break;
      }
    unsigned Order = 1u;
    for (auto &Option : llvm::reverse(OptionStack))
      Result.mergeWith(*Option, Order++);
  }
};

} // namespace

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
                       "-bugprone-use-after-move",
                       // Alias for bugprone-use-after-moe.
                       "-hicpp-invalid-access-moved");

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
    const auto &CurTidyConfig = Config::current().Diagnostics.ClangTidy;
    if (!CurTidyConfig.Checks.empty())
      mergeCheckList(Opts.Checks, CurTidyConfig.Checks);

    for (const auto &CheckOption : CurTidyConfig.CheckOptions)
      Opts.CheckOptions.insert_or_assign(CheckOption.getKey(),
                                         tidy::ClangTidyOptions::ClangTidyValue(
                                             CheckOption.getValue(), 10000U));
  };
}

TidyProvider provideClangTidyFiles(ThreadsafeFS &TFS) {
  return [Tree = std::make_unique<DotClangTidyTree>(TFS)](
             tidy::ClangTidyOptions &Opts, llvm::StringRef Filename) {
    Tree->apply(Opts, Filename);
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

bool isRegisteredTidyCheck(llvm::StringRef Check) {
  assert(!Check.empty());
  assert(!Check.contains('*') && !Check.contains(',') &&
         "isRegisteredCheck doesn't support globs");
  assert(Check.ltrim().front() != '-');

  static const llvm::StringSet<llvm::BumpPtrAllocator> AllChecks = [] {
    llvm::StringSet<llvm::BumpPtrAllocator> Result;
    tidy::ClangTidyCheckFactories Factories;
    for (tidy::ClangTidyModuleRegistry::entry E :
         tidy::ClangTidyModuleRegistry::entries())
      E.instantiate()->addCheckFactories(Factories);
    for (const auto &Factory : Factories)
      Result.insert(Factory.getKey());
    return Result;
  }();

  return AllChecks.contains(Check);
}
} // namespace clangd
} // namespace clang
