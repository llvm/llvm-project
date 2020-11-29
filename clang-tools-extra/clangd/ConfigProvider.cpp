//===--- ConfigProvider.cpp - Loading of user configuration ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConfigProvider.h"
#include "Config.h"
#include "ConfigFragment.h"
#include "support/FileCache.h"
#include "support/ThreadsafeFS.h"
#include "support/Trace.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include <chrono>
#include <mutex>
#include <string>

namespace clang {
namespace clangd {
namespace config {

// Threadsafe cache around reading a YAML config file from disk.
class FileConfigCache : public FileCache {
  mutable llvm::SmallVector<CompiledFragment, 1> CachedValue;
  std::string Directory;

public:
  FileConfigCache(llvm::StringRef Path, llvm::StringRef Directory)
      : FileCache(Path), Directory(Directory) {}

  void get(const ThreadsafeFS &TFS, DiagnosticCallback DC,
           std::chrono::steady_clock::time_point FreshTime,
           std::vector<CompiledFragment> &Out) const {
    read(
        TFS, FreshTime,
        [&](llvm::Optional<llvm::StringRef> Data) {
          CachedValue.clear();
          if (Data)
            for (auto &Fragment : Fragment::parseYAML(*Data, path(), DC)) {
              Fragment.Source.Directory = Directory;
              CachedValue.push_back(std::move(Fragment).compile(DC));
            }
        },
        [&]() { llvm::copy(CachedValue, std::back_inserter(Out)); });
  }
};

std::unique_ptr<Provider> Provider::fromYAMLFile(llvm::StringRef AbsPath,
                                                 llvm::StringRef Directory,
                                                 const ThreadsafeFS &FS) {
  class AbsFileProvider : public Provider {
    mutable FileConfigCache Cache; // threadsafe
    const ThreadsafeFS &FS;

    std::vector<CompiledFragment>
    getFragments(const Params &P, DiagnosticCallback DC) const override {
      std::vector<CompiledFragment> Result;
      Cache.get(FS, DC, P.FreshTime, Result);
      return Result;
    };

  public:
    AbsFileProvider(llvm::StringRef Path, llvm::StringRef Directory,
                    const ThreadsafeFS &FS)
        : Cache(Path, Directory), FS(FS) {
      assert(llvm::sys::path::is_absolute(Path));
    }
  };

  return std::make_unique<AbsFileProvider>(AbsPath, Directory, FS);
}

std::unique_ptr<Provider>
Provider::fromAncestorRelativeYAMLFiles(llvm::StringRef RelPath,
                                        const ThreadsafeFS &FS) {
  class RelFileProvider : public Provider {
    std::string RelPath;
    const ThreadsafeFS &FS;

    mutable std::mutex Mu;
    // Keys are the ancestor directory, not the actual config path within it.
    // We only insert into this map, so pointers to values are stable forever.
    // Mutex guards the map itself, not the values (which are threadsafe).
    mutable llvm::StringMap<FileConfigCache> Cache;

    std::vector<CompiledFragment>
    getFragments(const Params &P, DiagnosticCallback DC) const override {
      namespace path = llvm::sys::path;

      if (P.Path.empty())
        return {};

      // Compute absolute paths to all ancestors (substrings of P.Path).
      llvm::StringRef Parent = path::parent_path(P.Path);
      llvm::SmallVector<llvm::StringRef, 8> Ancestors;
      for (auto I = path::begin(Parent, path::Style::posix),
                E = path::end(Parent);
           I != E; ++I) {
        // Avoid weird non-substring cases like phantom "." components.
        // In practice, Component is a substring for all "normal" ancestors.
        if (I->end() < Parent.begin() || I->end() > Parent.end())
          continue;
        Ancestors.emplace_back(Parent.begin(), I->end() - Parent.begin());
      }
      // Ensure corresponding cache entries exist in the map.
      llvm::SmallVector<FileConfigCache *, 8> Caches;
      {
        std::lock_guard<std::mutex> Lock(Mu);
        for (llvm::StringRef Ancestor : Ancestors) {
          auto It = Cache.find(Ancestor);
          // Assemble the actual config file path only once.
          if (It == Cache.end()) {
            llvm::SmallString<256> ConfigPath = Ancestor;
            path::append(ConfigPath, RelPath);
            It = Cache.try_emplace(Ancestor, ConfigPath.str(), Ancestor).first;
          }
          Caches.push_back(&It->second);
        }
      }
      // Finally query each individual file.
      // This will take a (per-file) lock for each file that actually exists.
      std::vector<CompiledFragment> Result;
      for (FileConfigCache *Cache : Caches)
        Cache->get(FS, DC, P.FreshTime, Result);
      return Result;
    };

  public:
    RelFileProvider(llvm::StringRef RelPath, const ThreadsafeFS &FS)
        : RelPath(RelPath), FS(FS) {
      assert(llvm::sys::path::is_relative(RelPath));
    }
  };

  return std::make_unique<RelFileProvider>(RelPath, FS);
}

std::unique_ptr<Provider>
Provider::combine(std::vector<const Provider *> Providers) {
  struct CombinedProvider : Provider {
    std::vector<const Provider *> Providers;

    std::vector<CompiledFragment>
    getFragments(const Params &P, DiagnosticCallback DC) const override {
      std::vector<CompiledFragment> Result;
      for (const auto &Provider : Providers) {
        for (auto &Fragment : Provider->getFragments(P, DC))
          Result.push_back(std::move(Fragment));
      }
      return Result;
    }
  };
  auto Result = std::make_unique<CombinedProvider>();
  Result->Providers = std::move(Providers);
  // FIXME: This is a workaround for a bug in older versions of clang (< 3.9)
  //   The constructor that is supposed to allow for Derived to Base
  //   conversion does not work. Remove this if we drop support for such
  //   configurations.
  return std::unique_ptr<Provider>(Result.release());
}

Config Provider::getConfig(const Params &P, DiagnosticCallback DC) const {
  trace::Span Tracer("getConfig");
  if (!P.Path.empty())
    SPAN_ATTACH(Tracer, "path", P.Path);
  Config C;
  for (const auto &Fragment : getFragments(P, DC))
    Fragment(P, C);
  return C;
}

} // namespace config
} // namespace clangd
} // namespace clang
