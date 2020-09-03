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
#include "support/ThreadsafeFS.h"
#include "support/Trace.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Path.h"
#include <chrono>
#include <mutex>

namespace clang {
namespace clangd {
namespace config {

// Threadsafe cache around reading a YAML config file from disk.
class FileConfigCache {
  std::mutex Mu;
  std::chrono::steady_clock::time_point ValidTime = {};
  llvm::SmallVector<CompiledFragment, 1> CachedValue;
  llvm::sys::TimePoint<> MTime = {};
  unsigned Size = -1;

  // Called once we are sure we want to read the file.
  // REQUIRES: Cache keys are set. Mutex must be held.
  void fillCacheFromDisk(llvm::vfs::FileSystem &FS, DiagnosticCallback DC) {
    CachedValue.clear();

    auto Buf = FS.getBufferForFile(Path);
    // If we failed to read (but stat succeeded), don't cache failure.
    if (!Buf) {
      Size = -1;
      MTime = {};
      return;
    }

    // If file changed between stat and open, we don't know its mtime.
    // For simplicity, don't cache the value in this case (use a bad key).
    if (Buf->get()->getBufferSize() != Size) {
      Size = -1;
      MTime = {};
    }

    // Finally parse and compile the actual fragments.
    for (auto &Fragment :
         Fragment::parseYAML(Buf->get()->getBuffer(), Path, DC))
      CachedValue.push_back(std::move(Fragment).compile(DC));
  }

public:
  // Must be set before the cache is used. Not a constructor param to allow
  // computing ancestor-relative paths to be deferred.
  std::string Path;

  // Retrieves up-to-date config fragments from disk.
  // A cached result may be reused if the mtime and size are unchanged.
  // (But several concurrent read()s can miss the cache after a single change).
  // Future performance ideas:
  // - allow caches to be reused based on short elapsed walltime
  // - allow latency-sensitive operations to skip revalidating the cache
  void read(const ThreadsafeFS &TFS, DiagnosticCallback DC,
            llvm::Optional<std::chrono::steady_clock::time_point> FreshTime,
            std::vector<CompiledFragment> &Out) {
    std::lock_guard<std::mutex> Lock(Mu);
    // We're going to update the cache and return whatever's in it.
    auto Return = llvm::make_scope_exit(
        [&] { llvm::copy(CachedValue, std::back_inserter(Out)); });

    // Return any sufficiently recent result without doing any further work.
    if (FreshTime && ValidTime >= FreshTime)
      return;

    // Ensure we bump the ValidTime at the end to allow for reuse.
    auto MarkTime = llvm::make_scope_exit(
        [&] { ValidTime = std::chrono::steady_clock::now(); });

    // Stat is cheaper than opening the file, it's usually unchanged.
    assert(llvm::sys::path::is_absolute(Path));
    auto FS = TFS.view(/*CWD=*/llvm::None);
    auto Stat = FS->status(Path);
    // If there's no file, the result is empty. Ensure we have an invalid key.
    if (!Stat || !Stat->isRegularFile()) {
      MTime = {};
      Size = -1;
      CachedValue.clear();
      return;
    }
    // If the modified-time and size match, assume the content does too.
    if (Size == Stat->getSize() && MTime == Stat->getLastModificationTime())
      return;

    // OK, the file has actually changed. Update cache key, compute new value.
    Size = Stat->getSize();
    MTime = Stat->getLastModificationTime();
    fillCacheFromDisk(*FS, DC);
  }
};

std::unique_ptr<Provider> Provider::fromYAMLFile(llvm::StringRef AbsPath,
                                                 const ThreadsafeFS &FS) {
  class AbsFileProvider : public Provider {
    mutable FileConfigCache Cache; // threadsafe
    const ThreadsafeFS &FS;

    std::vector<CompiledFragment>
    getFragments(const Params &P, DiagnosticCallback DC) const override {
      std::vector<CompiledFragment> Result;
      Cache.read(FS, DC, P.FreshTime, Result);
      return Result;
    };

  public:
    AbsFileProvider(llvm::StringRef Path, const ThreadsafeFS &FS) : FS(FS) {
      assert(llvm::sys::path::is_absolute(Path));
      Cache.Path = Path.str();
    }
  };

  return std::make_unique<AbsFileProvider>(AbsPath, FS);
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
        if (I->end() < Parent.begin() && I->end() > Parent.end())
          continue;
        Ancestors.emplace_back(Parent.begin(), I->end() - Parent.begin());
      }
      // Ensure corresponding cache entries exist in the map.
      llvm::SmallVector<FileConfigCache *, 8> Caches;
      {
        std::lock_guard<std::mutex> Lock(Mu);
        for (llvm::StringRef Ancestor : Ancestors) {
          auto R = Cache.try_emplace(Ancestor);
          // Assemble the actual config file path only once.
          if (R.second) {
            llvm::SmallString<256> ConfigPath = Ancestor;
            path::append(ConfigPath, RelPath);
            R.first->second.Path = ConfigPath.str().str();
          }
          Caches.push_back(&R.first->second);
        }
      }
      // Finally query each individual file.
      // This will take a (per-file) lock for each file that actually exists.
      std::vector<CompiledFragment> Result;
      for (FileConfigCache *Cache : Caches)
        Cache->read(FS, DC, P.FreshTime, Result);
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
