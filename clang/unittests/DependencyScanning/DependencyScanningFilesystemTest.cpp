//===- DependencyScanningFilesystemTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/DependencyScanning/DependencyScanningFilesystem.h"
#include "clang/DependencyScanning/DependencyScanningService.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"
#include <condition_variable>
#include <mutex>
#include <thread>

using namespace clang::dependencies;

namespace {

/// Releases all waiting threads simultaneously so that the worker logic can
/// be observed under maximal concurrency rather than a thread-spawn cascade.
struct StartBarrier {
  std::mutex M;
  std::condition_variable CV;
  bool Released = false;

  void wait() {
    std::unique_lock<std::mutex> Lock(M);
    CV.wait(Lock, [&] { return Released; });
  }

  void release() {
    {
      std::lock_guard<std::mutex> Lock(M);
      Released = true;
    }
    CV.notify_all();
  }
};

/// Build \p NumWorkers DependencyScanningWorkerFilesystems sharing \p Service
/// and \p FS, fan out one thread per worker waiting on a common barrier, then
/// release the barrier and join. Returns each worker's result (initialized to
/// \p Default for slots a thread did not write to).
template <typename R, typename Fn>
std::vector<R>
runConcurrentWorkers(DependencyScanningService &Service,
                     llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
                     unsigned NumWorkers, R Default, Fn &&PerWorker) {
  std::vector<std::unique_ptr<DependencyScanningWorkerFilesystem>> Workers;
  Workers.reserve(NumWorkers);
  for (unsigned I = 0; I < NumWorkers; ++I)
    Workers.push_back(
        std::make_unique<DependencyScanningWorkerFilesystem>(Service, FS));

  StartBarrier Barrier;
  std::vector<R> Results(NumWorkers, std::move(Default));
  std::vector<std::thread> Threads;
  Threads.reserve(NumWorkers);
  for (unsigned I = 0; I < NumWorkers; ++I) {
    Threads.emplace_back([&, I] {
      Barrier.wait();
      Results[I] = PerWorker(*Workers[I], I);
    });
  }
  Barrier.release();
  for (auto &T : Threads)
    T.join();
  return Results;
}

} // namespace

TEST(DependencyScanningFilesystem, OpenFileAndGetBufferRepeatedly) {
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMemoryFS->setCurrentWorkingDirectory("/");
  InMemoryFS->addFile("/foo", 0, llvm::MemoryBuffer::getMemBuffer("content"));

  DependencyScanningService Service({});
  DependencyScanningWorkerFilesystem DepFS(Service, InMemoryFS);

  auto FileOrErr1 = DepFS.openFileForRead("foo");
  auto FileOrErr2 = DepFS.openFileForRead("foo");
  ASSERT_EQ(FileOrErr1.getError(), std::error_code{});
  ASSERT_EQ(FileOrErr1.getError(), std::error_code{});
  std::unique_ptr<llvm::vfs::File> File1 = std::move(*FileOrErr1);
  std::unique_ptr<llvm::vfs::File> File2 = std::move(*FileOrErr2);
  ASSERT_NE(File1, nullptr);
  ASSERT_NE(File2, nullptr);
  auto BufOrErr11 = File1->getBuffer("buf11");
  auto BufOrErr12 = File1->getBuffer("buf12");
  auto BufOrErr21 = File1->getBuffer("buf21");
  ASSERT_EQ(BufOrErr11.getError(), std::error_code{});
  ASSERT_EQ(BufOrErr12.getError(), std::error_code{});
  ASSERT_EQ(BufOrErr21.getError(), std::error_code{});
  std::unique_ptr<llvm::MemoryBuffer> Buf11 = std::move(*BufOrErr11);
  std::unique_ptr<llvm::MemoryBuffer> Buf12 = std::move(*BufOrErr12);
  std::unique_ptr<llvm::MemoryBuffer> Buf21 = std::move(*BufOrErr21);
  ASSERT_NE(Buf11, nullptr);
  ASSERT_NE(Buf12, nullptr);
  ASSERT_NE(Buf21, nullptr);
  ASSERT_EQ(Buf11->getBuffer().data(), Buf12->getBuffer().data());
  ASSERT_EQ(Buf11->getBuffer().data(), Buf21->getBuffer().data());
  EXPECT_EQ(Buf11->getBuffer(), "content");
}

TEST(DependencyScanningWorkerFilesystem, CacheStatusFailures) {
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();

  auto InstrumentingFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::TracingFileSystem>(InMemoryFS);

  DependencyScanningServiceOptions Opts;
  Opts.CacheNegativeStats = true;
  DependencyScanningService Service(std::move(Opts));
  DependencyScanningWorkerFilesystem DepFS(Service, InstrumentingFS);
  DependencyScanningWorkerFilesystem DepFS2(Service, InstrumentingFS);

  DepFS.status("/foo.c");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 1u);

  DepFS.status("/foo.c");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 1u); // Cached, no increase.

  DepFS.status("/bar.c");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 2u);

  DepFS2.status("/foo.c");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 2u); // Shared cache.
}

TEST(DependencyScanningFilesystem, CacheGetRealPath) {
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMemoryFS->setCurrentWorkingDirectory("/");
  InMemoryFS->addFile("/foo", 0, llvm::MemoryBuffer::getMemBuffer(""));
  InMemoryFS->addFile("/bar", 0, llvm::MemoryBuffer::getMemBuffer(""));

  auto InstrumentingFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::TracingFileSystem>(InMemoryFS);

  DependencyScanningService Service({});
  DependencyScanningWorkerFilesystem DepFS(Service, InstrumentingFS);
  DependencyScanningWorkerFilesystem DepFS2(Service, InstrumentingFS);

  {
    llvm::SmallString<128> Result;
    DepFS.getRealPath("/foo", Result);
    EXPECT_EQ(InstrumentingFS->NumGetRealPathCalls, 1u);
  }

  {
    llvm::SmallString<128> Result;
    DepFS.getRealPath("/foo", Result);
    EXPECT_EQ(InstrumentingFS->NumGetRealPathCalls, 1u); // Cached, no increase.
  }

  {
    llvm::SmallString<128> Result;
    DepFS.getRealPath("/bar", Result);
    EXPECT_EQ(InstrumentingFS->NumGetRealPathCalls, 2u);
  }

  {
    llvm::SmallString<128> Result;
    DepFS2.getRealPath("/foo", Result);
    EXPECT_EQ(InstrumentingFS->NumGetRealPathCalls, 2u); // Shared cache.
  }
}

TEST(DependencyScanningFilesystem, RealPathAndStatusInvariants) {
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMemoryFS->setCurrentWorkingDirectory("/");
  InMemoryFS->addFile("/foo.c", 0, llvm::MemoryBuffer::getMemBuffer(""));
  InMemoryFS->addFile("/bar.c", 0, llvm::MemoryBuffer::getMemBuffer(""));

  DependencyScanningService Service({});
  DependencyScanningWorkerFilesystem DepFS(Service, InMemoryFS);

  // Success.
  {
    DepFS.status("/foo.c");

    llvm::SmallString<128> Result;
    DepFS.getRealPath("/foo.c", Result);
  }
  {
    llvm::SmallString<128> Result;
    DepFS.getRealPath("/bar.c", Result);

    DepFS.status("/bar.c");
  }

  // Failure.
  {
    DepFS.status("/foo.m");

    llvm::SmallString<128> Result;
    DepFS.getRealPath("/foo.m", Result);
  }
  {
    llvm::SmallString<128> Result;
    DepFS.getRealPath("/bar.m", Result);

    DepFS.status("/bar.m");
  }

  // Failure without caching.
  {
    DepFS.status("/foo");

    llvm::SmallString<128> Result;
    DepFS.getRealPath("/foo", Result);
  }
  {
    llvm::SmallString<128> Result;
    DepFS.getRealPath("/bar", Result);

    DepFS.status("/bar");
  }
}

TEST(DependencyScanningFilesystem, CacheStatOnExists) {
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  auto InstrumentingFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::TracingFileSystem>(InMemoryFS);
  InMemoryFS->setCurrentWorkingDirectory("/");
  InMemoryFS->addFile("/foo", 0, llvm::MemoryBuffer::getMemBuffer(""));
  InMemoryFS->addFile("/bar", 0, llvm::MemoryBuffer::getMemBuffer(""));
  DependencyScanningService Service({});
  DependencyScanningWorkerFilesystem DepFS(Service, InstrumentingFS);

  DepFS.status("/foo");
  DepFS.status("/foo");
  DepFS.status("/bar");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 2u);

  DepFS.exists("/foo");
  DepFS.exists("/bar");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 2u);
  EXPECT_EQ(InstrumentingFS->NumExistsCalls, 0u);
}

TEST(DependencyScanningFilesystem, CacheStatFailures) {
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMemoryFS->setCurrentWorkingDirectory("/");
  InMemoryFS->addFile("/dir/present.h", 0,
                      llvm::MemoryBuffer::getMemBuffer(""));

  auto InstrumentingFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::TracingFileSystem>(InMemoryFS);

  DependencyScanningServiceOptions Opts;
  Opts.CacheNegativeStats = true;
  DependencyScanningService Service(std::move(Opts));
  DependencyScanningWorkerFilesystem DepFS(Service, InstrumentingFS);

  DepFS.status("/dir");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 1u);
  DepFS.status("/dir");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 1u);

  DepFS.status("/dir/present.h");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 2u);
  DepFS.status("/dir/present.h");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 2u);

  DepFS.status("/dir/missing.h");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 3u);
  DepFS.status("/dir/missing.h");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 3u);
}

TEST(DependencyScanningFilesystem, NoNegativeCache) {
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMemoryFS->setCurrentWorkingDirectory("/");
  InMemoryFS->addFile("/dir/present.h", 0,
                      llvm::MemoryBuffer::getMemBuffer(""));

  auto InstrumentingFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::TracingFileSystem>(InMemoryFS);

  DependencyScanningServiceOptions Opts;
  Opts.CacheNegativeStats = false;
  DependencyScanningService Service(std::move(Opts));
  DependencyScanningWorkerFilesystem DepFS(Service, InstrumentingFS);

  DepFS.status("/dir");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 1u);
  DepFS.status("/dir");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 1u);

  DepFS.status("/dir/present.h");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 2u);
  DepFS.status("/dir/present.h");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 2u);

  DepFS.status("/dir/missing.h");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 3u);
  DepFS.status("/dir/missing.h");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 4u);
}

TEST(DependencyScanningFilesystem, DiagnoseStaleStatFailures) {
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMemoryFS->setCurrentWorkingDirectory("/");

  DependencyScanningServiceOptions Opts;
  Opts.CacheNegativeStats = true;
  DependencyScanningService Service(std::move(Opts));
  DependencyScanningWorkerFilesystem DepFS(Service, InMemoryFS);

  bool Path1Exists = DepFS.exists("/path1.suffix");
  ASSERT_EQ(Path1Exists, false);

  // Adding a file that has been stat-ed,
  InMemoryFS->addFile("/path1.suffix", 0, llvm::MemoryBuffer::getMemBuffer(""));
  Path1Exists = DepFS.exists("/path1.suffix");
  // Due to caching in SharedCache, path1 should not exist in
  // DepFS's eyes.
  ASSERT_EQ(Path1Exists, false);

  auto InvalidEntries =
      Service.getSharedCache().getOutOfDateEntries(*InMemoryFS);

  EXPECT_EQ(InvalidEntries.size(), 1u);
  ASSERT_STREQ("/path1.suffix", InvalidEntries[0].Path);
}

TEST(DependencyScanningFilesystem, DiagnoseCachedFileSizeChange) {
  auto InMemoryFS1 = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  auto InMemoryFS2 = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMemoryFS1->setCurrentWorkingDirectory("/");
  InMemoryFS2->setCurrentWorkingDirectory("/");

  DependencyScanningServiceOptions Opts;
  Opts.CacheNegativeStats = true;
  DependencyScanningService Service(std::move(Opts));
  DependencyScanningWorkerFilesystem DepFS(Service, InMemoryFS1);

  InMemoryFS1->addFile("/path1.suffix", 0,
                       llvm::MemoryBuffer::getMemBuffer(""));
  bool Path1Exists = DepFS.exists("/path1.suffix");
  ASSERT_EQ(Path1Exists, true);

  // Add a file to a new FS that has the same path but different content.
  InMemoryFS2->addFile("/path1.suffix", 1,
                       llvm::MemoryBuffer::getMemBuffer("        "));

  // Check against the new file system. InMemoryFS2 could be the underlying
  // physical system in the real world.
  auto InvalidEntries =
      Service.getSharedCache().getOutOfDateEntries(*InMemoryFS2);

  ASSERT_EQ(InvalidEntries.size(), 1u);
  ASSERT_STREQ("/path1.suffix", InvalidEntries[0].Path);
  auto SizeInfo = std::get_if<
      DependencyScanningFilesystemSharedCache::OutOfDateEntry::SizeChangedInfo>(
      &InvalidEntries[0].Info);
  ASSERT_TRUE(SizeInfo);
  ASSERT_EQ(SizeInfo->CachedSize, 0u);
  ASSERT_EQ(SizeInfo->ActualSize, 8u);
}

TEST(DependencyScanningFilesystem, DoNotDiagnoseDirSizeChange) {
  llvm::SmallString<128> Dir;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("tmp", Dir));

  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS =
      llvm::vfs::createPhysicalFileSystem();

  DependencyScanningService Service({});
  DependencyScanningWorkerFilesystem DepFS(Service, FS);

  // Trigger the file system cache.
  ASSERT_EQ(DepFS.exists(Dir), true);

  // Add a file to the FS to change its size.
  // It seems that directory sizes reported are not meaningful,
  // and should not be used to check for size changes.
  // This test is setup only to trigger a size change so that we
  // know we are excluding directories from reporting.
  llvm::SmallString<128> FilePath = Dir;
  llvm::sys::path::append(FilePath, "file.h");
  {
    std::error_code EC;
    llvm::raw_fd_ostream TempFile(FilePath, EC);
    ASSERT_FALSE(EC);
  }

  // We do not report directory size changes.
  auto InvalidEntries = Service.getSharedCache().getOutOfDateEntries(*FS);
  EXPECT_EQ(InvalidEntries.size(), 0u);
}

TEST(DependencyScanningWorkerFilesystem, ConcurrentSameFilenameDeduplicates) {
  llvm::unittest::TempDir TmpDir("dswf-same-filename", /*Unique=*/true);
  llvm::unittest::TempFile TmpFile(TmpDir.path("foo.c"), /*Suffix=*/"",
                                   /*Contents=*/"hello");

  auto TracingFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::AtomicTracingFileSystem>(
          llvm::vfs::getRealFileSystem());
  DependencyScanningService Service({});

  constexpr unsigned NumWorkers = 16;
  auto Results = runConcurrentWorkers<llvm::ErrorOr<EntryRef>>(
      Service, TracingFS, NumWorkers, std::error_code{},
      [&](DependencyScanningWorkerFilesystem &W, unsigned) {
        return W.getOrCreateFileSystemEntry(TmpFile.path());
      });

  EXPECT_EQ(TracingFS->NumStatusCalls.load(), 1u);
  EXPECT_EQ(TracingFS->NumOpenFileForReadCalls.load(), 1u);

  // All workers must have observed the same underlying entry.
  ASSERT_TRUE(Results[0]);
  llvm::sys::fs::UniqueID FirstUID = Results[0]->getStatus().getUniqueID();
  const char *FirstContents = Results[0]->getContents().data();
  for (unsigned I = 0; I < NumWorkers; ++I) {
    ASSERT_TRUE(Results[I]) << "worker " << I << " failed";
    EXPECT_EQ(Results[I]->getStatus().getUniqueID(), FirstUID);
    EXPECT_EQ(Results[I]->getContents().data(), FirstContents);
  }
}

// On Windows, llvm::sys::fs::UniqueID is computed from a hashed canonical
// path, so two hard-linked filenames produce different UniqueIDs and the
// premise of this test does not hold.
#ifndef _WIN32
TEST(DependencyScanningWorkerFilesystem,
     ConcurrentSameUIDDifferentFilenamesDeduplicatesOpen) {
  // Use a real on-disk file plus a hard link so the two filenames share a
  // UniqueID, exercising the per-UID slot.
  llvm::unittest::TempDir TmpDir("dswf-same-uid", /*Unique=*/true);
  llvm::SmallString<128> RealPath = TmpDir.path("real.c");
  llvm::SmallString<128> AliasPath = TmpDir.path("alias.c");
  llvm::unittest::TempFile TmpFile(RealPath, /*Suffix=*/"", /*Contents=*/"hi");
  llvm::unittest::TempLink TmpLink(RealPath, AliasPath);

  auto TracingFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::AtomicTracingFileSystem>(
          llvm::vfs::getRealFileSystem());
  DependencyScanningService Service({});

  constexpr unsigned NumWorkers = 16;
  auto Results = runConcurrentWorkers<llvm::ErrorOr<EntryRef>>(
      Service, TracingFS, NumWorkers, std::error_code{},
      [&](DependencyScanningWorkerFilesystem &W, unsigned I) {
        llvm::StringRef Path = (I % 2 == 0) ? llvm::StringRef(RealPath)
                                            : llvm::StringRef(AliasPath);
        return W.getOrCreateFileSystemEntry(Path);
      });

  // Each filename's slot dedupes its own stats; with two filenames we expect
  // at most two stats. The UID slot then collapses the actual reads.
  EXPECT_LE(TracingFS->NumStatusCalls.load(), 2u);
  EXPECT_EQ(TracingFS->NumOpenFileForReadCalls.load(), 1u);

  ASSERT_TRUE(Results[0]);
  llvm::sys::fs::UniqueID FirstUID = Results[0]->getStatus().getUniqueID();
  for (unsigned I = 0; I < NumWorkers; ++I) {
    ASSERT_TRUE(Results[I]) << "worker " << I << " failed";
    EXPECT_EQ(Results[I]->getStatus().getUniqueID(), FirstUID);
  }
}
#endif // !_WIN32

TEST(DependencyScanningWorkerFilesystem, ConcurrentNegativeStatDeduplicates) {
  // Construct a path inside a temporary directory but never create the
  // file, so concurrent stat() calls land on the negative-stat path through
  // the real filesystem.
  llvm::unittest::TempDir TmpDir("dswf-negative", /*Unique=*/true);
  llvm::SmallString<128> MissingPath = TmpDir.path("missing.h");

  auto TracingFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::AtomicTracingFileSystem>(
          llvm::vfs::getRealFileSystem());
  // The dedup under test only applies when negative stats are cached; enable
  // caching and use a path whose extension is eligible for it.
  DependencyScanningServiceOptions Opts;
  Opts.CacheNegativeStats = true;
  DependencyScanningService Service(std::move(Opts));

  constexpr unsigned NumWorkers = 16;
  auto Results = runConcurrentWorkers<llvm::ErrorOr<llvm::vfs::Status>>(
      Service, TracingFS, NumWorkers, std::error_code{},
      [&](DependencyScanningWorkerFilesystem &W, unsigned) {
        return W.status(MissingPath);
      });

  EXPECT_EQ(TracingFS->NumStatusCalls.load(), 1u);
  EXPECT_EQ(TracingFS->NumOpenFileForReadCalls.load(), 0u);

  // Every worker observed the same negative result.
  ASSERT_FALSE(Results[0]);
  std::error_code FirstError = Results[0].getError();
  for (unsigned I = 0; I < NumWorkers; ++I) {
    ASSERT_FALSE(Results[I]) << "worker " << I << " unexpectedly succeeded";
    EXPECT_EQ(Results[I].getError(), FirstError);
  }
}

TEST(DependencyScanningWorkerFilesystem,
     ConcurrentUncachedNegativeStatIsSharedButNotPersisted) {
  // Same as above, but with negative-stat caching disabled. Concurrent queries
  // that overlap the producer's in-flight stat still adopt its negative
  // result (sharing an answer for overlapping queries is not "caching"), so no
  // worker opens the missing file. Because the result is not persisted, a
  // later, separate query must re-run the stat.
  llvm::unittest::TempDir TmpDir("dswf-uncached-negative", /*Unique=*/true);
  llvm::SmallString<128> MissingPath = TmpDir.path("missing.h");

  auto TracingFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::AtomicTracingFileSystem>(
          llvm::vfs::getRealFileSystem());
  DependencyScanningServiceOptions Opts;
  Opts.CacheNegativeStats = false;
  DependencyScanningService Service(std::move(Opts));

  constexpr unsigned NumWorkers = 16;
  auto Results = runConcurrentWorkers<llvm::ErrorOr<llvm::vfs::Status>>(
      Service, TracingFS, NumWorkers, std::error_code{},
      [&](DependencyScanningWorkerFilesystem &W, unsigned) {
        return W.status(MissingPath);
      });

  // A negative stat never opens the file, regardless of how many workers
  // produced versus adopted. Producers are bounded by the worker count;
  // adopters add none. With caching off the result is not persisted, so the
  // count is not pinned to one, but it can never exceed one stat per worker.
  EXPECT_EQ(TracingFS->NumOpenFileForReadCalls.load(), 0u);
  unsigned StatsAfterBurst = TracingFS->NumStatusCalls.load();
  EXPECT_GE(StatsAfterBurst, 1u);
  EXPECT_LE(StatsAfterBurst, NumWorkers);

  // Every worker observed the same negative result.
  ASSERT_FALSE(Results[0]);
  std::error_code FirstError = Results[0].getError();
  for (unsigned I = 0; I < NumWorkers; ++I) {
    ASSERT_FALSE(Results[I]) << "worker " << I << " unexpectedly succeeded";
    EXPECT_EQ(Results[I].getError(), FirstError);
  }

  // The uncached negative was shared, not cached: a later, separate query
  // re-runs the stat rather than adopting a persisted miss.
  DependencyScanningWorkerFilesystem PostWorker(Service, TracingFS);
  EXPECT_FALSE(PostWorker.status(MissingPath));
  EXPECT_EQ(TracingFS->NumStatusCalls.load(), StatsAfterBurst + 1);
}
