//===- DependencyScanningFilesystemTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningFilesystem.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gtest/gtest.h"

using namespace clang::tooling::dependencies;

TEST(DependencyScanningWorkerFilesystem, CacheStatusFailures) {
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();

  auto InstrumentingFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::TracingFileSystem>(InMemoryFS);

  DependencyScanningFilesystemSharedCache SharedCache;
  DependencyScanningWorkerFilesystem DepFS(SharedCache, InstrumentingFS);
  DependencyScanningWorkerFilesystem DepFS2(SharedCache, InstrumentingFS);

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

  DependencyScanningFilesystemSharedCache SharedCache;
  DependencyScanningWorkerFilesystem DepFS(SharedCache, InstrumentingFS);
  DependencyScanningWorkerFilesystem DepFS2(SharedCache, InstrumentingFS);

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

  DependencyScanningFilesystemSharedCache SharedCache;
  DependencyScanningWorkerFilesystem DepFS(SharedCache, InMemoryFS);

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
  DependencyScanningFilesystemSharedCache SharedCache;
  DependencyScanningWorkerFilesystem DepFS(SharedCache, InstrumentingFS);

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
  InMemoryFS->addFile("/dir/vector", 0, llvm::MemoryBuffer::getMemBuffer(""));
  InMemoryFS->addFile("/cache/a.pcm", 0, llvm::MemoryBuffer::getMemBuffer(""));

  auto InstrumentingFS =
      llvm::makeIntrusiveRefCnt<InstrumentingFilesystem>(InMemoryFS);

  DependencyScanningFilesystemSharedCache SharedCache;
  DependencyScanningWorkerFilesystem DepFS(SharedCache, InstrumentingFS);

  DepFS.status("/dir");
  DepFS.status("/dir");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 1u);

  DepFS.status("/dir/vector");
  DepFS.status("/dir/vector");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 2u);

  DepFS.setBypassedPathPrefix("/cache");
  DepFS.exists("/cache/a.pcm");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 3u);
  DepFS.exists("/cache/a.pcm");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 4u);

  DepFS.resetBypassedPathPrefix();
  DepFS.exists("/cache/a.pcm");
  DepFS.exists("/cache/a.pcm");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 5u);
}
