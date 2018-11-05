//===-- profile_collector_test.cc -----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a function call tracing system.
//
//===----------------------------------------------------------------------===//
#include "gtest/gtest.h"

#include "xray_profile_collector.h"
#include "xray_profiling_flags.h"
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace __xray {
namespace {

static constexpr auto kHeaderSize = 16u;

constexpr uptr ExpectedProfilingVersion = 0x20180424;

struct ExpectedProfilingFileHeader {
  const u64 MagicBytes = 0x7872617970726f66; // Identifier for XRay profiling
                                             // files 'xrayprof' in hex.
  const u64 Version = ExpectedProfilingVersion;
  u64 Timestamp = 0;
  u64 PID = 0;
};

void ValidateFileHeaderBlock(XRayBuffer B) {
  ASSERT_NE(static_cast<const void *>(B.Data), nullptr);
  ASSERT_EQ(B.Size, sizeof(ExpectedProfilingFileHeader));
  typename std::aligned_storage<sizeof(ExpectedProfilingFileHeader)>::type
      FileHeaderStorage;
  ExpectedProfilingFileHeader ExpectedHeader;
  std::memcpy(&FileHeaderStorage, B.Data, B.Size);
  auto &FileHeader =
      *reinterpret_cast<ExpectedProfilingFileHeader *>(&FileHeaderStorage);
  ASSERT_EQ(ExpectedHeader.MagicBytes, FileHeader.MagicBytes);
  ASSERT_EQ(ExpectedHeader.Version, FileHeader.Version);
}

void ValidateBlock(XRayBuffer B) {
  profilingFlags()->setDefaults();
  ASSERT_NE(static_cast<const void *>(B.Data), nullptr);
  ASSERT_NE(B.Size, 0u);
  ASSERT_GE(B.Size, kHeaderSize);
  // We look at the block size, the block number, and the thread ID to ensure
  // that none of them are zero (or that the header data is laid out as we
  // expect).
  char LocalBuffer[kHeaderSize] = {};
  internal_memcpy(LocalBuffer, B.Data, kHeaderSize);
  u32 BlockSize = 0;
  u32 BlockNumber = 0;
  u64 ThreadId = 0;
  internal_memcpy(&BlockSize, LocalBuffer, sizeof(u32));
  internal_memcpy(&BlockNumber, LocalBuffer + sizeof(u32), sizeof(u32));
  internal_memcpy(&ThreadId, LocalBuffer + (2 * sizeof(u32)), sizeof(u64));
  ASSERT_NE(BlockSize, 0u);
  ASSERT_GE(BlockNumber, 0u);
  ASSERT_NE(ThreadId, 0u);
}

std::tuple<u32, u32, u64> ParseBlockHeader(XRayBuffer B) {
  char LocalBuffer[kHeaderSize] = {};
  internal_memcpy(LocalBuffer, B.Data, kHeaderSize);
  u32 BlockSize = 0;
  u32 BlockNumber = 0;
  u64 ThreadId = 0;
  internal_memcpy(&BlockSize, LocalBuffer, sizeof(u32));
  internal_memcpy(&BlockNumber, LocalBuffer + sizeof(u32), sizeof(u32));
  internal_memcpy(&ThreadId, LocalBuffer + (2 * sizeof(u32)), sizeof(u64));
  return std::make_tuple(BlockSize, BlockNumber, ThreadId);
}

struct Profile {
  int64_t CallCount;
  int64_t CumulativeLocalTime;
  std::vector<int32_t> Path;
};

std::tuple<Profile, const char *> ParseProfile(const char *P) {
  Profile Result;
  // Read the path first, until we find a sentinel 0.
  int32_t F;
  do {
    internal_memcpy(&F, P, sizeof(int32_t));
    P += sizeof(int32_t);
    Result.Path.push_back(F);
  } while (F != 0);

  // Then read the CallCount.
  internal_memcpy(&Result.CallCount, P, sizeof(int64_t));
  P += sizeof(int64_t);

  // Then read the CumulativeLocalTime.
  internal_memcpy(&Result.CumulativeLocalTime, P, sizeof(int64_t));
  P += sizeof(int64_t);
  return std::make_tuple(std::move(Result), P);
}

TEST(profileCollectorServiceTest, PostSerializeCollect) {
  profilingFlags()->setDefaults();
  // The most basic use-case (the one we actually only care about) is the one
  // where we ensure that we can post FunctionCallTrie instances, which are then
  // destroyed but serialized properly.
  //
  // First, we initialise a set of allocators in the local scope. This ensures
  // that we're able to copy the contents of the FunctionCallTrie that uses
  // the local allocators.
  auto Allocators = FunctionCallTrie::InitAllocators();
  FunctionCallTrie T(Allocators);

  // Then, we populate the trie with some data.
  T.enterFunction(1, 1, 0);
  T.enterFunction(2, 2, 0);
  T.exitFunction(2, 3, 0);
  T.exitFunction(1, 4, 0);

  // Then we post the data to the global profile collector service.
  profileCollectorService::post(T, 1);

  // Then we serialize the data.
  profileCollectorService::serialize();

  // Then we go through two buffers to see whether we're getting the data we
  // expect. The first block must always be as large as a file header, which
  // will have a fixed size.
  auto B = profileCollectorService::nextBuffer({nullptr, 0});
  ValidateFileHeaderBlock(B);

  B = profileCollectorService::nextBuffer(B);
  ValidateBlock(B);
  u32 BlockSize;
  u32 BlockNum;
  u64 ThreadId;
  std::tie(BlockSize, BlockNum, ThreadId) = ParseBlockHeader(B);

  // We look at the serialized buffer to see whether the Trie we're expecting
  // to see is there.
  auto DStart = static_cast<const char *>(B.Data) + kHeaderSize;
  std::vector<char> D(DStart, DStart + BlockSize);
  B = profileCollectorService::nextBuffer(B);
  ASSERT_EQ(B.Data, nullptr);
  ASSERT_EQ(B.Size, 0u);

  Profile Profile1, Profile2;
  auto P = static_cast<const char *>(D.data());
  std::tie(Profile1, P) = ParseProfile(P);
  std::tie(Profile2, P) = ParseProfile(P);

  ASSERT_NE(Profile1.Path.size(), Profile2.Path.size());
  auto &P1 = Profile1.Path.size() < Profile2.Path.size() ? Profile2 : Profile1;
  auto &P2 = Profile1.Path.size() < Profile2.Path.size() ? Profile1 : Profile2;
  std::vector<int32_t> P1Expected = {2, 1, 0};
  std::vector<int32_t> P2Expected = {1, 0};
  ASSERT_EQ(P1.Path.size(), P1Expected.size());
  ASSERT_EQ(P2.Path.size(), P2Expected.size());
  ASSERT_EQ(P1.Path, P1Expected);
  ASSERT_EQ(P2.Path, P2Expected);
}

// We break out a function that will be run in multiple threads, one that will
// use a thread local allocator, and will post the FunctionCallTrie to the
// profileCollectorService. This simulates what the threads being profiled would
// be doing anyway, but through the XRay logging implementation.
void threadProcessing() {
  thread_local auto Allocators = FunctionCallTrie::InitAllocators();
  FunctionCallTrie T(Allocators);

  T.enterFunction(1, 1, 0);
  T.enterFunction(2, 2, 0);
  T.exitFunction(2, 3, 0);
  T.exitFunction(1, 4, 0);

  profileCollectorService::post(T, GetTid());
}

TEST(profileCollectorServiceTest, PostSerializeCollectMultipleThread) {
  profilingFlags()->setDefaults();
  std::thread t1(threadProcessing);
  std::thread t2(threadProcessing);

  t1.join();
  t2.join();

  // At this point, t1 and t2 are already done with what they were doing.
  profileCollectorService::serialize();

  // Ensure that we see two buffers.
  auto B = profileCollectorService::nextBuffer({nullptr, 0});
  ValidateFileHeaderBlock(B);

  B = profileCollectorService::nextBuffer(B);
  ValidateBlock(B);

  B = profileCollectorService::nextBuffer(B);
  ValidateBlock(B);
}

} // namespace
} // namespace __xray
