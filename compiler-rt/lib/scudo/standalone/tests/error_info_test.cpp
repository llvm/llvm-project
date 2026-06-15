//===-- error_info_test.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "allocator_config.h"
#include "combined.h"
#include "memtag.h"
#include "tests/scudo_unit_test.h"
#include <stdlib.h>

namespace scudo {

template <typename Config> struct TestAllocator : Allocator<Config> {
  TestAllocator() { this->initThreadMaybe(); }
  ~TestAllocator() { this->unmapTestOnly(); }
};

template <class TypeParam> struct ScudoErrorInfoTest : public Test {
  ScudoErrorInfoTest() {
    setenv("SCUDO_OPTIONS", "allocation_ring_buffer_size=32768", 1);
    Allocator = std::make_unique<AllocatorT>();
    Allocator->setTrackAllocationStacks(true);
  }

  ~ScudoErrorInfoTest() {
    Allocator->releaseToOS(scudo::ReleaseToOS::Force);
    unsetenv("SCUDO_OPTIONS");
  }

  using AllocatorT = TestAllocator<TypeParam>;
  std::unique_ptr<AllocatorT> Allocator;
};

struct TestConfig : public scudo::DefaultConfig {
  static const bool MaySupportMemoryTagging = true;
};

using ScudoErrorInfoTestTypes = ::testing::Types<TestConfig>;
TYPED_TEST_SUITE(ScudoErrorInfoTest, ScudoErrorInfoTestTypes);

TYPED_TEST(ScudoErrorInfoTest, RingBufferErrorInfo) {
  auto *Allocator = this->Allocator.get();
  if (!scudo::archSupportsMemoryTagging() ||
      !Allocator->useMemoryTaggingTestOnly()) {
    GTEST_SKIP() << "MTE not supported or enabled";
  }

  EXPECT_GT(Allocator->getRingBufferSize(), 0u);
  EXPECT_NE(nullptr, Allocator->getRingBufferAddress());

  const scudo::uptr Size = 64U;
  void *P = nullptr;
  P = Allocator->allocate(Size, Chunk::Origin::Malloc);
  ASSERT_NE(P, nullptr);
  Allocator->deallocate(P, Chunk::Origin::Malloc);

  scudo::uptr Ptr = reinterpret_cast<scudo::uptr>(P);

  scudo_error_info ErrorInfo = {};
  size_t ReportIndex = 0;
  Allocator->getRingBufferErrorInfo(Ptr, &ErrorInfo.reports[0], ReportIndex);

  EXPECT_EQ(ReportIndex, 1U);
  EXPECT_EQ(ErrorInfo.reports[0].error_type, USE_AFTER_FREE);
  EXPECT_EQ(ErrorInfo.reports[0].allocation_address, scudo::untagPointer(Ptr));
  EXPECT_EQ(ErrorInfo.reports[0].allocation_size, Size);

  // Now verify the ReportIndex is followed.
  memset(&ErrorInfo, 0, sizeof(ErrorInfo));
  Allocator->getRingBufferErrorInfo(Ptr, &ErrorInfo.reports[0], ReportIndex);
  EXPECT_EQ(ReportIndex, 2U);
  EXPECT_EQ(ErrorInfo.reports[0].error_type, UNKNOWN);
  EXPECT_EQ(ErrorInfo.reports[1].error_type, USE_AFTER_FREE);
  EXPECT_EQ(ErrorInfo.reports[1].allocation_address, scudo::untagPointer(Ptr));
  EXPECT_EQ(ErrorInfo.reports[1].allocation_size, Size);

  // Verify if at max, nothing happens.
  ReportIndex = 3;
  memset(&ErrorInfo, 0, sizeof(ErrorInfo));
  Allocator->getRingBufferErrorInfo(Ptr, &ErrorInfo.reports[0], ReportIndex);
  EXPECT_EQ(ReportIndex, 3U);
  EXPECT_EQ(ErrorInfo.reports[0].error_type, UNKNOWN);
  EXPECT_EQ(ErrorInfo.reports[1].error_type, UNKNOWN);
  EXPECT_EQ(ErrorInfo.reports[2].error_type, UNKNOWN);
}

TYPED_TEST(ScudoErrorInfoTest, GetErrorInfoUAF) {
  auto *Allocator = this->Allocator.get();
  if (!scudo::archSupportsMemoryTagging() ||
      !Allocator->useMemoryTaggingTestOnly()) {
    GTEST_SKIP() << "MTE not supported or enabled";
  }

  const scudo::uptr Size = 64U;
  void *P = Allocator->allocate(Size, Chunk::Origin::Malloc);
  ASSERT_NE(P, nullptr);
  Allocator->deallocate(P, Chunk::Origin::Malloc);

  scudo::uptr Ptr = reinterpret_cast<scudo::uptr>(P);
  scudo_error_info ErrorInfo = {};
  Allocator->getErrorInfo(Ptr, &ErrorInfo);

  EXPECT_EQ(ErrorInfo.reports[0].error_type, USE_AFTER_FREE);
  EXPECT_EQ(ErrorInfo.reports[0].allocation_address, scudo::untagPointer(Ptr));
  EXPECT_EQ(ErrorInfo.reports[0].allocation_size, Size);
}

TYPED_TEST(ScudoErrorInfoTest, GetErrorInfoOverflow) {
  auto *Allocator = this->Allocator.get();
  if (!scudo::archSupportsMemoryTagging() ||
      !Allocator->useMemoryTaggingTestOnly()) {
    GTEST_SKIP() << "MTE not supported or enabled";
  }

  const scudo::uptr Size = 64U;
  void *P = Allocator->allocate(Size, Chunk::Origin::Malloc);
  ASSERT_NE(P, nullptr);

  scudo::uptr PtrAddr = reinterpret_cast<scudo::uptr>(P);
  scudo::uptr FaultAddr = PtrAddr + Size;
  scudo_error_info ErrorInfo = {};
  Allocator->getErrorInfo(FaultAddr, &ErrorInfo);

  EXPECT_EQ(ErrorInfo.reports[0].error_type, BUFFER_OVERFLOW);
  EXPECT_EQ(ErrorInfo.reports[0].allocation_address,
            scudo::untagPointer(PtrAddr));
  EXPECT_EQ(ErrorInfo.reports[0].allocation_size, Size);

  Allocator->deallocate(P, Chunk::Origin::Malloc);
}

TYPED_TEST(ScudoErrorInfoTest, GetErrorInfoUnderflow) {
  auto *Allocator = this->Allocator.get();
  if (!scudo::archSupportsMemoryTagging() ||
      !Allocator->useMemoryTaggingTestOnly()) {
    GTEST_SKIP() << "MTE not supported or enabled";
  }

  const scudo::uptr Size = 64U;
  void *P = Allocator->allocate(Size, Chunk::Origin::Malloc);
  ASSERT_NE(P, nullptr);

  scudo::uptr PtrAddr = reinterpret_cast<scudo::uptr>(P);
  scudo::uptr FaultAddr = PtrAddr - 1;
  scudo_error_info ErrorInfo = {};
  Allocator->getErrorInfo(FaultAddr, &ErrorInfo);

  EXPECT_EQ(ErrorInfo.reports[0].error_type, BUFFER_UNDERFLOW);
  EXPECT_EQ(ErrorInfo.reports[0].allocation_address,
            scudo::untagPointer(PtrAddr));
  EXPECT_EQ(ErrorInfo.reports[0].allocation_size, Size);

  Allocator->deallocate(P, Chunk::Origin::Malloc);
}

} // namespace scudo
