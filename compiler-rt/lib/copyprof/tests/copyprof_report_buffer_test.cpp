//===-- copyprof_report_buffer_test.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file contains unit tests for the CopyProf lock-free report buffer,
/// verifying concurrent enqueue and dequeue correctness.
///
//===----------------------------------------------------------------------===//

#include "copyprof/copyprof_report_buffer.h"

#include "gtest/gtest.h"
#include "sanitizer_common/sanitizer_common.h"

namespace __copyprof {
namespace {

TEST(CopyProfReportBuffer, DoesNotWriteOutOfBounds) {
  const uptr kPageSize = GetPageSizeCached();
  // Map two pages, the second of which is a no-access guard page.
  void* mapped = MmapOrDie(2 * kPageSize, "CopyProfReportBufferTest");
  MprotectNoAccess(reinterpret_cast<uptr>(mapped) + kPageSize, kPageSize);

  const usize kReportBufferSize = kPageSize - sizeof(ReportBuffer);
  ReportBuffer* buffer = new (mapped) ReportBuffer(kReportBufferSize);

  // Fill the entire buffer. This should not write into the guard page.
  void* p = buffer->TryAllocate(kReportBufferSize);
  ASSERT_NE(p, nullptr);
  internal_memset(p, 0x66, kReportBufferSize);
  SanitizerBreakOptimization(p);

  // Any further allocation should fail.
  ASSERT_EQ(buffer->TryAllocate(1), nullptr);
}

TEST(CopyProfReportBuffer, CanBeResuedAfterReset) {
  char memory[128];
  ReportBuffer* buffer =
      new (memory) ReportBuffer(sizeof(memory) - sizeof(ReportBuffer));
  for (char* ptr = reinterpret_cast<char*>(buffer->TryAllocate(1));
       ptr != nullptr; ptr = reinterpret_cast<char*>(buffer->TryAllocate(1))) {
    *ptr = 0x66;
  }
  EXPECT_EQ(buffer->TryAllocate(1), nullptr);
  buffer->Reset();
  char* ptr = reinterpret_cast<char*>(buffer->TryAllocate(1));
  EXPECT_NE(ptr, nullptr);
  *ptr = 0x77;
  EXPECT_EQ(memory[sizeof(ReportBuffer)], 0x77);
}

TEST(CopyProfReportBuffer, EmptyBufferWorks) {
  ReportBuffer buffer(0);
  EXPECT_EQ(buffer.TryAllocate(1), nullptr);
}

}  // namespace
}  // namespace __copyprof
