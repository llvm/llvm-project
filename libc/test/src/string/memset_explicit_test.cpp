//===-- Unittests for memset_explicit -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memory_utils/memory_check_utils.h"
#include "src/string/memset_explicit.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE {

// Apply the same tests as memset

static inline void Adaptor(cpp::span<char> p1, uint8_t value, size_t size) {
  LIBC_NAMESPACE::memset_explicit(p1.begin(), value, size);
}

TEST(LlvmLibcmemsetExplicitTest, SizeSweep) {
  static constexpr size_t kMaxSize = 400;
  Buffer DstBuffer(kMaxSize);
  for (size_t size = 0; size < kMaxSize; ++size) {
    const char value = size % 10;
    auto dst = DstBuffer.span().subspan(0, size);
    ASSERT_TRUE((CheckMemset<Adaptor>(dst, value, size)));
  }
}

} // namespace LIBC_NAMESPACE
