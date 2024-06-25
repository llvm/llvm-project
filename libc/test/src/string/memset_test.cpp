//===-- Unittests for memset ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memory_utils/memory_check_utils.h"
#include "src/__support/macros/properties/os.h" // LIBC_TARGET_OS_IS_LINUX
#include "src/string/memset.h"
#include "test/UnitTest/Test.h"

#if !defined(LIBC_FULL_BUILD) && defined(LIBC_TARGET_OS_IS_LINUX)
#include "memory_utils/protected_pages.h"
#endif // !defined(LIBC_FULL_BUILD) && defined(LIBC_TARGET_OS_IS_LINUX)

namespace LIBC_NAMESPACE {

// Adapt CheckMemset signature to memset.
static inline void Adaptor(cpp::span<char> p1, uint8_t value, size_t size) {
  LIBC_NAMESPACE::memset(p1.begin(), value, size);
}

TEST(LlvmLibcMemsetTest, SizeSweep) {
  static constexpr size_t kMaxSize = 400;
  Buffer DstBuffer(kMaxSize);
  for (size_t size = 0; size < kMaxSize; ++size) {
    const char value = size % 10;
    auto dst = DstBuffer.span().subspan(0, size);
    ASSERT_TRUE((CheckMemset<Adaptor>(dst, value, size)));
  }
}

#if !defined(LIBC_FULL_BUILD) && defined(LIBC_TARGET_OS_IS_LINUX)

TEST(LlvmLibcMemsetTest, CheckAccess) {
  static constexpr size_t MAX_SIZE = 1024;
  LIBC_ASSERT(MAX_SIZE < GetPageSize());
  ProtectedPages pages;
  const Page write_buffer = pages.GetPageA().WithAccess(PROT_WRITE);
  const cpp::array<int, 2> fill_chars = {0, 0x7F};
  for (int fill_char : fill_chars) {
    for (size_t size = 0; size < MAX_SIZE; ++size) {
      // We cross-check the function with two destinations.
      // - The first of them (bottom) is always page aligned and faults when
      //   accessing bytes before it.
      // - The second one (top) is not necessarily aligned and faults when
      //   accessing bytes after it.
      uint8_t *destinations[2] = {write_buffer.bottom(size),
                                  write_buffer.top(size)};
      for (uint8_t *dst : destinations) {
        LIBC_NAMESPACE::memset(dst, fill_char, size);
      }
    }
  }
}

#endif // !defined(LIBC_FULL_BUILD) && defined(LIBC_TARGET_OS_IS_LINUX)

} // namespace LIBC_NAMESPACE
