//===-- Unittests for memcpy ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memory_utils/memory_check_utils.h"
#include "src/__support/macros/properties/os.h" // LIBC_TARGET_OS_IS_LINUX
#include "src/string/memcpy.h"
#include "test/UnitTest/Test.h"

#if !defined(LIBC_FULL_BUILD) && defined(LIBC_TARGET_OS_IS_LINUX)
#include "memory_utils/protected_pages.h"
#endif // !defined(LIBC_FULL_BUILD) && defined(LIBC_TARGET_OS_IS_LINUX)

namespace LIBC_NAMESPACE {

// Adapt CheckMemcpy signature to memcpy.
static inline void Adaptor(cpp::span<char> dst, cpp::span<char> src,
                           size_t size) {
  LIBC_NAMESPACE::memcpy(dst.begin(), src.begin(), size);
}

TEST(LlvmLibcMemcpyTest, SizeSweep) {
  static constexpr size_t kMaxSize = 400;
  Buffer SrcBuffer(kMaxSize);
  Buffer DstBuffer(kMaxSize);
  Randomize(SrcBuffer.span());
  for (size_t size = 0; size < kMaxSize; ++size) {
    auto src = SrcBuffer.span().subspan(0, size);
    auto dst = DstBuffer.span().subspan(0, size);
    ASSERT_TRUE(CheckMemcpy<Adaptor>(dst, src, size));
  }
}

#if !defined(LIBC_FULL_BUILD) && defined(LIBC_TARGET_OS_IS_LINUX)

TEST(LlvmLibcMemcpyTest, CheckAccess) {
  static constexpr size_t MAX_SIZE = 1024;
  LIBC_ASSERT(MAX_SIZE < GetPageSize());
  ProtectedPages pages;
  const Page write_buffer = pages.GetPageA().WithAccess(PROT_WRITE);
  const Page read_buffer = [&]() {
    // We fetch page B in write mode.
    auto page = pages.GetPageB().WithAccess(PROT_WRITE);
    // And fill it with random numbers.
    for (size_t i = 0; i < page.page_size; ++i)
      page.page_ptr[i] = rand();
    // Then return it in read mode.
    return page.WithAccess(PROT_READ);
  }();
  for (size_t size = 0; size < MAX_SIZE; ++size) {
    // We cross-check the function with two sources and two destinations.
    //  - The first of them (bottom) is always page aligned and faults when
    //    accessing bytes before it.
    //  - The second one (top) is not necessarily aligned and faults when
    //    accessing bytes after it.
    const uint8_t *sources[2] = {read_buffer.bottom(size),
                                 read_buffer.top(size)};
    uint8_t *destinations[2] = {write_buffer.bottom(size),
                                write_buffer.top(size)};
    for (const uint8_t *src : sources) {
      for (uint8_t *dst : destinations) {
        LIBC_NAMESPACE::memcpy(dst, src, size);
      }
    }
  }
}

#endif // !defined(LIBC_FULL_BUILD) && defined(LIBC_TARGET_OS_IS_LINUX)

} // namespace LIBC_NAMESPACE
