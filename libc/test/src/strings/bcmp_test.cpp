//===-- Unittests for bcmp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/os.h" // LIBC_TARGET_OS_IS_LINUX
#include "src/strings/bcmp.h"
#include "test/UnitTest/Test.h"
#include "test/UnitTest/TestLogger.h"
#include "test/src/string/memory_utils/memory_check_utils.h"

#if !defined(LIBC_FULL_BUILD) && defined(LIBC_TARGET_OS_IS_LINUX)
#include "test/src/string/memory_utils/protected_pages.h"
#endif // !defined(LIBC_FULL_BUILD) && defined(LIBC_TARGET_OS_IS_LINUX)

namespace LIBC_NAMESPACE_DECL {

TEST(LlvmLibcBcmpTest, CmpZeroByte) {
  const char *lhs = "ab";
  const char *rhs = "bc";
  ASSERT_EQ(LIBC_NAMESPACE::bcmp(lhs, rhs, 0), 0);
}

TEST(LlvmLibcBcmpTest, LhsRhsAreTheSame) {
  const char *lhs = "ab";
  const char *rhs = "ab";
  ASSERT_EQ(LIBC_NAMESPACE::bcmp(lhs, rhs, 2), 0);
}

TEST(LlvmLibcBcmpTest, LhsBeforeRhsLexically) {
  const char *lhs = "ab";
  const char *rhs = "ac";
  ASSERT_NE(LIBC_NAMESPACE::bcmp(lhs, rhs, 2), 0);
}

TEST(LlvmLibcBcmpTest, LhsAfterRhsLexically) {
  const char *lhs = "ac";
  const char *rhs = "ab";
  ASSERT_NE(LIBC_NAMESPACE::bcmp(lhs, rhs, 2), 0);
}

// Adapt CheckBcmp signature to bcmp.
static inline int Adaptor(cpp::span<char> p1, cpp::span<char> p2, size_t size) {
  return LIBC_NAMESPACE::bcmp(p1.begin(), p2.begin(), size);
}

TEST(LlvmLibcBcmpTest, SizeSweep) {
  static constexpr size_t kMaxSize = 600;
  Buffer Buffer1(kMaxSize);
  Buffer Buffer2(kMaxSize);
  Randomize(Buffer1.span());
  for (size_t size = 0; size <= kMaxSize; ++size) {
    auto span1 = Buffer1.span().subspan(0, size);
    auto span2 = Buffer2.span().subspan(0, size);
    const bool OK = CheckBcmp<Adaptor>(span1, span2, size);
    if (!OK)
      testing::tlog << "Failed at size=" << size << '\n';
    ASSERT_TRUE(OK);
  }
}

static inline void check_flipped_byte(const char *a, char *b, size_t size,
                                      size_t idx) {
  b[idx] ^= 1;
  ASSERT_NE(LIBC_NAMESPACE::bcmp(a, b, size), 0);
  ASSERT_NE(LIBC_NAMESPACE::bcmp(b, a, size), 0);
  b[idx] ^= 1;
}

// Offsets lhs across every alignment modulo 64 so aligned loops and unaligned
// tails are exercised for all size classes.
TEST(LlvmLibcBcmpTest, AlignmentSweep) {
  static constexpr size_t MAX_OFFSET = 64;
  static constexpr size_t MAX_SIZE = 600;
  Buffer buffer1(MAX_OFFSET + MAX_SIZE);
  Buffer buffer2(MAX_SIZE);
  Randomize(buffer1.span());

  auto check_size = [&](size_t size) {
    for (size_t offset = 0; offset < MAX_OFFSET; ++offset) {
      auto span1 = buffer1.span().subspan(offset, size);
      auto span2 = buffer2.span().subspan(0, size);
      if (size <= 72) {
        ASSERT_TRUE(CheckBcmp<Adaptor>(span1, span2, size));
        ASSERT_EQ(LIBC_NAMESPACE::bcmp(span2.data(), span1.data(), size), 0);
      } else {
        ReferenceCopy(span2, span1);
        ASSERT_EQ(LIBC_NAMESPACE::bcmp(span1.data(), span2.data(), size), 0);
        ASSERT_EQ(LIBC_NAMESPACE::bcmp(span2.data(), span1.data(), size), 0);
        const size_t positions[] = {0, 1, 31, 32, 63, 64, size / 2, size - 1};
        for (size_t i : positions)
          check_flipped_byte(span1.data(), span2.data(), size, i);
      }
    }
  };

  for (size_t size = 0; size <= 72; ++size)
    check_size(size);
  const size_t bases[] = {126, 254, 510};
  for (size_t base : bases)
    for (size_t size = base; size <= base + 8; ++size)
      check_size(size);
  check_size(MAX_SIZE);
}

// Verify that bytes outside [0, size) never affect the result.
TEST(LlvmLibcBcmpTest, IgnoresSurroundingBytes) {
  static constexpr size_t PAD = 64;
  static constexpr size_t MAX_SIZE = 64;
  char lhs[PAD + MAX_SIZE + PAD];
  char rhs[PAD + MAX_SIZE + PAD];

  for (size_t size = 0; size <= MAX_SIZE; ++size) {
    for (size_t i = 0; i < PAD; ++i) {
      lhs[i] = lhs[PAD + size + i] = 'x';
      rhs[i] = rhs[PAD + size + i] = 'y';
    }
    for (size_t i = 0; i < size; ++i)
      lhs[PAD + i] = rhs[PAD + i] = static_cast<char>('a' + (i % 26));

    ASSERT_EQ(LIBC_NAMESPACE::bcmp(lhs + PAD, rhs + PAD, size), 0);
    if (size == 0)
      continue;

    for (size_t i = 0; i < PAD; ++i)
      rhs[i] = rhs[PAD + size + i] = 'x';
    check_flipped_byte(lhs + PAD, rhs + PAD, size, size - 1);
    check_flipped_byte(lhs + PAD, rhs + PAD, size, 0);
  }
}

#if !defined(LIBC_FULL_BUILD) && defined(LIBC_TARGET_OS_IS_LINUX)

// Places buffers against inaccessible guard pages so out-of-page reads fault.
TEST(LlvmLibcBcmpTest, CheckAccess) {
  static constexpr size_t MAX_SIZE = 600;
  LIBC_ASSERT(MAX_SIZE < GetPageSize());
  ProtectedPages pages;
  const Page page_a = pages.GetPageA().WithAccess(PROT_READ | PROT_WRITE);
  const Page page_b = pages.GetPageB().WithAccess(PROT_READ | PROT_WRITE);
  const size_t page_size = page_a.page_size;
  Randomize(
      cpp::span<char>(reinterpret_cast<char *>(page_a.page_ptr), page_size));

  const void *inaccessible[] = {
      nullptr,
      page_a.page_ptr - 1,
      page_a.page_ptr + page_size,
  };
  for (const void *bad : inaccessible) {
    ASSERT_EQ(LIBC_NAMESPACE::bcmp(bad, bad, 0), 0);
    ASSERT_EQ(LIBC_NAMESPACE::bcmp(page_a.page_ptr, bad, 0), 0);
    ASSERT_EQ(LIBC_NAMESPACE::bcmp(bad, page_a.page_ptr, 0), 0);
  }

  for (size_t size = 1; size <= 64; ++size) {
    for (size_t offset = 0; offset < 16; ++offset) {
      cpp::span<char> rhs(reinterpret_cast<char *>(page_b.top(size) - offset),
                          size);
      uint8_t *a_ptrs[] = {page_a.bottom(size) + offset,
                           page_a.top(size) - offset};
      for (uint8_t *a_raw : a_ptrs) {
        cpp::span<char> lhs(reinterpret_cast<char *>(a_raw), size);
        ASSERT_TRUE(CheckBcmp<Adaptor>(lhs, rhs, size));
        ASSERT_EQ(LIBC_NAMESPACE::bcmp(rhs.data(), lhs.data(), size), 0);
      }
    }
  }

  for (size_t size = 65; size <= MAX_SIZE; ++size) {
    uint8_t *a_ptrs[] = {page_a.bottom(size), page_a.top(size)};
    uint8_t *b_ptrs[] = {page_b.bottom(size), page_b.top(size)};
    for (uint8_t *a_raw : a_ptrs) {
      for (uint8_t *b_raw : b_ptrs) {
        cpp::span<char> a(reinterpret_cast<char *>(a_raw), size);
        cpp::span<char> b(reinterpret_cast<char *>(b_raw), size);
        ReferenceCopy(b, a);
        ASSERT_EQ(LIBC_NAMESPACE::bcmp(a.data(), b.data(), size), 0);
        check_flipped_byte(a.data(), b.data(), size, 0);
        check_flipped_byte(a.data(), b.data(), size, size - 1);
      }
    }
  }
}

#endif // !defined(LIBC_FULL_BUILD) && defined(LIBC_TARGET_OS_IS_LINUX)

} // namespace LIBC_NAMESPACE_DECL
