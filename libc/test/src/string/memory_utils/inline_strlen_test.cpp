//===-- Unit tests for the vectorised string search loops -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/string_length.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

// The vector loops read whole aligned vectors around the string, so they read
// outside the array on purpose. These tests call them on local arrays that the
// compiler can see through once the call is inlined, which is where a read
// outside the object used to give a wrong answer. They only fail in an
// optimised build, and under LTO when the array is filled in another file.

namespace {

// Fills the array in a separate function so the stores come right before the
// call and are dropped if the compiler does not see them being read.
template <size_t N> [[gnu::noinline]] void fill(char (&a)[N]) {
  for (size_t i = 0; i + 1 < N; ++i)
    a[i] = static_cast<char>('a' + i % 26);
  a[N - 1] = '\0';
}

template <size_t (*Length)(const char *),
          void *(*Find)(const unsigned char *, unsigned char, size_t)>
struct Impl {
  template <size_t N> static void check_array() {
    char a[N];
    fill(a);
    for (size_t offset = 0; offset < N; ++offset) {
      ASSERT_EQ(Length(a + offset), N - 1 - offset);
      if (N - 1 - offset == 0)
        continue;
      if constexpr (N >= 2) {
        // The last character of the string, which may also occur earlier.
        size_t n = N - 1 - offset;
        size_t first = offset;
        while (a[first] != a[N - 2])
          ++first;
        void *found = Find(reinterpret_cast<unsigned char *>(a) + offset,
                           static_cast<unsigned char>(a[N - 2]), n);
        ASSERT_TRUE(found == a + first);
        // Neither a character that is absent nor one just past the range.
        ASSERT_TRUE(Find(reinterpret_cast<unsigned char *>(a) + offset, '0',
                         n) == nullptr);
        a[N - 1] = '0';
        ASSERT_TRUE(Find(reinterpret_cast<unsigned char *>(a) + offset, '0',
                         n) == nullptr);
        a[N - 1] = '\0';
      }
    }
  }

  static void run() {
    check_array<1>();
    check_array<2>();
    check_array<7>();
    check_array<16>();
    check_array<17>();
    check_array<31>();
    check_array<32>();
    check_array<33>();
    check_array<63>();
    check_array<64>();
    check_array<65>();
    check_array<100>();
  }
};

} // namespace

TEST(LlvmLibcInlineStrlenTest, Element) {
  Impl<internal::element::string_length,
       internal::element::find_first_character>::run();
}

TEST(LlvmLibcInlineStrlenTest, Word) {
  Impl<internal::word::string_length,
       internal::word::find_first_character>::run();
}

#if LIBC_HAS_VECTOR_TYPE
TEST(LlvmLibcInlineStrlenTest, ClangVector) {
  Impl<clang_vector::string_length, clang_vector::find_first_character>::run();
}
#endif

#if defined(LIBC_TARGET_ARCH_IS_X86)
TEST(LlvmLibcInlineStrlenTest, ArchVector) {
  Impl<internal::arch_vector::string_length,
       internal::arch_vector::find_first_character>::run();
}
#endif

} // namespace LIBC_NAMESPACE_DECL
