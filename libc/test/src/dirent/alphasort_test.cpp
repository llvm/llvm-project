//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for the POSIX alphasort function.
///
//===----------------------------------------------------------------------===//

#include "src/dirent/alphasort.h"

#include "hdr/types/struct_dirent.h"
#include "src/stdlib/qsort.h"
#include "test/UnitTest/Test.h"

namespace {

template <size_t N = 64> struct MockDirent {
  alignas(struct dirent) char buf[sizeof(struct dirent) + N]{};

  const struct dirent *create(const char *name) {
    auto *d = reinterpret_cast<struct dirent *>(buf);
    char *dst = d->d_name;
    while (*name)
      *dst++ = *name++;
    *dst = '\0';
    return d;
  }
};

} // namespace

TEST(LlvmLibcAlphasortTest, BasicComparison) {
  MockDirent<> ent_a{};
  MockDirent<> ent_b{};
  const struct dirent *a = ent_a.create("apple");
  const struct dirent *b = ent_b.create("banana");

  EXPECT_LT(LIBC_NAMESPACE::alphasort(&a, &b), 0);
  EXPECT_GT(LIBC_NAMESPACE::alphasort(&b, &a), 0);
}

TEST(LlvmLibcAlphasortTest, EqualNames) {
  MockDirent<> ent_1{};
  MockDirent<> ent_2{};
  const struct dirent *d1 = ent_1.create("file.txt");
  const struct dirent *d2 = ent_2.create("file.txt");

  EXPECT_EQ(LIBC_NAMESPACE::alphasort(&d1, &d2), 0);
  EXPECT_EQ(LIBC_NAMESPACE::alphasort(&d1, &d1), 0);
}

TEST(LlvmLibcAlphasortTest, PrefixComparison) {
  MockDirent<> ent_short{};
  MockDirent<> ent_long{};
  const struct dirent *d_short = ent_short.create("file");
  const struct dirent *d_long = ent_long.create("file.txt");

  EXPECT_LT(LIBC_NAMESPACE::alphasort(&d_short, &d_long), 0);
  EXPECT_GT(LIBC_NAMESPACE::alphasort(&d_long, &d_short), 0);
}

TEST(LlvmLibcAlphasortTest, EmptyName) {
  MockDirent<> ent_empty{};
  MockDirent<> ent_nonempty{};
  const struct dirent *d_empty = ent_empty.create("");
  const struct dirent *d_nonempty = ent_nonempty.create("a");

  EXPECT_LT(LIBC_NAMESPACE::alphasort(&d_empty, &d_nonempty), 0);
  EXPECT_GT(LIBC_NAMESPACE::alphasort(&d_nonempty, &d_empty), 0);
  EXPECT_EQ(LIBC_NAMESPACE::alphasort(&d_empty, &d_empty), 0);
}

TEST(LlvmLibcAlphasortTest, QsortSorting) {
  MockDirent<> ent_delta{};
  MockDirent<> ent_alpha{};
  MockDirent<> ent_charlie{};
  MockDirent<> ent_bravo{};

  constexpr size_t NUM_ENTRIES = 4;
  const struct dirent *entries[NUM_ENTRIES] = {
      ent_delta.create("delta"),
      ent_alpha.create("alpha"),
      ent_charlie.create("charlie"),
      ent_bravo.create("bravo"),
  };

  using QsortComparator = int (*)(const void *, const void *);
  LIBC_NAMESPACE::qsort(
      entries, NUM_ENTRIES, sizeof(const struct dirent *),
      reinterpret_cast<QsortComparator>(LIBC_NAMESPACE::alphasort));

  const struct dirent *e0 = entries[0];
  const struct dirent *e1 = entries[1];
  const struct dirent *e2 = entries[2];
  const struct dirent *e3 = entries[3];

  EXPECT_LT(LIBC_NAMESPACE::alphasort(&e0, &e1), 0); // alpha < bravo
  EXPECT_LT(LIBC_NAMESPACE::alphasort(&e1, &e2), 0); // bravo < charlie
  EXPECT_LT(LIBC_NAMESPACE::alphasort(&e2, &e3), 0); // charlie < delta
}
