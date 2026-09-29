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
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/string_length.h"
#include "test/UnitTest/Test.h"

namespace {

struct MockDirent {
  static constexpr size_t EXTRA_NAME_LEN = 64;
  static constexpr size_t BUFFER_SIZE = sizeof(struct dirent) + EXTRA_NAME_LEN;

  alignas(struct dirent) char buf[BUFFER_SIZE]{};

  MockDirent(const char *name) {
    auto *d = reinterpret_cast<struct dirent *>(buf);
    size_t len = LIBC_NAMESPACE::internal::string_length(name);
    if (len > EXTRA_NAME_LEN)
      len = EXTRA_NAME_LEN;
    LIBC_NAMESPACE::inline_memcpy(d->d_name, name, len);
    d->d_name[len] = '\0';
  }

  const struct dirent *get() const {
    return reinterpret_cast<const struct dirent *>(buf);
  }
};

} // namespace

TEST(LlvmLibcAlphasortTest, BasicComparison) {
  MockDirent ent_a("apple");
  MockDirent ent_b("banana");
  const struct dirent *a = ent_a.get();
  const struct dirent *b = ent_b.get();

  EXPECT_LT(LIBC_NAMESPACE::alphasort(&a, &b), 0);
  EXPECT_GT(LIBC_NAMESPACE::alphasort(&b, &a), 0);
}

TEST(LlvmLibcAlphasortTest, EqualNames) {
  MockDirent ent_1("file.txt");
  MockDirent ent_2("file.txt");
  const struct dirent *d1 = ent_1.get();
  const struct dirent *d2 = ent_2.get();

  EXPECT_EQ(LIBC_NAMESPACE::alphasort(&d1, &d2), 0);
  EXPECT_EQ(LIBC_NAMESPACE::alphasort(&d1, &d1), 0);
}

TEST(LlvmLibcAlphasortTest, PrefixComparison) {
  MockDirent ent_short("file");
  MockDirent ent_long("file.txt");
  const struct dirent *d_short = ent_short.get();
  const struct dirent *d_long = ent_long.get();

  EXPECT_LT(LIBC_NAMESPACE::alphasort(&d_short, &d_long), 0);
  EXPECT_GT(LIBC_NAMESPACE::alphasort(&d_long, &d_short), 0);
}

TEST(LlvmLibcAlphasortTest, EmptyName) {
  MockDirent ent_empty("");
  MockDirent ent_nonempty("a");
  const struct dirent *d_empty = ent_empty.get();
  const struct dirent *d_nonempty = ent_nonempty.get();

  EXPECT_LT(LIBC_NAMESPACE::alphasort(&d_empty, &d_nonempty), 0);
  EXPECT_GT(LIBC_NAMESPACE::alphasort(&d_nonempty, &d_empty), 0);
  EXPECT_EQ(LIBC_NAMESPACE::alphasort(&d_empty, &d_empty), 0);
}

TEST(LlvmLibcAlphasortTest, QsortSorting) {
  MockDirent ent_delta("delta");
  MockDirent ent_alpha("alpha");
  MockDirent ent_charlie("charlie");
  MockDirent ent_bravo("bravo");

  constexpr size_t NUM_ENTRIES = 4;
  const struct dirent *entries[NUM_ENTRIES] = {
      ent_delta.get(),
      ent_alpha.get(),
      ent_charlie.get(),
      ent_bravo.get(),
  };

  using QsortComparator = int (*)(const void *, const void *);
  LIBC_NAMESPACE::qsort(
      entries, NUM_ENTRIES, sizeof(const struct dirent *),
      reinterpret_cast<QsortComparator>(LIBC_NAMESPACE::alphasort));

  EXPECT_STREQ(entries[0]->d_name, "alpha");
  EXPECT_STREQ(entries[1]->d_name, "bravo");
  EXPECT_STREQ(entries[2]->d_name, "charlie");
  EXPECT_STREQ(entries[3]->d_name, "delta");
}
