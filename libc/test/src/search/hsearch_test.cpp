//===-- Unittests for hsearch ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/bit.h" // bit_ceil
#include "src/__support/HashTable/table.h"
#include "src/search/hcreate.h"
#include "src/search/hcreate_r.h"
#include "src/search/hdestroy.h"
#include "src/search/hdestroy_r.h"
#include "src/search/hsearch.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcHsearchTest, CreateTooLarge) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  struct hsearch_data hdata;
  ASSERT_THAT(LIBC_NAMESPACE::hcreate(-1), Fails(ENOMEM, 0));
  ASSERT_THAT(LIBC_NAMESPACE::hcreate_r(-1, &hdata), Fails(ENOMEM, 0));
}

TEST(LlvmLibcHSearchTest, CreateInvalid) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(LIBC_NAMESPACE::hcreate_r(16, nullptr), Fails(EINVAL, 0));
}

TEST(LlvmLibcHSearchTest, CreateValid) {
  struct hsearch_data hdata;
  ASSERT_GT(LIBC_NAMESPACE::hcreate_r(1, &hdata), 0);
  LIBC_NAMESPACE::hdestroy_r(&hdata);

  ASSERT_GT(LIBC_NAMESPACE::hcreate(1), 0);
  LIBC_NAMESPACE::hdestroy();
}

char search_data[] = "1234567890abcdefghijklmnopqrstuvwxyz"
                     "1234567890abcdefghijklmnopqrstuvwxyz"
                     "1234567890abcdefghijklmnopqrstuvwxyz"
                     "1234567890abcdefghijklmnopqrstuvwxyz"
                     "1234567890abcdefghijklmnopqrstuvwxyz";
char search_data2[] =
    "@@@@@@@@@@@@@@!!!!!!!!!!!!!!!!!###########$$$$$$$$$$^^^^^^&&&&&&&&";

constexpr size_t GROUP_SIZE = sizeof(LIBC_NAMESPACE::internal::Group);
constexpr size_t CAP =
    LIBC_NAMESPACE::cpp::bit_ceil((GROUP_SIZE + 1) * 8 / 7) / 8 * 7;
static_assert(CAP < sizeof(search_data), "CAP too large");

TEST(LlvmLibcHSearchTest, GrowFromZero) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_GT(LIBC_NAMESPACE::hcreate(0), 0);
  for (size_t i = 0; i < sizeof(search_data) - 1; ++i) {
    ENTRY *inserted = LIBC_NAMESPACE::hsearch(
        {&search_data[i], reinterpret_cast<void *>(i)}, ENTER);
    ASSERT_NE(inserted, static_cast<ENTRY *>(nullptr));
    ASSERT_EQ(inserted->key, &search_data[i]);
  }
  for (size_t i = sizeof(search_data) - 1; i != 0; --i) {
    ASSERT_EQ(
        LIBC_NAMESPACE::hsearch({&search_data[i - 1], nullptr}, FIND)->data,
        reinterpret_cast<void *>(i - 1));
  }

  LIBC_NAMESPACE::hdestroy();
}

TEST(LlvmLibcHSearchTest, NotFound) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_GT(LIBC_NAMESPACE::hcreate(GROUP_SIZE + 1), 0);
  ASSERT_THAT(static_cast<void *>(
                  LIBC_NAMESPACE::hsearch({search_data2, nullptr}, FIND)),
              Fails(ESRCH, static_cast<void *>(nullptr)));
  for (size_t i = 0; i < CAP; ++i) {
    ASSERT_EQ(LIBC_NAMESPACE::hsearch({&search_data[i], nullptr}, ENTER)->key,
              &search_data[i]);
  }
  ASSERT_THAT(static_cast<void *>(
                  LIBC_NAMESPACE::hsearch({search_data2, nullptr}, FIND)),
              Fails(ESRCH, static_cast<void *>(nullptr)));
  LIBC_NAMESPACE::hdestroy();
}

TEST(LlvmLibcHSearchTest, Found) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_GT(LIBC_NAMESPACE::hcreate(GROUP_SIZE + 1), 0);
  for (size_t i = 0; i < CAP; ++i) {
    ENTRY *inserted = LIBC_NAMESPACE::hsearch(
        {&search_data[i], reinterpret_cast<void *>(i)}, ENTER);
    ASSERT_NE(inserted, static_cast<ENTRY *>(nullptr));
    ASSERT_EQ(inserted->key, &search_data[i]);
  }
  for (size_t i = 0; i < CAP; ++i) {
    ASSERT_EQ(LIBC_NAMESPACE::hsearch({&search_data[i], nullptr}, FIND)->data,
              reinterpret_cast<void *>(i));
  }
  LIBC_NAMESPACE::hdestroy();
}

TEST(LlvmLibcHSearchTest, OnlyInsertWhenNotFound) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_GT(LIBC_NAMESPACE::hcreate(GROUP_SIZE + 1), 0);
  for (size_t i = 0; i < CAP / 7 * 5; ++i) {
    ASSERT_EQ(LIBC_NAMESPACE::hsearch(
                  {&search_data[i], reinterpret_cast<void *>(i)}, ENTER)
                  ->key,
              &search_data[i]);
  }
  for (size_t i = 0; i < CAP; ++i) {
    ASSERT_EQ(LIBC_NAMESPACE::hsearch(
                  {&search_data[i], reinterpret_cast<void *>(1000 + i)}, ENTER)
                  ->key,
              &search_data[i]);
  }
  for (size_t i = 0; i < CAP / 7 * 5; ++i) {
    ASSERT_EQ(LIBC_NAMESPACE::hsearch({&search_data[i], nullptr}, FIND)->data,
              reinterpret_cast<void *>(i));
  }
  for (size_t i = CAP / 7 * 5; i < CAP; ++i) {
    ASSERT_EQ(LIBC_NAMESPACE::hsearch({&search_data[i], nullptr}, FIND)->data,
              reinterpret_cast<void *>(1000 + i));
  }
  LIBC_NAMESPACE::hdestroy();
}
