//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for FlatFileDatabase.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/File/file.h"
#include "src/__support/error_or.h"
#include "src/__support/pwd/dynamic_buffer.h"
#include "src/__support/pwd/field_tokenizer.h"
#include "src/__support/pwd/flat_file_db.h"
#include "src/stdio/remove.h"
#include "src/string/string_utils.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

namespace {

struct SimpleTestEntry {
  const char *key;
  const char *val;
};

class HermeticFile {
  char path[256];

public:
  HermeticFile(const char *file_path, const char *content) {
    LIBC_NAMESPACE::internal::strlcpy(path, file_path, sizeof(path));

    auto file_or = LIBC_NAMESPACE::openfile(path, "w");
    if (file_or.has_value()) {
      auto *f = file_or.value();
      size_t len = LIBC_NAMESPACE::internal::string_length(content);
      f->write(content, len);
      f->close();
    }
  }

  ~HermeticFile() { LIBC_NAMESPACE::remove(path); }

  const char *get_path() const { return path; }
};

class LlvmLibcFlatFileDbTest
    : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {};

} // namespace

namespace LIBC_NAMESPACE_DECL {
namespace pwd {

template <>
inline ErrorOr<void> parse_line<SimpleTestEntry>(cpp::span<char> buffer,
                                                 size_t line_len,
                                                 SimpleTestEntry *entry) {
  if (!entry || line_len == 0 || line_len >= buffer.size())
    return Error(EINVAL);

  FieldTokenizer tokenizer(buffer.first(line_len + 1));
  auto k = tokenizer.next_field();
  if (!k)
    return Error(EINVAL);
  entry->key = k->data();

  auto v = tokenizer.next_field();
  if (!v)
    return Error(EINVAL);
  entry->val = v->data();

  return {};
}

} // namespace pwd
} // namespace LIBC_NAMESPACE_DECL

TEST_F(LlvmLibcFlatFileDbTest, GetNextAndLookup) {
  const char *content = "user1:secret1\nuser2:secret2\n";
  HermeticFile test_file(libc_make_test_file_path("flat_db_test.test"),
                         content);

  LIBC_NAMESPACE::pwd::ScopedFlatFileDatabase<SimpleTestEntry> db(
      test_file.get_path());
  char buffer[128];
  SimpleTestEntry entry;

  // First record
  auto r1 = db.getnext(&entry, buffer);
  ASSERT_TRUE(r1.has_value());
  ASSERT_TRUE(r1.value());
  EXPECT_STREQ(entry.key, "user1");
  EXPECT_STREQ(entry.val, "secret1");

  // Second record
  auto r2 = db.getnext(&entry, buffer);
  ASSERT_TRUE(r2.has_value());
  ASSERT_TRUE(r2.value());
  EXPECT_STREQ(entry.key, "user2");
  EXPECT_STREQ(entry.val, "secret2");

  // EOF
  auto r3 = db.getnext(&entry, buffer);
  ASSERT_TRUE(r3.has_value());
  EXPECT_FALSE(r3.value());

  // Rewind and lookup
  db.setdb();
  auto matcher = [](const SimpleTestEntry &e) {
    return LIBC_NAMESPACE::cpp::string_view(e.key) == "user2";
  };
  auto lookup_res = db.lookup(matcher, &entry, buffer);
  ASSERT_TRUE(lookup_res.has_value());
  ASSERT_TRUE(lookup_res.value());
  EXPECT_STREQ(entry.key, "user2");
  EXPECT_STREQ(entry.val, "secret2");
}

TEST_F(LlvmLibcFlatFileDbTest, LookupNotFound) {
  const char *content = "foo:bar\n";
  HermeticFile test_file(libc_make_test_file_path("flat_db_not_found.test"),
                         content);

  LIBC_NAMESPACE::pwd::ScopedFlatFileDatabase<SimpleTestEntry> db(
      test_file.get_path());
  char buffer[128];
  SimpleTestEntry entry;

  auto matcher = [](const SimpleTestEntry &e) {
    return LIBC_NAMESPACE::cpp::string_view(e.key) == "nonexistent";
  };
  auto lookup_res = db.lookup(matcher, &entry, buffer);
  ASSERT_TRUE(lookup_res.has_value());
  EXPECT_FALSE(lookup_res.value());
}

TEST_F(LlvmLibcFlatFileDbTest, TruncatedLineReturnsErange) {
  const char *content = "verylongkeyname:verylongvaluename\n";
  HermeticFile test_file(libc_make_test_file_path("flat_db_trunc.test"),
                         content);

  LIBC_NAMESPACE::pwd::ScopedFlatFileDatabase<SimpleTestEntry> db(
      test_file.get_path());
  char small_buffer[8];
  SimpleTestEntry entry;

  auto res = db.getnext(&entry, small_buffer);
  ASSERT_FALSE(res.has_value());
  EXPECT_EQ(res.error(), ERANGE);
}

TEST_F(LlvmLibcFlatFileDbTest, MalformedLineReturnsEinval) {
  const char *content = "invalid_line_without_delimiter\n";
  HermeticFile test_file(libc_make_test_file_path("flat_db_malformed.test"),
                         content);

  LIBC_NAMESPACE::pwd::ScopedFlatFileDatabase<SimpleTestEntry> db(
      test_file.get_path());
  char buffer[128];
  SimpleTestEntry entry;

  auto res = db.getnext(&entry, buffer);
  ASSERT_FALSE(res.has_value());
  EXPECT_EQ(res.error(), EINVAL);
}

TEST_F(LlvmLibcFlatFileDbTest, BlankLinesSkipped) {
  const char *content = "\n\nuser1:secret1\n\n\nuser2:secret2\n\n";
  HermeticFile test_file(libc_make_test_file_path("flat_db_blank.test"),
                         content);

  LIBC_NAMESPACE::pwd::ScopedFlatFileDatabase<SimpleTestEntry> db(
      test_file.get_path());
  char buffer[128];
  SimpleTestEntry entry;

  // First record (skipping leading blank lines)
  auto r1 = db.getnext(&entry, buffer);
  ASSERT_TRUE(r1.has_value());
  ASSERT_TRUE(r1.value());
  ASSERT_STREQ(entry.key, "user1");
  ASSERT_STREQ(entry.val, "secret1");

  // Second record (skipping consecutive blank lines)
  auto r2 = db.getnext(&entry, buffer);
  ASSERT_TRUE(r2.has_value());
  ASSERT_TRUE(r2.value());
  ASSERT_STREQ(entry.key, "user2");
  ASSERT_STREQ(entry.val, "secret2");

  // EOF (skipping trailing blank lines)
  auto r3 = db.getnext(&entry, buffer);
  ASSERT_TRUE(r3.has_value());
  ASSERT_FALSE(r3.value());

  // Rewind and lookup across blank lines
  db.setdb();
  auto matcher = [](const SimpleTestEntry &e) {
    return LIBC_NAMESPACE::cpp::string_view(e.key) == "user2";
  };
  auto lookup_res = db.lookup(matcher, &entry, buffer);
  ASSERT_TRUE(lookup_res.has_value());
  ASSERT_TRUE(lookup_res.value());
  ASSERT_STREQ(entry.key, "user2");
  ASSERT_STREQ(entry.val, "secret2");
}

TEST_F(LlvmLibcFlatFileDbTest, DynamicBufferReadsArbitrarilyLongLines) {
  // Two records, each far beyond the buffer's initial capacity, so that
  // iteration exercises repeated growth.
  constexpr size_t RECORD_COUNT = 2;
  constexpr size_t VALUE_LENGTH = 4000;
  constexpr size_t RECORD_OVERHEAD = 32;
  char content[RECORD_COUNT * (VALUE_LENGTH + RECORD_OVERHEAD)];

  size_t pos = 0;
  for (size_t record = 0; record < RECORD_COUNT; ++record) {
    const char *key = record == 0 ? "key0:" : "key1:";
    for (const char *p = key; *p != '\0'; ++p)
      content[pos++] = *p;
    for (size_t i = 0; i < VALUE_LENGTH; ++i)
      content[pos++] = 'v';
    content[pos++] = '\n';
  }
  content[pos] = '\0';

  HermeticFile test_file(libc_make_test_file_path("flat_db_longline.test"),
                         content);

  LIBC_NAMESPACE::pwd::ScopedFlatFileDatabase<SimpleTestEntry> db(
      test_file.get_path());
  LIBC_NAMESPACE::pwd::ScopedDynamicBuffer buffer;
  SimpleTestEntry entry;

  auto r1 = db.getnext(&entry, buffer);
  ASSERT_TRUE(r1.has_value());
  ASSERT_TRUE(r1.value());
  ASSERT_STREQ(entry.key, "key0");
  ASSERT_EQ(LIBC_NAMESPACE::internal::string_length(entry.val), VALUE_LENGTH);

  auto r2 = db.getnext(&entry, buffer);
  ASSERT_TRUE(r2.has_value());
  ASSERT_TRUE(r2.value());
  ASSERT_STREQ(entry.key, "key1");

  auto r3 = db.getnext(&entry, buffer);
  ASSERT_TRUE(r3.has_value());
  ASSERT_FALSE(r3.value());
}

TEST_F(LlvmLibcFlatFileDbTest, PrecedingLongRecordsSkippedDuringLookup) {
  const char *content =
      "huge_unrelated_key:012345678901234567890123456789012345\n"
      "target:short\n";
  HermeticFile test_file(libc_make_test_file_path("flat_db_longskip.test"),
                         content);

  LIBC_NAMESPACE::pwd::ScopedFlatFileDatabase<SimpleTestEntry> db(
      test_file.get_path());
  // 24 bytes is large enough for "target:short" (12 chars + '\0') but smaller
  // than "huge_unrelated_key:...". Fixed-buffer lookup must proceed past
  // unrelated long records without falsely returning ERANGE.
  constexpr size_t SMALL_BUFFER_SIZE = 24;
  char buffer[SMALL_BUFFER_SIZE];
  SimpleTestEntry entry;

  auto res = db.lookup(
      [](const SimpleTestEntry &e) {
        return LIBC_NAMESPACE::cpp::string_view(e.key) == "target";
      },
      &entry, buffer);
  ASSERT_TRUE(res.has_value());
  ASSERT_TRUE(res.value());
  ASSERT_STREQ(entry.key, "target");
  ASSERT_STREQ(entry.val, "short");
}

TEST_F(LlvmLibcFlatFileDbTest, DynamicBufferReserveGrowRelease) {
  LIBC_NAMESPACE::pwd::ScopedDynamicBuffer buffer;
  ASSERT_EQ(buffer.capacity(), static_cast<size_t>(0));

  ASSERT_TRUE(buffer.grow());
  size_t initial = buffer.capacity();
  ASSERT_GT(initial, static_cast<size_t>(0));

  ASSERT_TRUE(buffer.grow());
  ASSERT_EQ(buffer.capacity(), initial * 2);

  buffer.release();
  ASSERT_EQ(buffer.capacity(), static_cast<size_t>(0));
  buffer.release();
  ASSERT_EQ(buffer.capacity(), static_cast<size_t>(0));

  constexpr size_t TARGET_RESERVE_CAPACITY = 1024;
  ASSERT_TRUE(buffer.reserve(TARGET_RESERVE_CAPACITY));
  ASSERT_GE(buffer.capacity(), TARGET_RESERVE_CAPACITY);
}
