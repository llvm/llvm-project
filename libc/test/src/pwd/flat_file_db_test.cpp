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
#include "src/__support/CPP/span.h"
#include "src/__support/File/file.h"
#include "src/pwd/field_tokenizer.h"
#include "src/pwd/flat_file_db.h"
#include "src/stdio/remove.h"
#include "src/string/string_utils.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
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
inline bool parse_line<SimpleTestEntry>(cpp::span<char> line,
                                        SimpleTestEntry *entry) {
  if (line.empty() || !entry)
    return false;
  FieldTokenizer tokenizer(line);
  auto k = tokenizer.next_field();
  if (!k)
    return false;
  entry->key = k->data();

  auto v = tokenizer.next_field();
  if (!v)
    return false;
  entry->val = v->data();

  return true;
}

} // namespace pwd
} // namespace LIBC_NAMESPACE_DECL

TEST_F(LlvmLibcFlatFileDbTest, GetNextAndLookup) {
  const char *content = "user1:secret1\nuser2:secret2\n";
  HermeticFile test_file(libc_make_test_file_path("flat_db_test.test"),
                         content);

  LIBC_NAMESPACE::pwd::FlatFileDatabase<SimpleTestEntry> db(
      test_file.get_path());
  char buffer[128];
  SimpleTestEntry entry;

  // First record
  auto r1 = db.getnext(&entry, buffer);
  ASSERT_TRUE(r1.has_value());
  ASSERT_TRUE(r1.value());
  ASSERT_STREQ(entry.key, "user1");
  ASSERT_STREQ(entry.val, "secret1");

  // Second record
  auto r2 = db.getnext(&entry, buffer);
  ASSERT_TRUE(r2.has_value());
  ASSERT_TRUE(r2.value());
  ASSERT_STREQ(entry.key, "user2");
  ASSERT_STREQ(entry.val, "secret2");

  // EOF
  auto r3 = db.getnext(&entry, buffer);
  ASSERT_TRUE(r3.has_value());
  ASSERT_FALSE(r3.value());

  // Rewind and lookup
  db.setdb();
  auto matcher = [](const SimpleTestEntry &e) {
    return LIBC_NAMESPACE::cpp::string_view(e.key) == "user2";
  };
  auto lookup_res = db.lookup(matcher, &entry, buffer);
  ASSERT_TRUE(lookup_res.has_value());
  ASSERT_TRUE(lookup_res.value());
  ASSERT_STREQ(entry.key, "user2");
  ASSERT_STREQ(entry.val, "secret2");

  db.enddb();
}

TEST_F(LlvmLibcFlatFileDbTest, LookupNotFound) {
  const char *content = "foo:bar\n";
  HermeticFile test_file(libc_make_test_file_path("flat_db_not_found.test"),
                         content);

  LIBC_NAMESPACE::pwd::FlatFileDatabase<SimpleTestEntry> db(
      test_file.get_path());
  char buffer[128];
  SimpleTestEntry entry;

  auto matcher = [](const SimpleTestEntry &e) {
    return LIBC_NAMESPACE::cpp::string_view(e.key) == "nonexistent";
  };
  auto lookup_res = db.lookup(matcher, &entry, buffer);
  ASSERT_TRUE(lookup_res.has_value());
  ASSERT_FALSE(lookup_res.value());

  db.enddb();
}

TEST_F(LlvmLibcFlatFileDbTest, TruncatedLineReturnsErange) {
  const char *content = "verylongkeyname:verylongvaluename\n";
  HermeticFile test_file(libc_make_test_file_path("flat_db_trunc.test"),
                         content);

  LIBC_NAMESPACE::pwd::FlatFileDatabase<SimpleTestEntry> db(
      test_file.get_path());
  char small_buffer[8];
  SimpleTestEntry entry;

  auto res = db.getnext(&entry, small_buffer);
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), ERANGE);

  db.enddb();
}

TEST_F(LlvmLibcFlatFileDbTest, MalformedLineReturnsEinval) {
  const char *content = "invalid_line_without_delimiter\n";
  HermeticFile test_file(libc_make_test_file_path("flat_db_malformed.test"),
                         content);

  LIBC_NAMESPACE::pwd::FlatFileDatabase<SimpleTestEntry> db(
      test_file.get_path());
  char buffer[128];
  SimpleTestEntry entry;

  auto res = db.getnext(&entry, buffer);
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), EINVAL);

  db.enddb();
}
