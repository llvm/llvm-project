//===-- Unittests for queue -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string.h"
#include "src/__support/char_vector.h"
#include "src/__support/macros/config.h"
#include "test/UnitTest/Test.h"

#include "include/llvm-libc-macros/sys-queue-macros.h"

using LIBC_NAMESPACE::CharVector;
using LIBC_NAMESPACE::cpp::string;

namespace LIBC_NAMESPACE_DECL {

TEST(LlvmLibcQueueTest, SList) {
  struct Entry {
    char c;
    SLIST_ENTRY(Entry) entries;
  };

  SLIST_HEAD(Head, Entry);

  Head head = SLIST_HEAD_INITIALIZER(head);

  struct Contains : public testing::Matcher<Head> {
    string s;
    Contains(string s) : s(s) {}
    bool match(Head head) {
      Entry *e;
      CharVector v;
      SLIST_FOREACH(e, &head, entries) { v.append(e->c); }
      return s == v.c_str();
    }
  };

  Entry e1 = {'a', {NULL}};
  SLIST_INSERT_HEAD(&head, &e1, entries);

  ASSERT_THAT(head, Contains("a"));

  Entry e2 = {'b', {NULL}};
  SLIST_INSERT_AFTER(&e1, &e2, entries);

  ASSERT_THAT(head, Contains("ab"));

  Head head2 = SLIST_HEAD_INITIALIZER(head);

  Entry e3 = {'c', {NULL}};
  SLIST_INSERT_HEAD(&head2, &e3, entries);

  ASSERT_THAT(head2, Contains("c"));

  SLIST_SWAP(&head, &head2, Entry);

  ASSERT_THAT(head2, Contains("ab"));

  SLIST_CONCAT(&head2, &head, Entry, entries);

  ASSERT_THAT(head2, Contains("abc"));

  SLIST_CONCAT(&head, &head2, Entry, entries);

  ASSERT_THAT(head, Contains("abc"));

  Entry *e = NULL, *tmp = NULL;
  SLIST_FOREACH_SAFE(e, &head, entries, tmp) {
    if (e == &e2) {
      SLIST_REMOVE(&head, e, Entry, entries);
    }
  }

  ASSERT_THAT(head, Contains("ac"));

  while (!SLIST_EMPTY(&head)) {
    e = SLIST_FIRST(&head);
    SLIST_REMOVE_HEAD(&head, entries);
  }

  ASSERT_TRUE(SLIST_EMPTY(&head));
}

TEST(LlvmLibcQueueTest, STailQ) {
  struct Entry {
    char c;
    STAILQ_ENTRY(Entry) entries;
  };

  STAILQ_HEAD(Head, Entry);

  Head head = STAILQ_HEAD_INITIALIZER(head);

  struct Contains : public testing::Matcher<Head> {
    string s;
    Contains(string s) : s(s) {}
    bool match(Head head) {
      Entry *e;
      CharVector v;
      STAILQ_FOREACH(e, &head, entries) { v.append(e->c); }
      return s == v.c_str();
    }
  };

  STAILQ_INIT(&head);
  ASSERT_TRUE(STAILQ_EMPTY(&head));

  Entry e1 = {'a', {NULL}};
  STAILQ_INSERT_HEAD(&head, &e1, entries);

  ASSERT_THAT(head, Contains("a"));

  Entry e2 = {'b', {NULL}};
  STAILQ_INSERT_TAIL(&head, &e2, entries);

  ASSERT_THAT(head, Contains("ab"));

  Entry e3 = {'c', {NULL}};
  STAILQ_INSERT_AFTER(&head, &e2, &e3, entries);

  ASSERT_THAT(head, Contains("abc"));

  Head head2 = STAILQ_HEAD_INITIALIZER(head);

  Entry e4 = {'d', {NULL}};
  STAILQ_INSERT_HEAD(&head2, &e4, entries);

  ASSERT_THAT(head2, Contains("d"));

  STAILQ_SWAP(&head, &head2, Entry);

  ASSERT_THAT(head2, Contains("abc"));

  STAILQ_CONCAT(&head2, &head, Entry, entries);

  ASSERT_EQ(STAILQ_FIRST(&head2), &e1);
  ASSERT_EQ(STAILQ_LAST(&head2, Entry, entries), &e4);

  ASSERT_THAT(head2, Contains("abcd"));

  STAILQ_CONCAT(&head, &head2, Entry, entries);

  ASSERT_EQ(STAILQ_FIRST(&head), &e1);
  ASSERT_EQ(STAILQ_LAST(&head, Entry, entries), &e4);

  ASSERT_THAT(head, Contains("abcd"));

  Entry *e = NULL, *tmp = NULL;
  STAILQ_FOREACH_SAFE(e, &head, entries, tmp) {
    if (e == &e2) {
      STAILQ_REMOVE(&head, e, Entry, entries);
    }
  }

  ASSERT_THAT(head, Contains("acd"));

  while (!STAILQ_EMPTY(&head)) {
    e = STAILQ_FIRST(&head);
    STAILQ_REMOVE_HEAD(&head, entries);
  }

  ASSERT_TRUE(STAILQ_EMPTY(&head));
}

} // namespace LIBC_NAMESPACE_DECL
