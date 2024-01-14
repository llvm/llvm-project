//===-- Unittests for queue -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string.h"
#include "src/__support/char_vector.h"
#include "test/UnitTest/Test.h"

#include <sys/queue.h>

using LIBC_NAMESPACE::CharVector;
using LIBC_NAMESPACE::cpp::string;

namespace LIBC_NAMESPACE {

TEST(LlvmLibcQueueTest, SList) {
  struct Contrived {
    char c;
    SLIST_ENTRY(Contrived) entry;
  };

  SLIST_HEAD(Head, Contrived) head = SLIST_HEAD_INITIALIZER(head);

  struct Contains : public testing::Matcher<Head> {
    string s;
    Contains(string s) : s(s) {}
    bool match(Head head) {
      Contrived *e;
      CharVector v;
      SLIST_FOREACH(e, &head, entry) { v.append(e->c); }
      return s == v.c_str();
    }
  };

  SLIST_INIT(&head);
  ASSERT_TRUE(SLIST_EMPTY(&head));

  Contrived e1 = {'a', {NULL}};
  SLIST_INSERT_HEAD(&head, &e1, entry);

  ASSERT_THAT(head, Contains("a"));

  Contrived e2 = {'b', {NULL}};
  SLIST_INSERT_AFTER(&e1, &e2, entry);

  ASSERT_THAT(head, Contains("ab"));

  Contrived *e, *tmp = NULL;
  SLIST_FOREACH_SAFE(e, &head, entry, tmp) {
    if (e == &e2) {
      SLIST_REMOVE(&head, e, Contrived, entry);
    }
  }

  ASSERT_THAT(head, Contains("a"));

  while (!SLIST_EMPTY(&head)) {
    e = SLIST_FIRST(&head);
    SLIST_REMOVE_HEAD(&head, entry);
  }

  ASSERT_TRUE(SLIST_EMPTY(&head));
}

TEST(LlvmLibcQueueTest, STailQ) {
  struct Contrived {
    char c;
    STAILQ_ENTRY(Contrived) entry;
  };

  STAILQ_HEAD(Head, Contrived) head = STAILQ_HEAD_INITIALIZER(head);

  struct Contains : public testing::Matcher<Head> {
    string s;
    Contains(string s) : s(s) {}
    bool match(Head head) {
      Contrived *e;
      CharVector v;
      STAILQ_FOREACH(e, &head, entry) { v.append(e->c); }
      return s == v.c_str();
    }
  };

  STAILQ_INIT(&head);
  ASSERT_TRUE(STAILQ_EMPTY(&head));

  Contrived e1 = {'a', {NULL}};
  STAILQ_INSERT_HEAD(&head, &e1, entry);

  ASSERT_THAT(head, Contains("a"));

  Contrived e2 = {'b', {NULL}};
  STAILQ_INSERT_TAIL(&head, &e2, entry);

  ASSERT_THAT(head, Contains("ab"));

  Contrived e3 = {'c', {NULL}};
  SLIST_INSERT_AFTER(&e2, &e3, entry);

  ASSERT_THAT(head, Contains("abc"));

  Contrived *e, *tmp = NULL;
  STAILQ_FOREACH_SAFE(e, &head, entry, tmp) {
    if (e == &e2) {
      STAILQ_REMOVE(&head, e, Contrived, entry);
    }
  }

  ASSERT_THAT(head, Contains("ac"));

  while (!STAILQ_EMPTY(&head)) {
    e = STAILQ_FIRST(&head);
    STAILQ_REMOVE_HEAD(&head, entry);
  }

  ASSERT_TRUE(STAILQ_EMPTY(&head));
}

} // namespace LIBC_NAMESPACE
