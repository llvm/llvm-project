//===-- Unittests for tsearch ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/posix_tnode.h"
#include "src/search/tdelete.h"
#include "src/search/tdestroy.h"
#include "src/search/tfind.h"
#include "src/search/tsearch.h"
#include "src/search/twalk.h"
#include "src/search/twalk_r.h"
#include "test/UnitTest/Test.h"

static void *encode(int val) {
  return reinterpret_cast<void *>(static_cast<intptr_t>(val));
}

static int decode(const void *val) {
  return static_cast<int>(reinterpret_cast<intptr_t>(val));
}

static int read_and_decode(const __llvm_libc_tnode *val) {
  return static_cast<int>(*static_cast<const intptr_t *>(val));
}

static int compare(const void *a, const void *b) {
  int x = decode(a);
  int y = decode(b);
  return (x > y) - (x < y);
}

TEST(LlvmLibcTSearchTest, TSearch) {
  void *root = nullptr;
  void *result;
  int key = 10;

  // Insert 10
  result = LIBC_NAMESPACE::tsearch(encode(key), &root, compare);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(read_and_decode(result), key);

  // Find 10
  result = LIBC_NAMESPACE::tfind(encode(key), &root, compare);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(read_and_decode(result), key);

  // Insert 20
  int key2 = 20;
  result = LIBC_NAMESPACE::tsearch(encode(key2), &root, compare);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(read_and_decode(result), key2);

  // Find 20
  result = LIBC_NAMESPACE::tfind(encode(key2), &root, compare);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(read_and_decode(result), key2);

  // Delete 10
  result = LIBC_NAMESPACE::tdelete(encode(key), &root, compare);
  ASSERT_NE(result, nullptr);

  // Find 10 should fail
  result = LIBC_NAMESPACE::tfind(encode(key), &root, compare);
  ASSERT_EQ(result, nullptr);

  // Delete 20
  result = LIBC_NAMESPACE::tdelete(encode(key2), &root, compare);
  ASSERT_NE(result, nullptr);
  // Tree should be empty
  ASSERT_EQ(root, nullptr);
}

constexpr size_t MAX_VALUE = 64;
static bool free_flags[MAX_VALUE + 1];
static void clear_free_flags() {
  for (bool &flag : free_flags)
    flag = false;
}
static void free_node(void *node) {
  int key = decode(node);
  free_flags[key] = true;
}

TEST(LlvmLibcTSearchTest, TDestroy) {
  void *root = nullptr;
  clear_free_flags();
  for (int i = 0; i < 10; ++i)
    LIBC_NAMESPACE::tsearch(encode(i), &root, compare);

  LIBC_NAMESPACE::tdestroy(root, free_node);
  for (int i = 0; i < 10; ++i)
    ASSERT_TRUE(free_flags[i]);
}

static int walk_sum = 0;
static void action(const __llvm_libc_tnode *node, VISIT visit, int) {
  if (visit == leaf || visit == postorder)
    walk_sum += read_and_decode(node);
}

TEST(LlvmLibcTSearchTest, TWalk) {
  void *root = nullptr;
  int sum = 0;
  for (int i = 1; i <= 5; ++i) {
    LIBC_NAMESPACE::tsearch(encode(i), &root, compare);
    sum += i;
  }

  walk_sum = 0;
  LIBC_NAMESPACE::twalk(root, action);
  ASSERT_EQ(walk_sum, sum);

  LIBC_NAMESPACE::tdestroy(root, free_node);
}

static void action_closure(const __llvm_libc_tnode *node, VISIT visit,
                           void *closure) {
  if (visit == leaf || visit == postorder) {
    int **cursor = static_cast<int **>(closure);
    **cursor = read_and_decode(node);
    *cursor += 1;
  }
}

TEST(LlvmLibcTSearchTest, TWalkR) {
  void *root = nullptr;
  clear_free_flags();
  constexpr int LIMIT = 64;
  int sorted[LIMIT];
  for (int i = 0; i < LIMIT; ++i) {
    int current = (i * 37) % LIMIT; // pseudo-random insertion order
    LIBC_NAMESPACE::tsearch(encode(current), &root, compare);
  }

  walk_sum = 0;
  int *cursor = &sorted[0];
  LIBC_NAMESPACE::twalk_r(root, action_closure, &cursor);

  for (int i = 0; i < LIMIT; ++i)
    ASSERT_EQ(sorted[i], i);

  LIBC_NAMESPACE::tdestroy(root, free_node);
  for (int i = 0; i < LIMIT; ++i)
    ASSERT_TRUE(free_flags[i]);
}
