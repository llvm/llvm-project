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
#include "src/string/strcmp.h"
#include "src/string/strcpy.h"
#include "src/string/strlen.h"
#include "test/UnitTest/Test.h"

// ---------------------------------------------------------------------------
// Helpers: encode integers as void* keys (no heap allocation needed).
// ---------------------------------------------------------------------------
static void *encode(int val) {
  return reinterpret_cast<void *>(static_cast<intptr_t>(val));
}

static int decode(const void *val) {
  return static_cast<int>(reinterpret_cast<intptr_t>(val));
}

static int read_node(const __llvm_libc_tnode *node) {
  return static_cast<int>(*static_cast<const intptr_t *>(node));
}

static int int_compare(const void *a, const void *b) {
  int x = decode(a);
  int y = decode(b);
  return (x > y) - (x < y);
}

static void noop_free(void *) {}

// ===== tsearch tests =======================================================

TEST(LlvmLibcTSearchTest, InsertIntoEmptyTree) {
  void *root = nullptr;
  void *r = LIBC_NAMESPACE::tsearch(encode(42), &root, int_compare);
  ASSERT_NE(r, nullptr);
  ASSERT_EQ(read_node(r), 42);
  LIBC_NAMESPACE::tdestroy(root, noop_free);
}

TEST(LlvmLibcTSearchTest, InsertMultiple) {
  void *root = nullptr;
  for (int i = 0; i < 8; ++i) {
    void *r = LIBC_NAMESPACE::tsearch(encode(i), &root, int_compare);
    ASSERT_NE(r, nullptr);
    ASSERT_EQ(read_node(r), i);
  }
  LIBC_NAMESPACE::tdestroy(root, noop_free);
}

TEST(LlvmLibcTSearchTest, InsertDuplicateReturnsSameNode) {
  void *root = nullptr;
  void *first = LIBC_NAMESPACE::tsearch(encode(7), &root, int_compare);
  ASSERT_NE(first, nullptr);

  // Inserting the same key again must return the existing node.
  void *second = LIBC_NAMESPACE::tsearch(encode(7), &root, int_compare);
  ASSERT_EQ(first, second);
  ASSERT_EQ(read_node(second), 7);
  LIBC_NAMESPACE::tdestroy(root, noop_free);
}

// ===== tfind tests =========================================================

TEST(LlvmLibcTSearchTest, FindInEmptyTree) {
  void *root = nullptr;
  void *r = LIBC_NAMESPACE::tfind(encode(1), &root, int_compare);
  ASSERT_EQ(r, nullptr);
}

TEST(LlvmLibcTSearchTest, FindExistingKey) {
  void *root = nullptr;
  LIBC_NAMESPACE::tsearch(encode(5), &root, int_compare);
  LIBC_NAMESPACE::tsearch(encode(10), &root, int_compare);
  LIBC_NAMESPACE::tsearch(encode(15), &root, int_compare);

  void *r = LIBC_NAMESPACE::tfind(encode(10), &root, int_compare);
  ASSERT_NE(r, nullptr);
  ASSERT_EQ(read_node(r), 10);
  LIBC_NAMESPACE::tdestroy(root, noop_free);
}

TEST(LlvmLibcTSearchTest, FindNonExistentKey) {
  void *root = nullptr;
  LIBC_NAMESPACE::tsearch(encode(1), &root, int_compare);
  LIBC_NAMESPACE::tsearch(encode(3), &root, int_compare);

  void *r = LIBC_NAMESPACE::tfind(encode(2), &root, int_compare);
  ASSERT_EQ(r, nullptr); // not exist
  LIBC_NAMESPACE::tdestroy(root, noop_free);
}

// ===== tdelete tests =======================================================

TEST(LlvmLibcTSearchTest, DeleteFromEmptyTree) {
  void *root = nullptr;
  void *r = LIBC_NAMESPACE::tdelete(encode(1), &root, int_compare);
  ASSERT_EQ(r, nullptr);
}

TEST(LlvmLibcTSearchTest, DeleteNonExistentKey) {
  void *root = nullptr;
  LIBC_NAMESPACE::tsearch(encode(10), &root, int_compare);
  void *r = LIBC_NAMESPACE::tdelete(encode(99), &root, int_compare);
  ASSERT_EQ(r, nullptr);
  // Original key must still be findable.
  ASSERT_NE(LIBC_NAMESPACE::tfind(encode(10), &root, int_compare), nullptr);
  LIBC_NAMESPACE::tdestroy(root, noop_free);
}

TEST(LlvmLibcTSearchTest, DeleteOnlyElement) {
  void *root = nullptr;
  LIBC_NAMESPACE::tsearch(encode(42), &root, int_compare);
  void *r = LIBC_NAMESPACE::tdelete(encode(42), &root, int_compare);
  // POSIX: on success tdelete returns a non-null value.
  ASSERT_NE(r, nullptr);
  // The tree is now empty.
  ASSERT_EQ(root, nullptr);
}

TEST(LlvmLibcTSearchTest, DeleteLeaf) {
  void *root = nullptr;
  LIBC_NAMESPACE::tsearch(encode(10), &root, int_compare);
  LIBC_NAMESPACE::tsearch(encode(5), &root, int_compare);
  LIBC_NAMESPACE::tsearch(encode(15), &root, int_compare);

  void *r = LIBC_NAMESPACE::tdelete(encode(5), &root, int_compare);
  ASSERT_NE(r, nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::tfind(encode(5), &root, int_compare), nullptr);
  // Siblings remain.
  ASSERT_NE(LIBC_NAMESPACE::tfind(encode(10), &root, int_compare), nullptr);
  ASSERT_NE(LIBC_NAMESPACE::tfind(encode(15), &root, int_compare), nullptr);
  LIBC_NAMESPACE::tdestroy(root, noop_free);
}

TEST(LlvmLibcTSearchTest, DeleteInternalNode) {
  void *root = nullptr;
  // Build a small chain: 10 -> 5 -> 3
  LIBC_NAMESPACE::tsearch(encode(10), &root, int_compare);
  LIBC_NAMESPACE::tsearch(encode(5), &root, int_compare);
  LIBC_NAMESPACE::tsearch(encode(3), &root, int_compare);

  void *r = LIBC_NAMESPACE::tdelete(encode(5), &root, int_compare);
  ASSERT_NE(r, nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::tfind(encode(5), &root, int_compare), nullptr);
  ASSERT_NE(LIBC_NAMESPACE::tfind(encode(3), &root, int_compare), nullptr);
  ASSERT_NE(LIBC_NAMESPACE::tfind(encode(10), &root, int_compare), nullptr);
  LIBC_NAMESPACE::tdestroy(root, noop_free);
}

TEST(LlvmLibcTSearchTest, DeleteNodeWithTwoChildren) {
  void *root = nullptr;
  //     10
  //    / \
  //   5   \
  //  /\    \
  // 3  7    15
  // For insertion process, see libc/test/src/__support/weak_avl_test.cpp:208
  LIBC_NAMESPACE::tsearch(encode(10), &root, int_compare);
  LIBC_NAMESPACE::tsearch(encode(5), &root, int_compare);
  LIBC_NAMESPACE::tsearch(encode(15), &root, int_compare);
  LIBC_NAMESPACE::tsearch(encode(3), &root, int_compare);
  LIBC_NAMESPACE::tsearch(encode(7), &root, int_compare);

  void *r = LIBC_NAMESPACE::tdelete(encode(5), &root, int_compare);
  ASSERT_NE(r, nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::tfind(encode(5), &root, int_compare), nullptr);
  // All other keys survive.
  ASSERT_NE(LIBC_NAMESPACE::tfind(encode(3), &root, int_compare), nullptr);
  ASSERT_NE(LIBC_NAMESPACE::tfind(encode(7), &root, int_compare), nullptr);
  ASSERT_NE(LIBC_NAMESPACE::tfind(encode(10), &root, int_compare), nullptr);
  ASSERT_NE(LIBC_NAMESPACE::tfind(encode(15), &root, int_compare), nullptr);
  LIBC_NAMESPACE::tdestroy(root, noop_free);
}

// ===== twalk tests =========================================================

static int walk_sum;

static void sum_action(const __llvm_libc_tnode *node, VISIT visit, int) {
  if (visit == leaf || visit == postorder)
    walk_sum += read_node(node);
}

TEST(LlvmLibcTSearchTest, WalkEmptyTree) {
  walk_sum = 0;
  LIBC_NAMESPACE::twalk(nullptr, sum_action);
  ASSERT_EQ(walk_sum, 0);
}

TEST(LlvmLibcTSearchTest, WalkSingleNode) {
  void *root = nullptr;
  LIBC_NAMESPACE::tsearch(encode(99), &root, int_compare);

  walk_sum = 0;
  LIBC_NAMESPACE::twalk(root, sum_action);
  // A single node is a leaf, visited once.
  ASSERT_EQ(walk_sum, 99);
  LIBC_NAMESPACE::tdestroy(root, noop_free);
}

TEST(LlvmLibcTSearchTest, WalkSumsCorrectly) {
  void *root = nullptr;
  int expected = 0;
  for (int i = 1; i <= 10; ++i) {
    ASSERT_NE(LIBC_NAMESPACE::tsearch(encode(i), &root, int_compare), nullptr);
    expected += i;
  }

  walk_sum = 0;
  LIBC_NAMESPACE::twalk(root, sum_action);
  ASSERT_EQ(walk_sum, expected);
  LIBC_NAMESPACE::tdestroy(root, noop_free);
}

// Verify that twalk visits nodes in sorted (in-order) sequence.
static int walk_prev;
static bool walk_sorted;

static void sorted_action(const __llvm_libc_tnode *node, VISIT visit, int) {
  if (visit == leaf || visit == postorder) {
    int val = read_node(node);
    if (val <= walk_prev)
      walk_sorted = false;
    walk_prev = val;
  }
}

TEST(LlvmLibcTSearchTest, WalkInOrder) {
  void *root = nullptr;
  // Insert in pseudo-random order.
  constexpr int N = 32;
  for (int i = 0; i < N; ++i)
    ASSERT_NE(
        LIBC_NAMESPACE::tsearch(encode((i * 13 + 7) % N), &root, int_compare),
        nullptr);

  walk_prev = -1;
  walk_sorted = true;
  LIBC_NAMESPACE::twalk(root, sorted_action);
  ASSERT_TRUE(walk_sorted);
  LIBC_NAMESPACE::tdestroy(root, noop_free);
}

// Verify twalk depth parameter.
static int walk_max_depth;
static int walk_node_count;

static void depth_action(const __llvm_libc_tnode *, VISIT visit, int depth) {
  if (visit == leaf || visit == postorder) {
    ++walk_node_count;
    if (depth > walk_max_depth)
      walk_max_depth = depth;
  }
}

TEST(LlvmLibcTSearchTest, WalkReportsDepth) {
  void *root = nullptr;
  for (int i = 1; i <= 15; ++i)
    ASSERT_NE(LIBC_NAMESPACE::tsearch(encode(i), &root, int_compare), nullptr);

  walk_max_depth = -1;
  walk_node_count = 0;
  LIBC_NAMESPACE::twalk(root, depth_action);
  ASSERT_EQ(walk_node_count, 15);
  // The maximum depth must be positive for a multi-node tree.
  ASSERT_GT(walk_max_depth, 0);
  LIBC_NAMESPACE::tdestroy(root, noop_free);
}

// ===== twalk_r tests =======================================================

static void collect_action(const __llvm_libc_tnode *node, VISIT visit,
                           void *closure) {
  if (visit == leaf || visit == postorder) {
    int **cursor = static_cast<int **>(closure);
    **cursor = read_node(node);
    *cursor += 1;
  }
}

TEST(LlvmLibcTSearchTest, WalkRProducesSortedOutput) {
  void *root = nullptr;
  constexpr int N = 64;
  for (int i = 0; i < N; ++i)
    ASSERT_NE(LIBC_NAMESPACE::tsearch(encode((i * 37) % N), &root, int_compare),
              nullptr);

  int sorted[N];
  int *cursor = sorted;
  LIBC_NAMESPACE::twalk_r(root, collect_action, &cursor);

  for (int i = 0; i < N; ++i)
    ASSERT_EQ(sorted[i], i);
  LIBC_NAMESPACE::tdestroy(root, noop_free);
}

TEST(LlvmLibcTSearchTest, WalkREmptyTree) {
  int count = 0;
  auto counter = [](const __llvm_libc_tnode *, VISIT, void *c) {
    ++*static_cast<int *>(c);
  };
  LIBC_NAMESPACE::twalk_r(nullptr, counter, &count);
  ASSERT_EQ(count, 0);
}

// ===== tdestroy tests ======================================================

constexpr size_t MAX_VALUE = 64;
static bool free_flags[MAX_VALUE + 1];

static void clear_free_flags() {
  for (bool &flag : free_flags)
    flag = false;
}

static void flag_free(void *node) {
  int key = decode(node);
  free_flags[key] = true;
}

TEST(LlvmLibcTSearchTest, DestroyNullTree) {
  // tdestroy on a null root must not crash.
  LIBC_NAMESPACE::tdestroy(nullptr, flag_free);
}

TEST(LlvmLibcTSearchTest, DestroyVisitsAllNodes) {
  void *root = nullptr;
  clear_free_flags();
  constexpr int N = 10;
  for (int i = 0; i < N; ++i)
    ASSERT_NE(LIBC_NAMESPACE::tsearch(encode(i), &root, int_compare), nullptr);

  LIBC_NAMESPACE::tdestroy(root, flag_free);
  for (int i = 0; i < N; ++i)
    ASSERT_TRUE(free_flags[i]);
}

// ---------------------------------------------------------------------------
// tdestroy with a free-function that has real semantics:
// nodes hold heap-allocated C strings; the free-function calls free() on them.
// ---------------------------------------------------------------------------
static int string_compare(const void *a, const void *b) {
  return LIBC_NAMESPACE::strcmp(static_cast<const char *>(a),
                                static_cast<const char *>(b));
}

static int strings_freed;

static void string_free(void *ptr) {
  ++strings_freed;
  delete[] (static_cast<char *>(ptr));
}

TEST(LlvmLibcTSearchTest, DestroyHeapStrings) {
  void *root = nullptr;

  // Include deliberate duplicates to exercise the tsearch "already exists"
  // path.
  const char *words[] = {"cherry",     "apple", "banana", "date",
                         "elderberry", "apple", "cherry"};
  constexpr int NWORDS = sizeof(words) / sizeof(words[0]);
  constexpr int UNIQUE_WORDS = 5; // number of distinct strings

  int duplicates_detected = 0;
  for (int i = 0; i < NWORDS; ++i) {
    char *dup = new char[LIBC_NAMESPACE::strlen(words[i]) + 1];
    ASSERT_NE(dup, nullptr);
    LIBC_NAMESPACE::strcpy(dup, words[i]);
    void *r = LIBC_NAMESPACE::tsearch(dup, &root, string_compare);
    ASSERT_NE(r, nullptr);
    // tsearch returns a pointer to the node's datum. If the stored key
    // is not our newly allocated string, a pre-existing node was returned
    // and we must free the unused duplicate.
    if (*static_cast<void **>(r) != dup) {
      ++duplicates_detected;
      delete[] dup; // match new[] with delete[]
    }
  }

  ASSERT_EQ(duplicates_detected, NWORDS - UNIQUE_WORDS);

  // Every unique word should be findable.
  const char *unique[] = {"cherry", "apple", "banana", "date", "elderberry"};
  for (int i = 0; i < UNIQUE_WORDS; ++i) {
    void *r = LIBC_NAMESPACE::tfind(unique[i], &root, string_compare);
    ASSERT_NE(r, nullptr);
  }

  strings_freed = 0;
  LIBC_NAMESPACE::tdestroy(root, string_free);
  ASSERT_EQ(strings_freed, UNIQUE_WORDS);
}

// ---------------------------------------------------------------------------
// tdestroy via heap-allocated int boxes – demonstrates real resource cleanup
// with proper duplicate handling.
// ---------------------------------------------------------------------------
static int boxed_int_compare(const void *a, const void *b) {
  int x = *static_cast<const int *>(a);
  int y = *static_cast<const int *>(b);
  return (x > y) - (x < y);
}

static int int_box_freed;

static void int_box_free(void *ptr) {
  ++int_box_freed;
  delete static_cast<int *>(ptr);
}

TEST(LlvmLibcTSearchTest, DestroyIntBoxes) {
  int_box_freed = 0;
  void *root = nullptr;
  constexpr int N = 8;

  // Insert 0..N-1, each as a heap-allocated int.
  for (int i = 0; i < N; ++i) {
    int *p = new int(i);
    ASSERT_NE(p, nullptr);
    void *r = LIBC_NAMESPACE::tsearch(p, &root, boxed_int_compare);
    ASSERT_NE(r, nullptr);
    // All keys are unique here, so the stored key must be ours.
    ASSERT_EQ(*static_cast<void **>(r), static_cast<void *>(p));
  }

  // Now re-insert some duplicates; the caller must free the rejected key.
  int duplicates_detected = 0;
  for (int i = 0; i < N; i += 2) {
    int *p = new int(i);
    void *r = LIBC_NAMESPACE::tsearch(p, &root, boxed_int_compare);
    ASSERT_NE(r, nullptr);
    if (*static_cast<void **>(r) != static_cast<void *>(p)) {
      ++duplicates_detected;
      delete p; // match new with delete
    }
  }
  ASSERT_EQ(duplicates_detected, N / 2);

  // Destroy the tree; int_box_free must be called exactly N times
  // (once per unique key that was actually stored).
  LIBC_NAMESPACE::tdestroy(root, int_box_free);
  ASSERT_EQ(int_box_freed, N);
}
