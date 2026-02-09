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

// ---------------------------------------------------------------------------
// OOP-style RAII int_box around the POSIX tsearch family.
//
// The int_box owns the tree root and calls tdestroy in its destructor.
// It is parameterised on a free-function so callers can track or customise
// destruction.
// ---------------------------------------------------------------------------
using FreeFn = void (*)(void *);

static void noop_free(void *) {}

class TTree {
  void *root;
  FreeFn free_fn;

  // Non-copyable.
  TTree(const TTree &) = delete;
  TTree &operator=(const TTree &) = delete;

public:
  explicit TTree(FreeFn f = noop_free) : root(nullptr), free_fn(f) {}

  ~TTree() {
    if (root)
      LIBC_NAMESPACE::tdestroy(root, free_fn);
  }

  // Insert key.  Returns the node pointer on success, nullptr on failure.
  void *insert(void *key) {
    return LIBC_NAMESPACE::tsearch(key, &root, int_compare);
  }

  // Find key.  Returns nullptr when not found.
  void *find(void *key) const {
    return LIBC_NAMESPACE::tfind(key, &root, int_compare);
  }

  // Delete key.  Returns the parent node (or the new root) on success,
  // nullptr when the key is not in the tree.
  void *remove(void *key) {
    return LIBC_NAMESPACE::tdelete(key, &root, int_compare);
  }

  void walk(void (*action)(const __llvm_libc_tnode *, VISIT, int)) const {
    LIBC_NAMESPACE::twalk(root, action);
  }

  void walk_r(void (*action)(const __llvm_libc_tnode *, VISIT, void *),
              void *closure) const {
    LIBC_NAMESPACE::twalk_r(root, action, closure);
  }

  // Release ownership so the destructor will not call tdestroy.
  void *release() {
    void *r = root;
    root = nullptr;
    return r;
  }

  bool empty() const { return root == nullptr; }
  void *get_root() const { return root; }
};

// ===== tsearch tests =======================================================

TEST(LlvmLibcTSearchTest, InsertIntoEmptyTree) {
  TTree tree;
  void *r = tree.insert(encode(42));
  ASSERT_NE(r, nullptr);
  ASSERT_EQ(read_node(r), 42);
}

TEST(LlvmLibcTSearchTest, InsertMultiple) {
  TTree tree;
  for (int i = 0; i < 8; ++i) {
    void *r = tree.insert(encode(i));
    ASSERT_NE(r, nullptr);
    ASSERT_EQ(read_node(r), i);
  }
}

TEST(LlvmLibcTSearchTest, InsertDuplicateReturnsSameNode) {
  TTree tree;
  void *first = tree.insert(encode(7));
  ASSERT_NE(first, nullptr);

  // Inserting the same key again must return the existing node.
  void *second = tree.insert(encode(7));
  ASSERT_EQ(first, second);
  ASSERT_EQ(read_node(second), 7);
}

// ===== tfind tests =========================================================

TEST(LlvmLibcTSearchTest, FindInEmptyTree) {
  TTree tree;
  void *r = tree.find(encode(1));
  ASSERT_EQ(r, nullptr);
}

TEST(LlvmLibcTSearchTest, FindExistingKey) {
  TTree tree;
  tree.insert(encode(5));
  tree.insert(encode(10));
  tree.insert(encode(15));

  void *r = tree.find(encode(10));
  ASSERT_NE(r, nullptr);
  ASSERT_EQ(read_node(r), 10);
}

TEST(LlvmLibcTSearchTest, FindNonExistentKey) {
  TTree tree;
  tree.insert(encode(1));
  tree.insert(encode(3));

  void *r = tree.find(encode(2));
  ASSERT_EQ(r, nullptr);
}

// ===== tdelete tests =======================================================

TEST(LlvmLibcTSearchTest, DeleteFromEmptyTree) {
  void *root = nullptr;
  void *r = LIBC_NAMESPACE::tdelete(encode(1), &root, int_compare);
  ASSERT_EQ(r, nullptr);
}

TEST(LlvmLibcTSearchTest, DeleteNonExistentKey) {
  TTree tree;
  tree.insert(encode(10));
  void *r = tree.remove(encode(99));
  ASSERT_EQ(r, nullptr);
  // Original key must still be findable.
  ASSERT_NE(tree.find(encode(10)), nullptr);
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
  TTree tree;
  tree.insert(encode(10));
  tree.insert(encode(5));
  tree.insert(encode(15));

  void *r = tree.remove(encode(5));
  ASSERT_NE(r, nullptr);
  ASSERT_EQ(tree.find(encode(5)), nullptr);
  // Siblings remain.
  ASSERT_NE(tree.find(encode(10)), nullptr);
  ASSERT_NE(tree.find(encode(15)), nullptr);
}

TEST(LlvmLibcTSearchTest, DeleteNodeWithOneChild) {
  TTree tree;
  // Build a small chain: 10 -> 5 -> 3
  tree.insert(encode(10));
  tree.insert(encode(5));
  tree.insert(encode(3));

  void *r = tree.remove(encode(5));
  ASSERT_NE(r, nullptr);
  ASSERT_EQ(tree.find(encode(5)), nullptr);
  ASSERT_NE(tree.find(encode(3)), nullptr);
  ASSERT_NE(tree.find(encode(10)), nullptr);
}

TEST(LlvmLibcTSearchTest, DeleteNodeWithTwoChildren) {
  TTree tree;
  tree.insert(encode(10));
  tree.insert(encode(5));
  tree.insert(encode(15));
  tree.insert(encode(3));
  tree.insert(encode(7));

  void *r = tree.remove(encode(5));
  ASSERT_NE(r, nullptr);
  ASSERT_EQ(tree.find(encode(5)), nullptr);
  // All other keys survive.
  ASSERT_NE(tree.find(encode(3)), nullptr);
  ASSERT_NE(tree.find(encode(7)), nullptr);
  ASSERT_NE(tree.find(encode(10)), nullptr);
  ASSERT_NE(tree.find(encode(15)), nullptr);
}

TEST(LlvmLibcTSearchTest, InsertAndDeleteMany) {
  TTree tree;
  constexpr int N = 64;
  // Insert in pseudo-random order.
  for (int i = 0; i < N; ++i)
    ASSERT_NE(tree.insert(encode((i * 37) % N)), nullptr);

  // Delete every other element.
  for (int i = 0; i < N; i += 2) {
    void *r = tree.remove(encode(i));
    ASSERT_NE(r, nullptr);
  }

  // Verify survivors and absences.
  for (int i = 0; i < N; ++i) {
    void *r = tree.find(encode(i));
    if (i % 2 == 0)
      ASSERT_EQ(r, nullptr);
    else
      ASSERT_NE(r, nullptr);
  }
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
  TTree tree;
  tree.insert(encode(99));

  walk_sum = 0;
  tree.walk(sum_action);
  // A single node is a leaf, visited once.
  ASSERT_EQ(walk_sum, 99);
}

TEST(LlvmLibcTSearchTest, WalkSumsCorrectly) {
  TTree tree;
  int expected = 0;
  for (int i = 1; i <= 10; ++i) {
    tree.insert(encode(i));
    expected += i;
  }

  walk_sum = 0;
  tree.walk(sum_action);
  ASSERT_EQ(walk_sum, expected);
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
  TTree tree;
  // Insert in pseudo-random order.
  constexpr int N = 32;
  for (int i = 0; i < N; ++i)
    tree.insert(encode((i * 13 + 7) % N));

  walk_prev = -1;
  walk_sorted = true;
  tree.walk(sorted_action);
  ASSERT_TRUE(walk_sorted);
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
  TTree tree;
  for (int i = 1; i <= 15; ++i)
    tree.insert(encode(i));

  walk_max_depth = -1;
  walk_node_count = 0;
  tree.walk(depth_action);
  ASSERT_EQ(walk_node_count, 15);
  // The maximum depth must be positive for a multi-node tree.
  ASSERT_GT(walk_max_depth, 0);
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
  TTree tree;
  constexpr int N = 64;
  for (int i = 0; i < N; ++i)
    tree.insert(encode((i * 37) % N));

  int sorted[N];
  int *cursor = sorted;
  tree.walk_r(collect_action, &cursor);

  for (int i = 0; i < N; ++i)
    ASSERT_EQ(sorted[i], i);
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
    LIBC_NAMESPACE::tsearch(encode(i), &root, int_compare);

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

  // Include deliberate duplicates to exercise the tsearch "already exists" path.
  const char *words[] = {"cherry", "apple",  "banana", "date",
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
      delete[] dup;
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
      delete p;
    }
  }
  ASSERT_EQ(duplicates_detected, N / 2);

  // Destroy the tree; int_box_free must be called exactly N times
  // (once per unique key that was actually stored).
  LIBC_NAMESPACE::tdestroy(root, int_box_free);
  ASSERT_EQ(int_box_freed, N);
}
