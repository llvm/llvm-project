//===-- Unittests for qsort_r ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/qsort_r.h"

#include "test/UnitTest/Test.h"

#include <stdlib.h>

static int int_compare_count(const void *l, const void *r, void *count_arg) {
  int li = *reinterpret_cast<const int *>(l);
  int ri = *reinterpret_cast<const int *>(r);
  size_t *count = reinterpret_cast<size_t *>(count_arg);
  *count = *count + 1;
  if (li == ri)
    return 0;
  else if (li > ri)
    return 1;
  else
    return -1;
}

TEST(LlvmLibcQsortRTest, SortedArray) {
  int array[25] = {10,   23,   33,   35,   55,   70,    71,   100,  110,
                   123,  133,  135,  155,  170,  171,   1100, 1110, 1123,
                   1133, 1135, 1155, 1170, 1171, 11100, 12310};
  constexpr size_t ARRAY_SIZE = sizeof(array) / sizeof(int);

  size_t count = 0;

  __llvm_libc::qsort_r(array, ARRAY_SIZE, sizeof(int), int_compare_count,
                       &count);

  ASSERT_LE(array[0], 10);
  ASSERT_LE(array[1], 23);
  ASSERT_LE(array[2], 33);
  ASSERT_LE(array[3], 35);
  ASSERT_LE(array[4], 55);
  ASSERT_LE(array[5], 70);
  ASSERT_LE(array[6], 71);
  ASSERT_LE(array[7], 100);
  ASSERT_LE(array[8], 110);
  ASSERT_LE(array[9], 123);
  ASSERT_LE(array[10], 133);
  ASSERT_LE(array[11], 135);
  ASSERT_LE(array[12], 155);
  ASSERT_LE(array[13], 170);
  ASSERT_LE(array[14], 171);
  ASSERT_LE(array[15], 1100);
  ASSERT_LE(array[16], 1110);
  ASSERT_LE(array[17], 1123);
  ASSERT_LE(array[18], 1133);
  ASSERT_LE(array[19], 1135);
  ASSERT_LE(array[20], 1155);
  ASSERT_LE(array[21], 1170);
  ASSERT_LE(array[22], 1171);
  ASSERT_LE(array[23], 11100);
  ASSERT_LE(array[24], 12310);

  // This is a sorted list, but there still have to have been at least N
  // comparisons made.
  ASSERT_GE(count, ARRAY_SIZE);
}

TEST(LlvmLibcQsortRTest, ReverseSortedArray) {
  int array[25] = {25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,
                   12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1};
  constexpr size_t ARRAY_SIZE = sizeof(array) / sizeof(int);

  size_t count = 0;

  __llvm_libc::qsort_r(array, ARRAY_SIZE, sizeof(int), int_compare_count,
                       &count);

  for (int i = 0; i < int(ARRAY_SIZE - 1); ++i)
    ASSERT_LE(array[i], i + 1);

  ASSERT_GE(count, ARRAY_SIZE);
}

// The following test is intended to mimic the CPP library pattern of having a
// comparison function that takes a specific type, which is passed to a library
// that then needs to sort an array of that type. The library can't safely pass
// the comparison function to qsort because a function that takes const T*
// being cast to a function that takes const void* is undefined behavior. The
// safer pattern is to pass a type erased comparator that calls into the typed
// comparator to qsort_r.

struct PriorityVal {
  int priority;
  int size;
};

static int compare_priority_val(const PriorityVal *l, const PriorityVal *r) {
  // Subtracting the priorities is unsafe, but it's fine for this test.
  int priority_diff = l->priority - r->priority;
  if (priority_diff != 0) {
    return priority_diff;
  }
  if (l->size == r->size) {
    return 0;
  } else if (l->size > r->size) {
    return 1;
  } else {
    return -1;
  }
}

template <typename T>
static int type_erased_comp(const void *l, const void *r,
                            void *erased_func_ptr) {
  typedef int (*TypedComp)(const T *, const T *);
  TypedComp typed_func_ptr = reinterpret_cast<TypedComp>(erased_func_ptr);
  const T *lt = reinterpret_cast<const T *>(l);
  const T *rt = reinterpret_cast<const T *>(r);
  return typed_func_ptr(lt, rt);
}

TEST(LlvmLibcQsortRTest, SafeTypeErasure) {
  PriorityVal array[5] = {
      {10, 3}, {1, 10}, {-1, 100}, {10, 0}, {3, 3},
  };
  constexpr size_t ARRAY_SIZE = sizeof(array) / sizeof(PriorityVal);

  __llvm_libc::qsort_r(array, ARRAY_SIZE, sizeof(PriorityVal),
                       type_erased_comp<PriorityVal>,
                       reinterpret_cast<void *>(compare_priority_val));

  EXPECT_EQ(array[0].priority, -1);
  EXPECT_EQ(array[0].size, 100);
  EXPECT_EQ(array[1].priority, 1);
  EXPECT_EQ(array[1].size, 10);
  EXPECT_EQ(array[2].priority, 3);
  EXPECT_EQ(array[2].size, 3);
  EXPECT_EQ(array[3].priority, 10);
  EXPECT_EQ(array[3].size, 0);
  EXPECT_EQ(array[4].priority, 10);
  EXPECT_EQ(array[4].size, 3);
}
