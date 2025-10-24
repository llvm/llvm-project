// RUN: %clangxx -fsanitize=alignment,null -O0 %s -o %t && %run %t
// RUN: %clangxx -fsanitize=alignment,null -O0 -DTEST_NULL_SRC %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NULL-SRC
// RUN: %clangxx -fsanitize=alignment,null -O0 -DTEST_NULL_DEST %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NULL-DEST
// RUN: %clangxx -fsanitize=alignment,null -O0 -DTEST_MISALIGN_SRC %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-ALIGN-SRC
// RUN: %clangxx -fsanitize=alignment,null -O0 -DTEST_MISALIGN_DEST %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-ALIGN-DEST

// Tests for null pointer and alignment checks in aggregate copy operations.
// This validates the sanitizer checks added to EmitAggregateCopy for both
// source and destination pointers with null and alignment violations.

#include <stdlib.h>
#include <string.h>

struct alignas(16) AlignedStruct {
  int a;
  int b;
  int c;
  int d;
};

struct NormalStruct {
  int x;
  int y;
  int z;
};

void test_null_src() {
  AlignedStruct dest;
  AlignedStruct *src = nullptr;
  // CHECK-NULL-SRC: runtime error: load of null pointer of type 'AlignedStruct'
  dest = *src;
}

void test_null_dest() {
  AlignedStruct src = {1, 2, 3, 4};
  AlignedStruct *dest = nullptr;
  // CHECK-NULL-DEST: runtime error: store to null pointer of type 'AlignedStruct'
  *dest = src;
}

void test_misaligned_src() {
  char buffer[sizeof(AlignedStruct) + 16];
  // Create a misaligned pointer (not 16-byte aligned)
  AlignedStruct *src = (AlignedStruct *)(buffer + 1);
  AlignedStruct dest;
  // CHECK-ALIGN-SRC: runtime error: load of misaligned address {{0x[0-9a-f]+}} for type 'AlignedStruct', which requires 16 byte alignment
  dest = *src;
}

void test_misaligned_dest() {
  AlignedStruct src = {1, 2, 3, 4};
  char buffer[sizeof(AlignedStruct) + 16];
  // Create a misaligned pointer (not 16-byte aligned)
  AlignedStruct *dest = (AlignedStruct *)(buffer + 1);
  // CHECK-ALIGN-DEST: runtime error: store to misaligned address {{0x[0-9a-f]+}} for type 'AlignedStruct', which requires 16 byte alignment
  *dest = src;
}

void test_normal_copy() {
  // This should work fine - properly aligned, non-null pointers
  AlignedStruct src = {1, 2, 3, 4};
  AlignedStruct dest;
  dest = src;
}

int main() {
#ifdef TEST_NULL_SRC
  test_null_src();
#elif defined(TEST_NULL_DEST)
  test_null_dest();
#elif defined(TEST_MISALIGN_SRC)
  test_misaligned_src();
#elif defined(TEST_MISALIGN_DEST)
  test_misaligned_dest();
#else
  test_normal_copy();
#endif
  return 0;
}
