// RUN: %clang_cc1 -S %s -emit-llvm -o - | FileCheck %s

#include <stdint.h>

// This test is meant to verify code that handles the 'p = nullptr + n' idiom
// used by some versions of glibc and gcc.  This is undefined behavior but
// it is intended there to act like a conversion from a pointer-sized integer
// to a pointer, and we would like to tolerate that.

#define NULLPTRI8 ((int8_t*)0)

// This should get the inttoptr instruction.
int8_t *test1(intptr_t n) {
  return NULLPTRI8 + n;
}
// CHECK-LABEL: test1
// CHECK: inttoptr
// CHECK-NOT: getelementptr

// This doesn't meet the idiom because the offset type isn't pointer-sized.
int8_t *test2(int8_t n) {
  return NULLPTRI8 + n;
}
// CHECK-LABEL: test2
// CHECK: getelementptr
// CHECK-NOT: inttoptr

// This doesn't meet the idiom because the element type is larger than a byte.
int16_t *test3(intptr_t n) {
  return (int16_t*)0 + n;
}
// CHECK-LABEL: test3
// CHECK: getelementptr
// CHECK-NOT: inttoptr

// This doesn't meet the idiom because the offset is subtracted.
int8_t* test4(intptr_t n) {
  return NULLPTRI8 - n;
}
// CHECK-LABEL: test4
// CHECK: getelementptr
// CHECK-NOT: inttoptr
