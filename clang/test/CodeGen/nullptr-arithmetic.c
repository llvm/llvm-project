// RUN: %clang_cc1 -S %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -S %s -emit-llvm -triple i686-unknown-unknown -o - | FileCheck %s
// RUN: %clang_cc1 -S %s -emit-llvm -triple x86_64-unknown-unknown -o - | FileCheck %s

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

// This doesn't meet the idiom because the element type is larger than a byte.
int16_t *test2(intptr_t n) {
  return (int16_t*)0 + n;
}
// CHECK-LABEL: test2
// CHECK: getelementptr
// CHECK-NOT: inttoptr

// This doesn't meet the idiom because the offset is subtracted.
int8_t* test3(intptr_t n) {
  return NULLPTRI8 - n;
}
// CHECK-LABEL: test3
// CHECK: getelementptr
// CHECK-NOT: inttoptr

// This checks the case where the offset isn't pointer-sized.
// The front end will implicitly cast the offset to an integer, so we need to
// make sure that doesn't cause problems on targets where integers and pointers
// are not the same size.
int8_t *test4(int8_t b) {
  return NULLPTRI8 + b;
}
// CHECK-LABEL: test4
// CHECK: inttoptr
// CHECK-NOT: getelementptr
