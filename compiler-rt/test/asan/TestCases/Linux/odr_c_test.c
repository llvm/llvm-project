// Test that we can properly report an ODR violation
// between an instrumented global and a non-instrumented global.

// RUN: %clang_asan %s -fPIC -shared -o %t-1.so  -DFILE1
// RUN: %clang_asan %s -fPIC -shared -o %t-2.so  -DFILE2
// RUN: %clang_asan %s -fPIE %t-1.so %t-2.so -Wl,-R`pwd` -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
//
// REQUIRES: x86_64-supported-target
//
// CHECK: The following global variable is not properly aligned.
// CHECK: ERROR: AddressSanitizer: odr-violation
#if defined(FILE1)
__attribute__((aligned(8))) int x;
__attribute__((aligned(1))) char y;
__attribute__((aligned(1))) char ZZZ[100];
#elif defined(FILE2)
int ZZZ = 1;
#else
extern int ZZZ;
int main() {
  return ZZZ;
}
#endif

