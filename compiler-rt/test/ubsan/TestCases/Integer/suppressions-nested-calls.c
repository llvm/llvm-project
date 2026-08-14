// REQUIRES: can-symbolize
// UNSUPPORTED: android

// # Test for UBSan suppressions with nested function calls
//
// RUN: %clang -fsanitize=integer -O0 -g %s -o %t.o0
//
// # Only the directly suppressed my_make_signed hit should disappear.
// RUN: echo "implicit-integer-sign-change:my_make_signed" > %t.make_signed.name.supp
// RUN: echo "implicit-integer-sign-change:Inputs/make_signed.h" > %t.make_signed.file.supp
//
// RUN: %env_ubsan_opts=suppressions='"%t.make_signed.name.supp"' %run %t.o0 2>&1 | FileCheck %s --check-prefix=CHECK-MAKE-SIGNED
// RUN: %env_ubsan_opts=suppressions='"%t.make_signed.file.supp"' %run %t.o0 2>&1 | FileCheck %s --check-prefix=CHECK-MAKE-SIGNED
//
// # Only the suppressed wrapper-originated hit should disappear.
// RUN: echo "implicit-integer-sign-change:my_wrapper_2" > %t.my_wrapper_2.name.supp
// RUN: echo "implicit-integer-sign-change:Inputs/wrappers.h" > %t.wrappers.file.supp
//
// RUN: %env_ubsan_opts=suppressions='"%t.my_wrapper_2.name.supp"' %run %t.o0 2>&1 | FileCheck %s --check-prefix=CHECK-WRAPPERS
// RUN: %env_ubsan_opts=suppressions='"%t.wrappers.file.supp"'     %run %t.o0 2>&1 | FileCheck %s --check-prefix=CHECK-WRAPPERS
//
// # Suppress both.
// RUN: cat %t.make_signed.name.supp %t.my_wrapper_2.name.supp > %t.both.name.supp
// RUN: cat %t.make_signed.file.supp %t.wrappers.file.supp > %t.both.file.supp
//
// RUN: %env_ubsan_opts=suppressions='"%t.both.name.supp"' %run %t.o0 2>&1 | FileCheck %s --allow-empty --check-prefix=CHECK-BOTH
// RUN: %env_ubsan_opts=suppressions='"%t.both.file.supp"' %run %t.o0 2>&1 | FileCheck %s --allow-empty --check-prefix=CHECK-BOTH

#include "Inputs/make_signed.h"
#include "Inputs/wrappers.h"

int main(void) {
  volatile unsigned a1 = 4001111111U;
  volatile unsigned a2 = 4002222222U;
  volatile unsigned a3 = 4003333333U;
  int r1 = my_make_signed(a1);
  int r2 = my_wrapper(a2);
  int r3 = my_wrapper_2(a3);
  return 0;
}

// CHECK-MAKE-SIGNED: wrappers.h:12:23: runtime error: implicit conversion from type 'unsigned int' of value 4003333333
// CHECK-MAKE-SIGNED-NOT: make_signed.h:{{.*}}runtime error:

// CHECK-WRAPPERS: make_signed.h:7:12: runtime error: implicit conversion from type 'unsigned int' of value 4001111111
// CHECK-WRAPPERS: make_signed.h:9:12: runtime error: implicit conversion from type 'unsigned int' of value 4002222222
// CHECK-WRAPPERS: make_signed.h:10:10: runtime error: implicit conversion from type 'unsigned int' of value 4003333333
// CHECK-WRAPPERS-NOT: wrappers.h:{{.*}}runtime error:

// CHECK-BOTH-NOT: runtime error:
