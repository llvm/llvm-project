// REQUIRES: can-symbolize
// UNSUPPORTED: android

// # Companion test for UBSan suppressions with -finline.
// # This variant intentionally differs from the -fno-inline baseline.
//
// RUN: %clang -fsanitize=integer -fsanitize-recover=integer -O1 -finline -g %s -o %t
//
// # Only the directly suppressed my_make_signed hit should disappear.
//
// RUN: echo "implicit-integer-sign-change:my_make_signed" > %t.make_signed.name.supp
// RUN: %env_ubsan_opts=report_error_type=1:print_stacktrace=1:suppressions='"%t.make_signed.name.supp"' \
// RUN:   %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-MAKE-SIGNED-INLINE-NAME
// RUN: echo "implicit-integer-sign-change:Inputs/make_signed.h" > %t.make_signed.file.supp
// RUN: %env_ubsan_opts=report_error_type=1:print_stacktrace=1:suppressions='"%t.make_signed.file.supp"' \
// RUN:   %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-MAKE-SIGNED
//
// # Only the suppressed wrapper-originated hit should disappear.
//
// RUN: echo "implicit-integer-sign-change:my_wrapper_2" > %t.wrapper_2.name.supp
// RUN: echo "implicit-integer-sign-change:my_wrapper" > %t.wrapper.name.supp
// RUN: cat %t.wrapper.name.supp %t.wrapper_2.name.supp > %t.wrappers.name.supp
// RUN: %env_ubsan_opts=report_error_type=1:print_stacktrace=1:suppressions='"%t.wrappers.name.supp"' \
// RUN:   %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-WRAPPERS
// RUN: echo "implicit-integer-sign-change:Inputs/wrappers.h" > %t.wrappers.file.supp
// RUN: %env_ubsan_opts=report_error_type=1:print_stacktrace=1:suppressions='"%t.wrappers.file.supp"' \
// RUN:   %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-WRAPPERS
//
// # Suppress both.
//
// RUN: cat %t.make_signed.name.supp %t.wrapper_2.name.supp > %t.both.name.supp
// RUN: %env_ubsan_opts=report_error_type=1:print_stacktrace=1:suppressions='"%t.both.name.supp"' \
// RUN:   %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-BOTH-INLINE-NAME
// RUN: cat %t.make_signed.file.supp %t.wrappers.file.supp > %t.both.file.supp
// RUN: %env_ubsan_opts=report_error_type=1:print_stacktrace=1:suppressions='"%t.both.file.supp"' \
// RUN:   %run %t 2>&1 | FileCheck %s --allow-empty --check-prefix=CHECK-BOTH

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

// The inlined my_make_signed call still produces a report under name-based
// suppression.
// CHECK-MAKE-SIGNED-INLINE-NAME: {{.*}}wrappers.h:12:23: runtime error: implicit conversion from type 'unsigned int' of value 4003333333
// CHECK-MAKE-SIGNED-INLINE-NAME: {{.*}}make_signed.h:10:10: runtime error: implicit conversion from type 'unsigned int' of value 4003333333

// CHECK-MAKE-SIGNED: {{.*}}wrappers.h:12:23: runtime error: implicit conversion from type 'unsigned int' of value 4003333333
// CHECK-MAKE-SIGNED-NOT: make_signed.h:{{.*}}runtime error:

// CHECK-WRAPPERS: {{.*}}make_signed.h:7:12: runtime error: implicit conversion from type 'unsigned int' of value 4001111111
// CHECK-WRAPPERS: {{.*}}make_signed.h:9:12: runtime error: implicit conversion from type 'unsigned int' of value 4002222222
// CHECK-WRAPPERS: {{.*}}make_signed.h:10:10: runtime error: implicit conversion from type 'unsigned int' of value 4003333333
// CHECK-WRAPPERS-NOT: {{.*}}wrappers.h:{{.*}}runtime error:

// The inlined my_make_signed call still produces a report under name-based
// suppression.
// CHECK-BOTH-INLINE-NAME: {{.*}}make_signed.h:10:10: runtime error: implicit conversion from type 'unsigned int' of value 4003333333

// CHECK-BOTH-NOT: runtime error:
