// REQUIRES: can-symbolize
// UNSUPPORTED: android

// RUN: %clang -fsanitize=integer -O1 -g %s -o %t
// RUN: echo "implicit-integer-sign-change:%s" > %t.main.supp
// RUN: %env_ubsan_opts=halt_on_error=1:print_stacktrace=1:report_error_type=1:suppressions='"%t.main.supp"' \
// RUN:   not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-MAIN-FILE
// RUN: echo "implicit-integer-sign-change:%p/Inputs/suppressions-inline-origin-file.h" > %t.header.supp
// RUN: %env_ubsan_opts=halt_on_error=1:print_stacktrace=1:report_error_type=1:suppressions='"%t.header.supp"' \
// RUN:   %run %t 2>&1 | FileCheck %s --allow-empty --check-prefix=CHECK-HEADER-FILE

#include "Inputs/suppressions-inline-origin-file.h"

int main(void) { (void)my_fun(4222111000U); }

// Suppressing the caller file must not suppress a UB originating in the
// inlined header.
// CHECK-MAIN-FILE: runtime error: implicit conversion
// CHECK-MAIN-FILE: {{.*}} my_fun
// CHECK-MAIN-FILE: {{.*}} main

// CHECK-HEADER-FILE-NOT: runtime error:
