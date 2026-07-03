// REQUIRES: can-symbolize
// UNSUPPORTED: android

// RUN: %clang -fsanitize=integer -O1 -g %s -o %t
// RUN: %env_ubsan_opts=halt_on_error=1:print_stacktrace=1:report_error_type=1 \
// RUN:   not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NOSUP
// RUN: echo "implicit-integer-sign-change:mid" > %t.supp
// RUN: %env_ubsan_opts=halt_on_error=1:print_stacktrace=1:report_error_type=1:suppressions='"%t.supp"' \
// RUN:   %run %t 2>&1 | FileCheck %s --allow-empty --check-prefix=CHECK-SUP

inline int leaf(unsigned a) { return a; }
inline int mid(unsigned a) { return leaf(a); }

int main(void) { (void)mid(4222111000U); }

// CHECK-NOSUP: runtime error: implicit conversion
// CHECK-NOSUP: {{.*}} leaf
// CHECK-NOSUP: {{.*}} mid

// CHECK-SUP-NOT: runtime error:
