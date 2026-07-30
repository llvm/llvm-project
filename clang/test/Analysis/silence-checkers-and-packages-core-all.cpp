// RUN: %clang_analyze_cc1 \
// RUN:  -analyzer-checker=core -analyzer-config \
// RUN:   silence-checkers=core \
// RUN:  -verify %s

// RUN: %clang_analyze_cc1 \
// RUN:  -analyzer-checker=core -analyzer-config \
// RUN:   silence-checkers="core.DivideZero;core.NullDereference" \
// RUN:  -verify %s

// RUN: not %clang_analyze_cc1 -verify %s \
// RUN:  -analyzer-checker=core -analyzer-config \
// RUN:   silence-checkers=core.NullDeref \
// RUN:  2>&1 | FileCheck %s -check-prefix=CHECK-CHECKER-ERROR

// CHECK-CHECKER-ERROR:      (frontend): no analyzer checkers or packages
// CHECK-CHECKER-ERROR-SAME:             are associated with 'core.NullDeref'

// RUN: not %clang_analyze_cc1 -verify %s \
// RUN:  -analyzer-checker=core -analyzer-config \
// RUN:   silence-checkers=coreModeling \
// RUN:  2>&1 | FileCheck %s -check-prefix=CHECK-PACKAGE-ERROR

// CHECK-PACKAGE-ERROR:      (frontend): no analyzer checkers or packages
// CHECK-PACKAGE-ERROR-SAME:             are associated with 'coreModeling'

void test_disable_core_div_by_zero() {
  (void)(1 / 0);
  // expected-warning@-1 {{division by zero is undefined}}
  // no-warning: 'Division by zero'
}

void test_disable_null_deref(int *p) {
  if (p)
    return;

  int x = p[0];
  // no-warning: Array access (from variable 'p') results in a null pointer dereference
}
