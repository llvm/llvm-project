// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus.Move \
// RUN:   -analyzer-checker=alpha.core.LifetimeModeling,debug.ExprInspection \
// RUN:   -analyzer-config cfg-lifetime=true \
// RUN:   -verify %s 2>&1 | FileCheck %s

#include "Inputs/system-header-simulator-cxx.h"

void clang_analyzer_printState();

class A {
public:
  A() = default;
  A(const A&);
  A(A &&);
  int foo() const;
};

void testCheckersOrderInStateDump() {
  {
    A a;
    A *p = &a;
    A b = std::move(a);

    clang_analyzer_printState();
    (void)p;
// CHECK:      "checker_messages": [
// CHECK-NEXT:   { "checker": "alpha.core.LifetimeModeling", "messages": [
// CHECK-NEXT:     "LifetimeBound bindings:",
// CHECK-NEXT:     " Origin &a contains Loan a",
// CHECK-NEXT:     ""
// CHECK-NEXT:   ]},
// CHECK-NEXT:   { "checker": "cplusplus.Move", "messages": [
// CHECK-NEXT:     "Moved-from objects :",
// CHECK-NEXT:     "a: moved",
// CHECK-NEXT:     ""
// CHECK-NEXT:   ]}
// CHECK-NEXT: ]

    a.foo(); // expected-warning {{Method called on moved-from object 'a'}}
  }
}
