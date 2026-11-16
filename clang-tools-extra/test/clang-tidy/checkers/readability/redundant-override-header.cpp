// RUN: %check_clang_tidy -check-header %S/Inputs/redundant-override/header.h \
// RUN:   %s readability-redundant-override %t
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -I%t.cpp.headers %t.cpp

#include "header.h"

void HeaderDerived::f() { HeaderBase::f(); }
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: redundant override
// CHECK-FIXES-NOT: void HeaderDerived::f()
int HeaderDerived::value(int n) const { return HeaderBase::value(n); }
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: redundant override
// CHECK-FIXES-NOT: int HeaderDerived::value(

void HeaderVisibility::f() { HeaderBase::f(); }
// CHECK-FIXES: void HeaderVisibility::f() { HeaderBase::f(); }

void use(HeaderDerived &object) {
  object.f();
  object.value(1);
}
