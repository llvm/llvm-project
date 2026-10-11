// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG \
// RUN:   -std=c++14 %s 2>&1 | FileCheck %s

// RUN: %clang_analyze_cc1 -std=c++14 -analyzer-checker=core,cplusplus.Move \
// RUN:   -analyzer-output=text -verify %s

#include "Inputs/system-header-simulator-cxx.h"

struct S {
  int v;
  int use(S &) { return v; }
};

void safe_useThenMove() {
  S s{7};
  auto lam = [x = s.use(s), y = std::move(s)]() mutable {
    (void)x;
    (void)y;
  };
  lam(); // no-warning: the object is used before it is moved from.
}

// CHECK-LABEL: void safe_useThenMove()
// CHECK:          1: 7
// CHECK-NEXT:     2: {[B1.1]}
// CHECK-NEXT:     3: S s{7};
// CHECK-NEXT:     4: s
// CHECK-NEXT:     5: [B1.4].use
// CHECK-NEXT:     6: s
// CHECK-NEXT:     7: [B1.5]([B1.6])
// CHECK-NEXT:     8: std::move
// CHECK-NEXT:     9: [B1.8] (ImplicitCastExpr, BuiltinFnToFnPtr, typename remove_reference<S &>::type &&(*)(struct S &))
// CHECK-NEXT:    10: s
// CHECK-NEXT:    11: [B1.9]([B1.10])
// CHECK-NEXT:    12: [B1.11] (CXXConstructExpr{{.*}}, typename remove_reference<S &>::type)
// CHECK-NEXT:    13: [x = [B1.7], y = [B1.12]]() mutable {

void unsafe_moveThenUse() {
  S s{7};
  auto lam = [y = std::move(s), x = s.use(s)]() mutable {
    // expected-warning@-1 {{Method called on moved-from object 's'}}
    // expected-note@-2    {{Method called on moved-from object 's'}}
    // expected-note@-3    {{Object 's' is moved}}
    (void)x;
    (void)y;
  };
  lam();
}

// CHECK-LABEL: void unsafe_moveThenUse()
// CHECK:          1: 7
// CHECK-NEXT:     2: {[B1.1]}
// CHECK-NEXT:     3: S s{7};
// CHECK-NEXT:     4: std::move
// CHECK-NEXT:     5: [B1.4] (ImplicitCastExpr, BuiltinFnToFnPtr, typename remove_reference<S &>::type &&(*)(struct S &))
// CHECK-NEXT:     6: s
// CHECK-NEXT:     7: [B1.5]([B1.6])
// CHECK-NEXT:     8: [B1.7] (CXXConstructExpr{{.*}}, typename remove_reference<S &>::type)
// CHECK-NEXT:     9: s
// CHECK-NEXT:    10: [B1.9].use
// CHECK-NEXT:    11: s
// CHECK-NEXT:    12: [B1.10]([B1.11])
// CHECK-NEXT:    13: [y = [B1.8], x = [B1.12]]() mutable {
