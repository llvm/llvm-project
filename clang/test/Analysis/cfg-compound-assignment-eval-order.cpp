// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -std=c++17 %s 2>&1 | FileCheck %s
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -std=c++11 %s 2>&1 | FileCheck %s

// RUN: %clang_analyze_cc1 -std=c++17 -analyzer-checker=core,cplusplus.Move \
// RUN:   -analyzer-output=text -verify %s

#include "Inputs/system-header-simulator-cxx.h"

int *getPtr(int);
int getVal(int);

// Compound assignment operators sequence the RHS before the LHS, exactly like
// simple assignment ([expr.ass]/1). The RHS (getVal) must therefore appear
// before the LHS (getPtr) in the CFG.
void test_builtin_compound(int a, int b) {
  *getPtr(a) += getVal(b);
}

// CHECK-LABEL: void test_builtin_compound(int a, int b)
// CHECK:          1: getVal
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, int (*)(int))
// CHECK-NEXT:     3: b
// CHECK-NEXT:     4: [B1.3] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:     5: [B1.2]([B1.4])
// CHECK-NEXT:     6: getPtr
// CHECK-NEXT:     7: [B1.6] (ImplicitCastExpr, FunctionToPointerDecay, int *(*)(int))
// CHECK-NEXT:     8: a
// CHECK-NEXT:     9: [B1.8] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:    10: [B1.7]([B1.9])
// CHECK-NEXT:    11: *[B1.10]
// CHECK-NEXT:    12: [B1.11] += [B1.5]

struct A { A(); ~A(); };
struct B { B(); ~B(); };

int &getLHS(const A &);
int getRHS(const B &);

// The RHS temporary (B) is constructed before the LHS temporary (A), because
// the RHS is fully sequenced before the LHS. Temporaries are destroyed in
// reverse order of construction, so ~A() runs before ~B().
void test_temp_dtor_order() {
  getLHS(A()) += getRHS(B());
}

// CHECK-LABEL: void test_temp_dtor_order()
// CHECK:       ~A() (Temporary object destructor)
// CHECK-NEXT:  ~B() (Temporary object destructor)

struct S {
  int use(S &) const { return 1; }
};

int consume(S);

// Because the RHS is evaluated before the LHS, the object 's' is moved from
// (in the RHS) before it is used (in the LHS index computation).
void test_move_then_use() {
  S s;
  int arr[10] = {};
  arr[s.use(s)] += consume(std::move(s));
  // expected-warning@-1 {{Method called on moved-from object 's'}}
  // expected-note@-2    {{Method called on moved-from object 's'}}
  // expected-note@-3    {{Object 's' is moved}}
  (void)arr;
}
