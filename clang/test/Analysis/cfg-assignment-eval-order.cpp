// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -std=c++17 %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s

// CHECK-LABEL: void test_eval_order(Map &m, int a, int b)
// CHECK:         1: operator=
// CHECK-NEXT:    2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, Map &(*)(const Map &))
// CHECK-NEXT:    3: operator[]
// CHECK-NEXT:    4: [B1.3] (ImplicitCastExpr, FunctionToPointerDecay, Map &(*)(int))
// CHECK-NEXT:    5: m
// CHECK-NEXT:    6: a
// CHECK-NEXT:    7: [B1.6] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:    8: [B1.5]{{\[\[}}B1.7]] (OperatorCall)
// CHECK-NEXT:    9: [B1.8] (ImplicitCastExpr, NoOp, const Map)
// CHECK-NEXT:   10: operator[]
// CHECK-NEXT:   11: [B1.10] (ImplicitCastExpr, FunctionToPointerDecay, Map &(*)(int))
// CHECK-NEXT:   12: m
// CHECK-NEXT:   13: b
// CHECK-NEXT:   14: [B1.13] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   15: [B1.12]{{\[\[}}B1.14]] (OperatorCall)
// CHECK-NEXT:   16: [B1.15] = [B1.9] (OperatorCall)

struct Map {
  Map &operator[](int);
  Map &operator=(const Map &);
};

void test_eval_order(Map &m, int a, int b) {
  m[b] = m[a];
}

struct A {
  A();
  ~A();
  A &operator=(const A &);
};

struct B {
  B();
  ~B();
};

struct C {
  C();
  ~C();
};

A &getLHS(const B &);
A &getRHS(const C &);

// CHECK-LABEL: void test_temp_dtor_order()
// CHECK:       ~B() (Temporary object destructor)
// CHECK-NEXT:  ~C() (Temporary object destructor)
void test_temp_dtor_order() {
  getLHS(B()) = getRHS(C());
}
