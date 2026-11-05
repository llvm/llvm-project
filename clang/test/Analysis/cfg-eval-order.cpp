// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -std=c++17 %s 2>&1 | FileCheck %s
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -std=c++11 %s 2>&1 | FileCheck %s

// The C++11 run intentionally reuses the same expectations.
// Before C++17 these orders are unspecified, so these lines pin
// a deliberate implementation choice.

int getL();
int getR();
int getIdx();
int arr[10];

// [expr.shift]: the left operand is sequenced before the right operand.
void test_shift() {
  int x = getL() << getR();
}

// CHECK-LABEL: void test_shift()
// CHECK:          1: getL
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, int (*)(void))
// CHECK-NEXT:     3: [B1.2]()
// CHECK-NEXT:     4: getR
// CHECK-NEXT:     5: [B1.4] (ImplicitCastExpr, FunctionToPointerDecay, int (*)(void))
// CHECK-NEXT:     6: [B1.5]()
// CHECK-NEXT:     7: [B1.3] << [B1.6]

// [expr.sub] (C++17): the array operand is sequenced before the index operand.
void test_subscript() {
  int x = arr[getIdx()];
}

// CHECK-LABEL: void test_subscript()
// CHECK:          1: arr
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, ArrayToPointerDecay, int *)
// CHECK-NEXT:     3: getIdx
// CHECK-NEXT:     4: [B1.3] (ImplicitCastExpr, FunctionToPointerDecay, int (*)(void))
// CHECK-NEXT:     5: [B1.4]()
// CHECK-NEXT:     6: [B1.2]{{\[\[}}B1.5]]

struct Obj {
  int m;
};
int Obj::*getPMD();
Obj *getObj();

// [expr.mptr.oper]: the left operand is sequenced before the right operand.
void test_ptr_to_member() {
  int x = getObj()->*getPMD();
}

// CHECK-LABEL: void test_ptr_to_member()
// CHECK:          1: getObj
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, Obj *(*)(void))
// CHECK-NEXT:     3: [B1.2]()
// CHECK-NEXT:     4: getPMD
// CHECK-NEXT:     5: [B1.4] (ImplicitCastExpr, FunctionToPointerDecay, int Obj::*(*)(void))
// CHECK-NEXT:     6: [B1.5]()
// CHECK-NEXT:     7: [B1.3] ->* [B1.6]

// [expr.comma]: the left operand is sequenced before the right operand.
void test_comma() {
  getL(), getR();
}

// CHECK-LABEL: void test_comma()
// CHECK:          1: getL
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, int (*)(void))
// CHECK-NEXT:     3: [B1.2]()
// CHECK-NEXT:     4: getR
// CHECK-NEXT:     5: [B1.4] (ImplicitCastExpr, FunctionToPointerDecay, int (*)(void))
// CHECK-NEXT:     6: [B1.5]()
// CHECK-NEXT:     7: ... , [B1.6]

int callee(int, int);

// [expr.call] (C++17): the callee (postfix-expression) is sequenced before the
// arguments. (The order among the arguments themselves is unspecified and is
// not pinned here.)
void test_call_callee_before_args() {
  callee(getL(), getR());
}

// CHECK-LABEL: void test_call_callee_before_args()
// CHECK:          1: callee
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, int (*)(int, int))
// CHECK-NEXT:     3: getL
// CHECK:          9: [B1.2]([B1.5], [B1.8])

struct Stream {
  Stream &operator<<(int);
};
Stream &getStream();

// A chained overloaded oper<< is left-associative: the inner call (object and
// its argument) is fully evaluated before the outer call's argument, matching
// the built-in left-to-right shift order.
void test_overloaded_shift() {
  getStream() << getL() << getR();
}

// CHECK-LABEL: void test_overloaded_shift()
// CHECK:          5: getStream
// CHECK:          7: [B1.6]()
// CHECK-NEXT:     8: getL
// CHECK:         10: [B1.9]()
// CHECK-NEXT:    11: [B1.7] << [B1.10] (OperatorCall)
// CHECK-NEXT:    12: getR
// CHECK:         14: [B1.13]()
// CHECK-NEXT:    15: [B1.11] << [B1.14] (OperatorCall)
