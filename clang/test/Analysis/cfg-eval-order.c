// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -std=c17 %s 2>&1 | FileCheck %s
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -std=c99 %s 2>&1 | FileCheck %s

int *getPtr(int);
int getVal(int);
int getL(void);
int getR(void);
int getIdx(void);
int arr[10];
int callee(int, int);

// The RHS is emitted before the LHS, matching simple assignment below.
void test_compound_assign(int a, int b) {
  *getPtr(a) += getVal(b);
}

// CHECK-LABEL: void test_compound_assign(int a, int b)
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

// Simple assignment must agree with the compound form above.
void test_simple_assign(int a, int b) {
  *getPtr(a) = getVal(b);
}

// CHECK-LABEL: void test_simple_assign(int a, int b)
// CHECK:          1: getVal
// CHECK:          5: [B1.2]([B1.4])
// CHECK-NEXT:     6: getPtr
// CHECK:         10: [B1.7]([B1.9])
// CHECK-NEXT:    11: *[B1.10]
// CHECK-NEXT:    12: [B1.11] = [B1.5]

// C17 6.5.17: the left operand of a comma is sequenced before the right one.
void test_comma(void) {
  getL(), getR();
}

// CHECK-LABEL: void test_comma(void)
// CHECK:          1: getL
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, int (*)(void))
// CHECK-NEXT:     3: [B1.2]()
// CHECK-NEXT:     4: getR
// CHECK-NEXT:     5: [B1.4] (ImplicitCastExpr, FunctionToPointerDecay, int (*)(void))
// CHECK-NEXT:     6: [B1.5]()
// CHECK-NEXT:     7: ... , [B1.6]

void test_subscript(void) {
  int x = arr[getIdx()];
}

// CHECK-LABEL: void test_subscript(void)
// CHECK:          1: arr
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, ArrayToPointerDecay, int *)
// CHECK-NEXT:     3: getIdx
// CHECK-NEXT:     4: [B1.3] (ImplicitCastExpr, FunctionToPointerDecay, int (*)(void))
// CHECK-NEXT:     5: [B1.4]()
// CHECK-NEXT:     6: [B1.2]{{\[\[}}B1.5]]

// The callee expression is emitted before the arguments. (The order among the
// arguments themselves is unspecified and is not pinned here.)
void test_call_callee_before_args(void) {
  callee(getL(), getR());
}

// CHECK-LABEL: void test_call_callee_before_args(void)
// CHECK:          1: callee
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, int (*)(int, int))
// CHECK-NEXT:     3: getL
// CHECK:          9: [B1.2]([B1.5], [B1.8])
