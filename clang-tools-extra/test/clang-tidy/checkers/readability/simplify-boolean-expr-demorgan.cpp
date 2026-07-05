// RUN: %check_clang_tidy %s readability-simplify-boolean-expr %t

// Check when we can convert !(A Op B) -> !A InvOp !B.
// RUN: %check_clang_tidy -check-suffixes=",RELAXED" %s \
// RUN: readability-simplify-boolean-expr %t -- -config="{CheckOptions: { \
// RUN: readability-simplify-boolean-expr.SimplifyDeMorganRelaxed: true}}" --

// Verify warning issued when invalid options are specified.
// RUN: clang-tidy %s -checks=-*,readability-simplify-boolean-expr -config="{CheckOptions: { \
// RUN:   readability-simplify-boolean-expr.SimplifyDeMorgan: false, \
// RUN:   readability-simplify-boolean-expr.SimplifyDeMorganRelaxed: true}}" \
// RUN: -- 2>&1 | FileCheck %s -check-prefix=CHECK-BAD-CONFIG \
// RUN:       -implicit-check-not="{{warning|error}}:"

// CHECK-BAD-CONFIG: warning: readability-simplify-boolean-expr: 'SimplifyDeMorganRelaxed' cannot be enabled without 'SimplifyDeMorgan' enabled
void eat(bool);

void foo(bool A1, bool A2, bool A3, bool A4) {
  bool X;

  X = !(A1 && A2);
  X = !(A1 || A2);
  // CHECK-MESSAGES-RELAXED: :[[@LINE-2]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-MESSAGES-RELAXED: :[[@LINE-2]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-FIXES-RELAXED: X = !A1 || !A2;
  // CHECK-FIXES-NEXT-RELAXED: X = !A1 && !A2;

  X = !(!A1 || A2);
  X = !(A1 || !A2);
  X = !(!A1 || !A2);
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-FIXES: X = A1 && !A2;
  // CHECK-FIXES-NEXT: X = !A1 && A2;
  // CHECK-FIXES-NEXT: X = A1 && A2;

  X = !(!A1 && A2);
  X = !(A1 && !A2);
  X = !(!A1 && !A2);
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-FIXES: X = A1 || !A2;
  // CHECK-FIXES-NEXT: X = !A1 || A2;
  // CHECK-FIXES-NEXT: X = A1 || A2;

  X = !(!A1 && !A2 && !A3);
  X = !(!A1 && (!A2 && !A3));
  X = !(!A1 && (A2 && A3));
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-FIXES: X = A1 || A2 || A3;
  // CHECK-FIXES-NEXT: X = A1 || A2 || A3;
  // CHECK-FIXES-NEXT: X = A1 || !A2 || !A3;

  X = !(A1 && A2 == A3);
  X = !(!A1 && A2 > A3);
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-FIXES: X = !A1 || A2 != A3;
  // CHECK-FIXES-NEXT: X = A1 || A2 <= A3;

  // Ensure the check doesn't try to combine fixes for the inner and outer demorgan simplification.
  X = !(!A1 && !(!A2 && !A3));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-FIXES: X = A1 || (!A2 && !A3);

  // Testing to see how it handles parens
  X = !(A1 && !A2 && !A3);
  X = !(A1 && !A2 || !A3);
  X = !(!A1 || A2 && !A3);
  X = !((A1 || !A2) && !A3);
  X = !((A1 || !A2) || !A3);
  // CHECK-MESSAGES: :[[@LINE-5]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-MESSAGES: :[[@LINE-5]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-MESSAGES: :[[@LINE-5]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-MESSAGES: :[[@LINE-5]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-MESSAGES: :[[@LINE-5]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-FIXES: X = !A1 || A2 || A3;
  // CHECK-FIXES-NEXT: X = (!A1 || A2) && A3;
  // CHECK-FIXES-NEXT: X = A1 && (!A2 || A3);
  // CHECK-FIXES-NEXT: X = (!A1 && A2) || A3;
  // CHECK-FIXES-NEXT: X = !A1 && A2 && A3;
  X = !((A1 || A2) && (!A3 || A4));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-FIXES: X = (!A1 && !A2) || (A3 && !A4);

  eat(!(!A1 && !A2));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-FIXES: eat(A1 || A2);

  bool Init = !(!A1 || !A2);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-FIXES: bool Init = A1 && A2;

  X = A1 && !(!A2 || !A3);
  X = A1 || !(!A2 || !A3);
  X = A1 && !(!A2 && !A3);
  // CHECK-MESSAGES: :[[@LINE-3]]:13: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-MESSAGES: :[[@LINE-3]]:13: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-MESSAGES: :[[@LINE-3]]:13: warning: boolean expression can be simplified by DeMorgan's theorem
  // CHECK-FIXES: X = A1 && A2 && A3;
  // CHECK-FIXES-NEXT: X = A1 || (A2 && A3);
  // CHECK-FIXES-NEXT: X = A1 && (A2 || A3);
}
