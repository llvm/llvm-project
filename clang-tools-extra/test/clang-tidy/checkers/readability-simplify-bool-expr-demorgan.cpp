// RUN: %check_clang_tidy %s readability-simplify-boolean-expr %t

void eat(bool);

void foo(bool A1, bool A2, bool A3, bool A4) {
  bool X;
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
