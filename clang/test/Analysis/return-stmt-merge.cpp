// RUN: %clang_analyze_cc1 -analyzer-checker=debug.AnalysisOrder,debug.ExprInspection -analyzer-config debug.AnalysisOrder:PreCall=true,debug.AnalysisOrder:PostCall=true,debug.AnalysisOrder:LiveSymbols=true %s 2>&1 | FileCheck %s

// This test ensures that check::LiveSymbols is called as many times on the
// path through the second "return" as it is through the first "return"
// (three), and therefore the two paths were not merged prematurely before the
// respective return statement is evaluated.
// The paths would still be merged later, so we'd have only one post-call for
// foo(), but it is incorrect to merge them in the middle of evaluating two
// different statements.
int coin();

void foo() {
  int x = coin();
  if (x > 0)
    return;
  else
    return;
}

void bar() {
  foo();
}

// CHECK:      LiveSymbols
// CHECK-NEXT: LiveSymbols
// CHECK-NEXT: PreCall (foo)
// CHECK-NEXT: LiveSymbols
// CHECK-NEXT: LiveSymbols
// CHECK-NEXT: PreCall (coin)
// CHECK-NEXT: PostCall (coin)
// CHECK-NEXT: LiveSymbols
// CHECK-NEXT: LiveSymbols
// CHECK-NEXT: LiveSymbols
// CHECK-NEXT: PostCall (foo)
// CHECK-NEXT: LiveSymbols
// CHECK-NEXT: LiveSymbols
// CHECK-NEXT: LiveSymbols
