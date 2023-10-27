// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core -fblocks -verify -x objective-c++ %s
// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,debug.DumpCFG -fblocks -analyzer-config cfg-rich-constructors=false %s > %t 2>&1
// RUN: FileCheck --input-file=%t -check-prefixes=CHECK,WARNINGS %s
// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,debug.DumpCFG -fblocks -analyzer-config cfg-rich-constructors=true %s > %t 2>&1
// RUN: FileCheck --input-file=%t -check-prefixes=CHECK,ANALYZER %s

// This file tests how we construct two different flavors of the Clang CFG -
// the CFG used by the Sema analysis-based warnings and the CFG used by the
// static analyzer. The difference in the behavior is checked via FileCheck
// prefixes (WARNINGS and ANALYZER respectively). When introducing new analyzer
// flags, no new run lines should be added - just these flags would go to the
// respective line depending on where is it turned on and where is it turned
// off. Feel free to add tests that test only one of the CFG flavors if you're
// not sure how the other flavor is supposed to work in your case.

// expected-no-diagnostics

void testBlockWithoutCopyExpression(int i) {
  // Captures i, with no copy expression.
  (void)(^void() {
    (void)i;
  });
}

// CHECK-LABEL:void testBlockWithoutCopyExpression(int i)
// CHECK-NEXT: [B2 (ENTRY)]
// CHECK-NEXT:   Succs (1): B1

// CHECK: [B1]
// CHECK-NEXT:   1: ^{ }
// CHECK-NEXT:   2: (void)([B1.1]) (CStyleCastExpr, ToVoid, void)
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0

// CHECK: [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1

struct StructWithCopyConstructor {
  StructWithCopyConstructor(int i);
  StructWithCopyConstructor(const StructWithCopyConstructor &s);
};
void testBlockWithCopyExpression(StructWithCopyConstructor s) {
  // Captures s, with a copy expression calling the copy constructor for StructWithCopyConstructor.
  (void)(^void() {
    (void)s;
  });
}

// CHECK-LABEL:void testBlockWithCopyExpression(StructWithCopyConstructor s)
// CHECK-NEXT: [B2 (ENTRY)]
// CHECK-NEXT:   Succs (1): B1

// CHECK: [B1]
// CHECK-NEXT:   1: s
// CHECK-NEXT:   2: [B1.1] (ImplicitCastExpr, NoOp, const StructWithCopyConstructor)
// CHECK-NEXT:   3: [B1.2] (CXXConstructExpr, const StructWithCopyConstructor)
// CHECK-NEXT:   4: ^{ }
// CHECK-NEXT:   5: (void)([B1.4]) (CStyleCastExpr, ToVoid, void)
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0

// CHECK: [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1

void testBlockWithCaptureByReference() {
  __block StructWithCopyConstructor s(5);
  // Captures s by reference, so no copy expression.
  (void)(^void() {
    (void)s;
  });
}

// CHECK-LABEL:void testBlockWithCaptureByReference()
// CHECK-NEXT: [B2 (ENTRY)]
// CHECK-NEXT:   Succs (1): B1

// CHECK: [B1]
// CHECK-NEXT:   1: 5
// WARNINGS-NEXT:   2: [B1.1] (CXXConstructExpr, StructWithCopyConstructor)
// ANALYZER-NEXT:   2: [B1.1] (CXXConstructExpr, [B1.3], StructWithCopyConstructor)
// CHECK-NEXT:   3: __attribute__((blocks("byref"))) StructWithCopyConstructor s(5);
// CHECK-NEXT:   4: ^{ }
// CHECK-NEXT:   5: (void)([B1.4]) (CStyleCastExpr, ToVoid, void)
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0

// CHECK: [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
