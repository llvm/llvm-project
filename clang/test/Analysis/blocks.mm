// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core -fblocks -analyzer-opt-analyze-nested-blocks -verify -x objective-c++ %s
// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,debug.DumpCFG -fblocks -analyzer-opt-analyze-nested-blocks %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s

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
// CHECK-NEXT:   2: [B1.1] (CXXConstructExpr, const struct StructWithCopyConstructor)
// CHECK-NEXT:   3: ^{ }
// CHECK-NEXT:   4: (void)([B1.3]) (CStyleCastExpr, ToVoid, void)
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
// CHECK-NEXT:   2: [B1.1] (CXXConstructExpr, struct StructWithCopyConstructor)
// CHECK-NEXT:   3: StructWithCopyConstructor s(5) __attribute__((blocks("byref")));
// CHECK-NEXT:   4: ^{ }
// CHECK-NEXT:   5: (void)([B1.4]) (CStyleCastExpr, ToVoid, void)
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0

// CHECK: [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
