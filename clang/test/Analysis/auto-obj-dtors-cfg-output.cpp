// RUN: %clang_analyze_cc1 -fcxx-exceptions -fexceptions -analyzer-checker=debug.DumpCFG -analyzer-config cfg-rich-constructors=false %s > %t 2>&1
// RUN: FileCheck --input-file=%t -check-prefixes=CHECK,WARNINGS %s
// RUN: %clang_analyze_cc1 -fcxx-exceptions -fexceptions -analyzer-checker=debug.DumpCFG -analyzer-config cfg-rich-constructors=true %s > %t 2>&1
// RUN: FileCheck --input-file=%t -check-prefixes=CHECK,ANALYZER %s

// This file tests how we construct two different flavors of the Clang CFG -
// the CFG used by the Sema analysis-based warnings and the CFG used by the
// static analyzer. The difference in the behavior is checked via FileCheck
// prefixes (WARNINGS and ANALYZER respectively). When introducing new analyzer
// flags, no new run lines should be added - just these flags would go to the
// respective line depending on where is it turned on and where is it turned
// off. Feel free to add tests that test only one of the CFG flavors if you're
// not sure how the other flavor is supposed to work in your case.

class A {
public:
// CHECK:      [B1 (ENTRY)]
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
  A() {}

// CHECK:      [B1 (ENTRY)]
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
  ~A() {}

// CHECK:      [B2 (ENTRY)]
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B1]
// CHECK-NEXT:   1: 1
// CHECK-NEXT:   2: return [B1.1];
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
  operator int() const { return 1; }
};

extern const bool UV;

// CHECK:      [B2 (ENTRY)]
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B1]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B1.2], class A)
// CHECK-NEXT:   2: A a;
// CHECK-NEXT:   3: a
// CHECK-NEXT:   4: [B1.3] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   5: const A &b = a;
// WARNINGS-NEXT:   6: A() (CXXConstructExpr, class A)
// ANALYZER-NEXT:   6: A() (CXXConstructExpr, [B1.7], [B1.9], class A)
// CHECK-NEXT:   7: [B1.6] (BindTemporary)
// CHECK-NEXT:   8: [B1.7] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   9: [B1.8]
// CHECK:       10: const A &c = A();
// CHECK:       11: [B1.10].~A() (Implicit destructor)
// CHECK:       12: [B1.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void test_const_ref() {
  A a;
  const A& b = a;
  const A& c = A();
}

// CHECK:      [B2 (ENTRY)]
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B1]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A [2])
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B1.2], class A [2])
// CHECK-NEXT:   2: A a[2];
// WARNINGS-NEXT:   3:  (CXXConstructExpr, class A [0])
// ANALYZER-NEXT:   3:  (CXXConstructExpr, [B1.4], class A [0])
// CHECK-NEXT:   4: A b[0];
// CHECK-NEXT:   5: [B1.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void test_array() {
  A a[2];
  A b[0];
}

// CHECK:      [B2 (ENTRY)]
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B1]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B1.2], class A)
// CHECK-NEXT:   2: A a;
// WARNINGS-NEXT:   3:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   3:  (CXXConstructExpr, [B1.4], class A)
// CHECK-NEXT:   4: A c;
// WARNINGS-NEXT:   5:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   5:  (CXXConstructExpr, [B1.6], class A)
// CHECK-NEXT:   6: A d;
// CHECK-NEXT:   7: [B1.6].~A() (Implicit destructor)
// CHECK-NEXT:   8: [B1.4].~A() (Implicit destructor)
// WARNINGS-NEXT:   9:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   9:  (CXXConstructExpr, [B1.10], class A)
// CHECK:       10: A b;
// CHECK:       11: [B1.10].~A() (Implicit destructor)
// CHECK:       12: [B1.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void test_scope() {
  A a;
  { A c;
    A d;
  }
  A b;
}

// CHECK:      [B4 (ENTRY)]
// CHECK-NEXT:   Succs (1): B3
// CHECK:      [B1]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B1.2], class A)
// CHECK-NEXT:   2: A c;
// CHECK-NEXT:   3: [B1.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B3.4].~A() (Implicit destructor)
// CHECK-NEXT:   5: [B3.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B3
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   1: return;
// CHECK-NEXT:   2: [B3.4].~A() (Implicit destructor)
// CHECK-NEXT:   3: [B3.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B3
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B3]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B3.2], class A)
// CHECK-NEXT:   2: A a;
// WARNINGS-NEXT:   3:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   3:  (CXXConstructExpr, [B3.4], class A)
// CHECK-NEXT:   4: A b;
// CHECK-NEXT:   5: UV
// CHECK-NEXT:   6: [B3.5] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B3.6]
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (2): B2 B1
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (2): B1 B2
void test_return() {
  A a;
  A b;
  if (UV) return;
  A c;
}

// CHECK:      [B8 (ENTRY)]
// CHECK-NEXT:   Succs (1): B7
// CHECK:      [B1]
// CHECK:       l1:
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B1.2], class A)
// CHECK-NEXT:   2: A c;
// CHECK-NEXT:   3: [B1.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B6.2].~A() (Implicit destructor)
// CHECK-NEXT:   5: [B7.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (2): B2 B3
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B2.2], class A)
// CHECK-NEXT:   2: A b;
// CHECK-NEXT:   3: [B2.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B6.4].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B3]
// CHECK-NEXT:   1: [B6.4].~A() (Implicit destructor)
// CHECK-NEXT:   T: goto l1;
// CHECK:        Preds (1): B4
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B4]
// CHECK-NEXT:   1: UV
// CHECK-NEXT:   2: [B4.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B4.2]
// CHECK-NEXT:   Preds (1): B6
// CHECK-NEXT:   Succs (2): B3 B2
// CHECK:      [B5]
// CHECK-NEXT:   1: [B6.4].~A() (Implicit destructor)
// CHECK-NEXT:   2: [B6.2].~A() (Implicit destructor)
// CHECK-NEXT:   T: goto l0;
// CHECK:        Preds (1): B6
// CHECK-NEXT:   Succs (1): B6
// CHECK:      [B6]
// CHECK:       l0:
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B6.2], class A)
// CHECK-NEXT:   2: A b;
// WARNINGS-NEXT:   3:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   3:  (CXXConstructExpr, [B6.4], class A)
// CHECK-NEXT:   4: A a;
// CHECK-NEXT:   5: UV
// CHECK-NEXT:   6: [B6.5] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B6.6]
// CHECK-NEXT:   Preds (2): B7 B5
// CHECK-NEXT:   Succs (2): B5 B4
// CHECK:      [B7]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B7.2], class A)
// CHECK-NEXT:   2: A a;
// CHECK-NEXT:   Preds (1): B8
// CHECK-NEXT:   Succs (1): B6
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void test_goto() {
  A a;
l0:
  A b;
  { A a;
    if (UV) goto l0;
    if (UV) goto l1;
    A b;
  }
l1:
  A c;
}

// CHECK:      [B5 (ENTRY)]
// CHECK-NEXT:   Succs (1): B4
// CHECK:      [B1]
// CHECK-NEXT:   1: [B4.6].~A() (Implicit destructor)
// CHECK-NEXT:   2: [B4.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (2): B2 B3
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B2.2], class A)
// CHECK-NEXT:   2: A c;
// CHECK-NEXT:   3: [B2.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B3]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B3.2], class A)
// CHECK-NEXT:   2: A c;
// CHECK-NEXT:   3: [B3.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B4]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B4.2], class A)
// CHECK-NEXT:   2: A a;
// CHECK-NEXT:   3: a
// CHECK-NEXT:   4: [B4.3] (ImplicitCastExpr, NoOp, const class A)
// WARNINGS-NEXT:   5: [B4.4] (CXXConstructExpr, class A)
// ANALYZER-NEXT:   5: [B4.4] (CXXConstructExpr, [B4.6], class A)
// CHECK-NEXT:   6: A b = a;
// CHECK-NEXT:   7: b
// CHECK-NEXT:   8: [B4.7] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   9: [B4.8].operator int
// CHECK:       10: [B4.8]
// CHECK:       11: [B4.10] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:       12: [B4.11] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:   T: if [B4.12]
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (2): B3 B2
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void test_if_implicit_scope() {
  A a;
  if (A b = a)
    A c;
  else A c;
}

// CHECK:      [B9 (ENTRY)]
// CHECK-NEXT:   Succs (1): B8
// CHECK:      [B1]
// CHECK-NEXT:   1: [B8.6].~A() (Implicit destructor)
// WARNINGS-NEXT:   2:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   2:  (CXXConstructExpr, [B1.3], class A)
// CHECK-NEXT:   3: A e;
// CHECK-NEXT:   4: [B1.3].~A() (Implicit destructor)
// CHECK-NEXT:   5: [B8.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (2): B2 B5
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B2.2], class A)
// CHECK-NEXT:   2: A d;
// CHECK-NEXT:   3: [B2.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B4.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B3]
// CHECK-NEXT:   1: return;
// CHECK-NEXT:   2: [B4.2].~A() (Implicit destructor)
// CHECK-NEXT:   3: [B8.6].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B8.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B4]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B4.2], class A)
// CHECK-NEXT:   2: A c;
// CHECK-NEXT:   3: UV
// CHECK-NEXT:   4: [B4.3] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B4.4]
// CHECK-NEXT:   Preds (1): B8
// CHECK-NEXT:   Succs (2): B3 B2
// CHECK:      [B5]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B5.2], class A)
// CHECK-NEXT:   2: A d;
// CHECK-NEXT:   3: [B5.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B7.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B7
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B6]
// CHECK-NEXT:   1: return;
// CHECK-NEXT:   2: [B7.2].~A() (Implicit destructor)
// CHECK-NEXT:   3: [B8.6].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B8.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B7
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B7]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B7.2], class A)
// CHECK-NEXT:   2: A c;
// CHECK-NEXT:   3: UV
// CHECK-NEXT:   4: [B7.3] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B7.4]
// CHECK-NEXT:   Preds (1): B8
// CHECK-NEXT:   Succs (2): B6 B5
// CHECK:      [B8]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B8.2], class A)
// CHECK-NEXT:   2: A a;
// CHECK-NEXT:   3: a
// CHECK-NEXT:   4: [B8.3] (ImplicitCastExpr, NoOp, const class A)
// WARNINGS-NEXT:   5: [B8.4] (CXXConstructExpr, class A)
// ANALYZER-NEXT:   5: [B8.4] (CXXConstructExpr, [B8.6], class A)
// CHECK-NEXT:   6: A b = a;
// CHECK-NEXT:   7: b
// CHECK-NEXT:   8: [B8.7] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   9: [B8.8].operator int
// CHECK:       10: [B8.8]
// CHECK:       11: [B8.10] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:       12: [B8.11] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:   T: if [B8.12]
// CHECK-NEXT:   Preds (1): B9
// CHECK-NEXT:   Succs (2): B7 B4
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (3): B1 B3 B6
void test_if_jumps() {
  A a;
  if (A b = a) {
    A c;
    if (UV) return;
    A d;
  } else {
    A c;
    if (UV) return;
    A d;
  }
  A e;
}

// CHECK:      [B6 (ENTRY)]
// CHECK-NEXT:   Succs (1): B5
// CHECK:      [B1]
// CHECK-NEXT:   1: [B4.4].~A() (Implicit destructor)
// CHECK-NEXT:   2: [B5.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   Preds (1): B3
// CHECK-NEXT:   Succs (1): B4
// CHECK:      [B3]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B3.2], class A)
// CHECK-NEXT:   2: A c;
// CHECK-NEXT:   3: [B3.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B4.4].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B4]
// CHECK-NEXT:   1: a
// CHECK-NEXT:   2: [B4.1] (ImplicitCastExpr, NoOp, const class A)
// WARNINGS-NEXT:   3: [B4.2] (CXXConstructExpr, class A)
// ANALYZER-NEXT:   3: [B4.2] (CXXConstructExpr, [B4.4], class A)
// CHECK-NEXT:   4: A b = a;
// CHECK-NEXT:   5: b
// CHECK-NEXT:   6: [B4.5] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   7: [B4.6].operator int
// CHECK-NEXT:   8: [B4.6]
// CHECK-NEXT:   9: [B4.8] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:       10: [B4.9] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:   T: while [B4.10]
// CHECK-NEXT:   Preds (2): B2 B5
// CHECK-NEXT:   Succs (2): B3 B1
// CHECK:      [B5]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B5.2], class A)
// CHECK-NEXT:   2: A a;
// CHECK-NEXT:   Preds (1): B6
// CHECK-NEXT:   Succs (1): B4
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void test_while_implicit_scope() {
  A a;
  while (A b = a)
    A c;
}

// CHECK:      [B12 (ENTRY)]
// CHECK-NEXT:   Succs (1): B11
// CHECK:      [B1]
// CHECK-NEXT:   1: [B10.4].~A() (Implicit destructor)
// WARNINGS-NEXT:   2:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   2:  (CXXConstructExpr, [B1.3], class A)
// CHECK-NEXT:   3: A e;
// CHECK-NEXT:   4: [B1.3].~A() (Implicit destructor)
// CHECK-NEXT:   5: [B11.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (2): B8 B10
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   Preds (2): B3 B6
// CHECK-NEXT:   Succs (1): B10
// CHECK:      [B3]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B3.2], class A)
// CHECK-NEXT:   2: A d;
// CHECK-NEXT:   3: [B3.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:   5: [B10.4].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B4]
// CHECK-NEXT:   1: return;
// CHECK-NEXT:   2: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:   3: [B10.4].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B11.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B5]
// CHECK-NEXT:   1: UV
// CHECK-NEXT:   2: [B5.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B5.2]
// CHECK-NEXT:   Preds (1): B7
// CHECK-NEXT:   Succs (2): B4 B3
// CHECK:      [B6]
// CHECK-NEXT:   1: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:   2: [B10.4].~A() (Implicit destructor)
// CHECK-NEXT:   T: continue;
// CHECK:        Preds (1): B7
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B7]
// CHECK-NEXT:   1: UV
// CHECK-NEXT:   2: [B7.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B7.2]
// CHECK-NEXT:   Preds (1): B9
// CHECK-NEXT:   Succs (2): B6 B5
// CHECK:      [B8]
// CHECK-NEXT:   1: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:   T: break;
// CHECK:        Preds (1): B9
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B9]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B9.2], class A)
// CHECK-NEXT:   2: A c;
// CHECK-NEXT:   3: UV
// CHECK-NEXT:   4: [B9.3] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B9.4]
// CHECK-NEXT:   Preds (1): B10
// CHECK-NEXT:   Succs (2): B8 B7
// CHECK:      [B10]
// CHECK-NEXT:   1: a
// CHECK-NEXT:   2: [B10.1] (ImplicitCastExpr, NoOp, const class A)
// WARNINGS-NEXT:   3: [B10.2] (CXXConstructExpr, class A)
// ANALYZER-NEXT:   3: [B10.2] (CXXConstructExpr, [B10.4], class A)
// CHECK-NEXT:   4: A b = a;
// CHECK-NEXT:   5: b
// CHECK-NEXT:   6: [B10.5] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   7: [B10.6].operator int
// CHECK-NEXT:   8: [B10.6]
// CHECK-NEXT:   9: [B10.8] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:       10: [B10.9] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:   T: while [B10.10]
// CHECK-NEXT:   Preds (2): B2 B11
// CHECK-NEXT:   Succs (2): B9 B1
// CHECK:      [B11]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B11.2], class A)
// CHECK-NEXT:   2: A a;
// CHECK-NEXT:   Preds (1): B12
// CHECK-NEXT:   Succs (1): B10
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (2): B1 B4
void test_while_jumps() {
  A a;
  while (A b = a) {
    A c;
    if (UV) break;
    if (UV) continue;
    if (UV) return;
    A d;
  }
  A e;
}

// CHECK:      [B4 (ENTRY)]
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B1]
// CHECK-NEXT:   1: UV
// CHECK-NEXT:   2: [B1.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: do ... while [B1.2]
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (2): B3 B0
// CHECK:      [B2]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B2.2], class A)
// CHECK-NEXT:   2: A a;
// CHECK-NEXT:   3: [B2.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (2): B3 B4
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B3]
// CHECK-NEXT:   Preds (1): B1
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void test_do_implicit_scope() {
  do A a;
  while (UV);
}

// CHECK:      [B12 (ENTRY)]
// CHECK-NEXT:   Succs (1): B11
// CHECK:      [B1]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B1.2], class A)
// CHECK-NEXT:   2: A d;
// CHECK-NEXT:   3: [B1.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B11.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (2): B8 B2
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   1: UV
// CHECK-NEXT:   2: [B2.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: do ... while [B2.2]
// CHECK-NEXT:   Preds (2): B3 B6
// CHECK-NEXT:   Succs (2): B10 B1
// CHECK:      [B3]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B3.2], class A)
// CHECK-NEXT:   2: A c;
// CHECK-NEXT:   3: [B3.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B4]
// CHECK-NEXT:   1: return;
// CHECK-NEXT:   2: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:   3: [B11.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B5]
// CHECK-NEXT:   1: UV
// CHECK-NEXT:   2: [B5.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B5.2]
// CHECK-NEXT:   Preds (1): B7
// CHECK-NEXT:   Succs (2): B4 B3
// CHECK:      [B6]
// CHECK-NEXT:   1: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:   T: continue;
// CHECK:        Preds (1): B7
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B7]
// CHECK-NEXT:   1: UV
// CHECK-NEXT:   2: [B7.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B7.2]
// CHECK-NEXT:   Preds (1): B9
// CHECK-NEXT:   Succs (2): B6 B5
// CHECK:      [B8]
// CHECK-NEXT:   1: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:   T: break;
// CHECK:        Preds (1): B9
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B9]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B9.2], class A)
// CHECK-NEXT:   2: A b;
// CHECK-NEXT:   3: UV
// CHECK-NEXT:   4: [B9.3] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B9.4]
// CHECK-NEXT:   Preds (2): B10 B11
// CHECK-NEXT:   Succs (2): B8 B7
// CHECK:      [B10]
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B9
// CHECK:      [B11]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B11.2], class A)
// CHECK-NEXT:   2: A a;
// CHECK-NEXT:   Preds (1): B12
// CHECK-NEXT:   Succs (1): B9
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (2): B1 B4
void test_do_jumps() {
  A a;
  do {
    A b;
    if (UV) break;
    if (UV) continue;
    if (UV) return;
    A c;
  } while (UV);
  A d;
}

// CHECK:      [B4 (ENTRY)]
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B1]
// CHECK-NEXT:   1: [B2.6].~A() (Implicit destructor)
// CHECK-NEXT:   2: [B2.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (2): B3 B2
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B2.2], class A)
// CHECK-NEXT:   2: A a;
// CHECK-NEXT:   3: a
// CHECK-NEXT:   4: [B2.3] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   5: [B2.4] (CXXConstructExpr, class A)
// CHECK-NEXT:   6: A b = a;
// CHECK-NEXT:   7: b
// CHECK-NEXT:   8: [B2.7] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   9: [B2.8].operator int
// CHECK:       10: [B2.8]
// CHECK:       11: [B2.10] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK-NEXT:   T: switch [B2.11]
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B3]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B3.2], class A)
// CHECK-NEXT:   2: A c;
// CHECK-NEXT:   3: [B3.2].~A() (Implicit destructor)
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void test_switch_implicit_scope() {
  A a;
  switch (A b = a)
    A c;
}

// CHECK:      [B9 (ENTRY)]
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B1]
// CHECK-NEXT:   1: [B2.6].~A() (Implicit destructor)
// WARNINGS-NEXT:   2:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   2:  (CXXConstructExpr, [B1.3], class A)
// CHECK-NEXT:   3: A g;
// CHECK-NEXT:   4: [B1.3].~A() (Implicit destructor)
// CHECK-NEXT:   5: [B2.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (3): B3 B7 B2
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B2.2], class A)
// CHECK-NEXT:   2: A a;
// CHECK-NEXT:   3: a
// CHECK-NEXT:   4: [B2.3] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   5: [B2.4] (CXXConstructExpr, class A)
// CHECK-NEXT:   6: A b = a;
// CHECK-NEXT:   7: b
// CHECK-NEXT:   8: [B2.7] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   9: [B2.8].operator int
// CHECK:       10: [B2.8]
// CHECK:       11: [B2.10] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK-NEXT:   T: switch [B2.11]
// CHECK-NEXT:   Preds (1): B9
// CHECK-NEXT:   Succs (3): B3 B8 B1
// CHECK:      [B3]
// CHECK:       case 1:
// CHECK-NEXT:   T: break;
// CHECK:        Preds (2): B2 B4
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B4]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B4.2], class A)
// CHECK-NEXT:   2: A f;
// CHECK-NEXT:   3: [B4.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B8.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B6
// CHECK-NEXT:   Succs (1): B3
// CHECK:      [B5]
// CHECK-NEXT:   1: return;
// CHECK-NEXT:   2: [B8.2].~A() (Implicit destructor)
// CHECK-NEXT:   3: [B2.6].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B2.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B6
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B6]
// CHECK-NEXT:   1: UV
// CHECK-NEXT:   2: [B6.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B6.2]
// CHECK-NEXT:   Preds (1): B8
// CHECK-NEXT:   Succs (2): B5 B4
// CHECK:      [B7]
// CHECK-NEXT:   1: [B8.2].~A() (Implicit destructor)
// CHECK-NEXT:   T: break;
// CHECK:        Preds (1): B8
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B8]
// CHECK:       case 0:
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B8.2], class A)
// CHECK-NEXT:   2: A c;
// CHECK-NEXT:   3: UV
// CHECK-NEXT:   4: [B8.3] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B8.4]
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (2): B7 B6
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (2): B1 B5
void test_switch_jumps() {
  A a;
  switch (A b = a) {
  case 0: {
    A c;
    if (UV) break;
    if (UV) return;
    A f;
  }
  case 1:
    break;
  }
  A g;
}

// CHECK:      [B6 (ENTRY)]
// CHECK-NEXT:   Succs (1): B5
// CHECK:      [B1]
// CHECK-NEXT:   1: [B4.4].~A() (Implicit destructor)
// CHECK-NEXT:   2: [B5.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   Preds (1): B3
// CHECK-NEXT:   Succs (1): B4
// CHECK:      [B3]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B3.2], class A)
// CHECK-NEXT:   2: A c;
// CHECK-NEXT:   3: [B3.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B4.4].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B4]
// CHECK-NEXT:   1: a
// CHECK-NEXT:   2: [B4.1] (ImplicitCastExpr, NoOp, const class A)
// WARNINGS-NEXT:   3: [B4.2] (CXXConstructExpr, class A)
// ANALYZER-NEXT:   3: [B4.2] (CXXConstructExpr, [B4.4], class A)
// CHECK-NEXT:   4: A b = a;
// CHECK-NEXT:   5: b
// CHECK-NEXT:   6: [B4.5] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   7: [B4.6].operator int
// CHECK-NEXT:   8: [B4.6]
// CHECK-NEXT:   9: [B4.8] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:       10: [B4.9] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:   T: for (...; [B4.10]; )
// CHECK-NEXT:   Preds (2): B2 B5
// CHECK-NEXT:   Succs (2): B3 B1
// CHECK:      [B5]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B5.2], class A)
// CHECK-NEXT:   2: A a;
// CHECK-NEXT:   Preds (1): B6
// CHECK-NEXT:   Succs (1): B4
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void test_for_implicit_scope() {
  for (A a; A b = a; )
    A c;
}

// CHECK:      [B12 (ENTRY)]
// CHECK-NEXT:   Succs (1): B11
// CHECK:      [B1]
// CHECK-NEXT:   1: [B10.4].~A() (Implicit destructor)
// CHECK-NEXT:   2: [B11.4].~A() (Implicit destructor)
// WARNINGS-NEXT:   3:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   3:  (CXXConstructExpr, [B1.4], class A)
// CHECK-NEXT:   4: A f;
// CHECK-NEXT:   5: [B1.4].~A() (Implicit destructor)
// CHECK-NEXT:   6: [B11.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (2): B8 B10
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   Preds (2): B3 B6
// CHECK-NEXT:   Succs (1): B10
// CHECK:      [B3]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B3.2], class A)
// CHECK-NEXT:   2: A e;
// CHECK-NEXT:   3: [B3.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:   5: [B10.4].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B4]
// CHECK-NEXT:   1: return;
// CHECK-NEXT:   2: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:   3: [B10.4].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B11.4].~A() (Implicit destructor)
// CHECK-NEXT:   5: [B11.2].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B5]
// CHECK-NEXT:   1: UV
// CHECK-NEXT:   2: [B5.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B5.2]
// CHECK-NEXT:   Preds (1): B7
// CHECK-NEXT:   Succs (2): B4 B3
// CHECK:      [B6]
// CHECK-NEXT:   1: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:   T: continue;
// CHECK:        Preds (1): B7
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B7]
// CHECK-NEXT:   1: UV
// CHECK-NEXT:   2: [B7.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B7.2]
// CHECK-NEXT:   Preds (1): B9
// CHECK-NEXT:   Succs (2): B6 B5
// CHECK:      [B8]
// CHECK-NEXT:   1: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:   T: break;
// CHECK:        Preds (1): B9
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B9]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B9.2], class A)
// CHECK-NEXT:   2: A d;
// CHECK-NEXT:   3: UV
// CHECK-NEXT:   4: [B9.3] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B9.4]
// CHECK-NEXT:   Preds (1): B10
// CHECK-NEXT:   Succs (2): B8 B7
// CHECK:      [B10]
// CHECK-NEXT:   1: b
// CHECK-NEXT:   2: [B10.1] (ImplicitCastExpr, NoOp, const class A)
// WARNINGS-NEXT:   3: [B10.2] (CXXConstructExpr, class A)
// ANALYZER-NEXT:   3: [B10.2] (CXXConstructExpr, [B10.4], class A)
// CHECK-NEXT:   4: A c = b;
// CHECK-NEXT:   5: c
// CHECK-NEXT:   6: [B10.5] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   7: [B10.6].operator int
// CHECK-NEXT:   8: [B10.6]
// CHECK-NEXT:   9: [B10.8] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:       10: [B10.9] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:   T: for (...; [B10.10]; )
// CHECK-NEXT:   Preds (2): B2 B11
// CHECK-NEXT:   Succs (2): B9 B1
// CHECK:      [B11]
// WARNINGS-NEXT:   1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   1:  (CXXConstructExpr, [B11.2], class A)
// CHECK-NEXT:   2: A a;
// WARNINGS-NEXT:   3:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   3:  (CXXConstructExpr, [B11.4], class A)
// CHECK-NEXT:   4: A b;
// CHECK-NEXT:   Preds (1): B12
// CHECK-NEXT:   Succs (1): B10
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (2): B1 B4
void test_for_jumps() {
  A a;
  for (A b; A c = b; ) {
    A d;
    if (UV) break;
    if (UV) continue;
    if (UV) return;
    A e;
  }
  A f;
}

// CHECK:      [B3 (ENTRY)]
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B1]
// CHECK-NEXT:   T: try ...
// CHECK-NEXT:   Succs (2): B2 B0
// CHECK:      [B2]
// CHECK-NEXT:  catch (const A &e):
// CHECK-NEXT:   1: catch (const A &e) {
// CHECK-NEXT:  }
// CHECK-NEXT:   Preds (1): B1
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (3): B2 B1 B3
void test_catch_const_ref() {
  try {
  } catch (const A& e) {
  }
}

// CHECK:      [B3 (ENTRY)]
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B1]
// CHECK-NEXT:   T: try ...
// CHECK-NEXT:   Succs (2): B2 B0
// CHECK:      [B2]
// CHECK-NEXT:  catch (A e):
// CHECK-NEXT:   1: catch (A e) {
// CHECK-NEXT:  }
// CHECK-NEXT:   2: [B2.1].~A() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B1
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (3): B2 B1 B3
void test_catch_copy() {
  try {
  } catch (A e) {
  }
}
