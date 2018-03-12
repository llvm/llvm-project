// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -analyze -analyzer-checker=debug.DumpCFG -analyzer-config cfg-scopes=true %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s

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

int getX();
extern const bool UV;

// CHECK:      [B2 (ENTRY)]
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B1]
// CHECK-NEXT:   1: CFGScopeBegin(a)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B1.3], class A [2])
// CHECK-NEXT:   3: A a[2];
// CHECK-NEXT:   4:  (CXXConstructExpr, [B1.5], class A [0])
// CHECK-NEXT:   5: A b[0];
// CHECK-NEXT:   6: [B1.3].~A() (Implicit destructor)
// CHECK-NEXT:   7: CFGScopeEnd(a)
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
// CHECK-NEXT:   1: CFGScopeBegin(a)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B1.3], class A)
// CHECK-NEXT:   3: A a;
// CHECK-NEXT:   4: CFGScopeBegin(c)
// CHECK-NEXT:   5:  (CXXConstructExpr, [B1.6], class A)
// CHECK-NEXT:   6: A c;
// CHECK-NEXT:   7:  (CXXConstructExpr, [B1.8], class A)
// CHECK-NEXT:   8: A d;
// CHECK-NEXT:   9: [B1.8].~A() (Implicit destructor)
// CHECK-NEXT:  10: [B1.6].~A() (Implicit destructor)
// CHECK-NEXT:  11: CFGScopeEnd(c)
// CHECK-NEXT:  12:  (CXXConstructExpr, [B1.13], class A)
// CHECK-NEXT:  13: A b;
// CHECK-NEXT:  14: [B1.13].~A() (Implicit destructor)
// CHECK-NEXT:  15: [B1.3].~A() (Implicit destructor)
// CHECK-NEXT:  16: CFGScopeEnd(a)
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
// CHECK-NEXT:   1:  (CXXConstructExpr, [B1.2], class A)
// CHECK-NEXT:   2: A c;
// CHECK-NEXT:   3: [B1.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B3.5].~A() (Implicit destructor)
// CHECK-NEXT:   5: [B3.3].~A() (Implicit destructor)
// CHECK-NEXT:   6: CFGScopeEnd(a)
// CHECK-NEXT:   Preds (1): B3
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   1: return;
// CHECK-NEXT:   2: [B3.5].~A() (Implicit destructor)
// CHECK-NEXT:   3: [B3.3].~A() (Implicit destructor)
// CHECK-NEXT:   4: CFGScopeEnd(a)
// CHECK-NEXT:   Preds (1): B3
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B3]
// CHECK-NEXT:   1: CFGScopeBegin(a)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B3.3], class A)
// CHECK-NEXT:   3: A a;
// CHECK-NEXT:   4:  (CXXConstructExpr, [B3.5], class A)
// CHECK-NEXT:   5: A b;
// CHECK-NEXT:   6: UV
// CHECK-NEXT:   7: [B3.6] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B3.7]
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

// CHECK:      [B5 (ENTRY)]
// CHECK-NEXT:   Succs (1): B4
// CHECK:      [B1]
// CHECK-NEXT:   1: [B4.8].~A() (Implicit destructor)
// CHECK-NEXT:   2: CFGScopeEnd(b)
// CHECK-NEXT:   3: [B4.3].~A() (Implicit destructor)
// CHECK-NEXT:   4: CFGScopeEnd(a)
// CHECK-NEXT:   Preds (2): B2 B3
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   1: CFGScopeBegin(c)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B2.3], class A)
// CHECK-NEXT:   3: A c;
// CHECK-NEXT:   4: [B2.3].~A() (Implicit destructor)
// CHECK-NEXT:   5: CFGScopeEnd(c)
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B3]
// CHECK-NEXT:   1: CFGScopeBegin(c)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B3.3], class A)
// CHECK-NEXT:   3: A c;
// CHECK-NEXT:   4: [B3.3].~A() (Implicit destructor)
// CHECK-NEXT:   5: CFGScopeEnd(c)
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B4]
// CHECK-NEXT:   1: CFGScopeBegin(a)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B4.3], class A)
// CHECK-NEXT:   3: A a;
// CHECK-NEXT:   4: CFGScopeBegin(b)
// CHECK-NEXT:   5: a
// CHECK-NEXT:   6: [B4.5] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   7: [B4.6] (CXXConstructExpr, [B4.8], class A)
// CHECK-NEXT:   8: A b = a;
// CHECK-NEXT:   9: b
// CHECK-NEXT:  10: [B4.9] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:  11: [B4.10].operator int
// CHECK-NEXT:  12: [B4.10]
// CHECK-NEXT:  13: [B4.12] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK-NEXT:  14: [B4.13] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:   T: if [B4.14]
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
// CHECK-NEXT:   1: [B8.8].~A() (Implicit destructor)
// CHECK-NEXT:   2: CFGScopeEnd(b)
// CHECK-NEXT:   3:  (CXXConstructExpr, [B1.4], class A)
// CHECK-NEXT:   4: A e;
// CHECK-NEXT:   5: [B1.4].~A() (Implicit destructor)
// CHECK-NEXT:   6: [B8.3].~A() (Implicit destructor)
// CHECK-NEXT:   7: CFGScopeEnd(a)
// CHECK-NEXT:   Preds (2): B2 B5
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   1:  (CXXConstructExpr, [B2.2], class A)
// CHECK-NEXT:   2: A d;
// CHECK-NEXT:   3: [B2.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B4.3].~A() (Implicit destructor)
// CHECK-NEXT:   5: CFGScopeEnd(c)
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B3]
// CHECK-NEXT:   1: return;
// CHECK-NEXT:   2: [B4.3].~A() (Implicit destructor)
// CHECK-NEXT:   3: CFGScopeEnd(c)
// CHECK-NEXT:   4: [B8.8].~A() (Implicit destructor)
// CHECK-NEXT:   5: CFGScopeEnd(b)
// CHECK-NEXT:   6: [B8.3].~A() (Implicit destructor)
// CHECK-NEXT:   7: CFGScopeEnd(a)
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B4]
// CHECK-NEXT:   1: CFGScopeBegin(c)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B4.3], class A)
// CHECK-NEXT:   3: A c;
// CHECK-NEXT:   4: UV
// CHECK-NEXT:   5: [B4.4] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B4.5]
// CHECK-NEXT:   Preds (1): B8
// CHECK-NEXT:   Succs (2): B3 B2
// CHECK:      [B5]
// CHECK-NEXT:   1:  (CXXConstructExpr, [B5.2], class A)
// CHECK-NEXT:   2: A d;
// CHECK-NEXT:   3: [B5.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B7.3].~A() (Implicit destructor)
// CHECK-NEXT:   5: CFGScopeEnd(c)
// CHECK-NEXT:   Preds (1): B7
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B6]
// CHECK-NEXT:   1: return;
// CHECK-NEXT:   2: [B7.3].~A() (Implicit destructor)
// CHECK-NEXT:   3: CFGScopeEnd(c)
// CHECK-NEXT:   4: [B8.8].~A() (Implicit destructor)
// CHECK-NEXT:   5: CFGScopeEnd(b)
// CHECK-NEXT:   6: [B8.3].~A() (Implicit destructor)
// CHECK-NEXT:   7: CFGScopeEnd(a)
// CHECK-NEXT:   Preds (1): B7
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B7]
// CHECK-NEXT:   1: CFGScopeBegin(c)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B7.3], class A)
// CHECK-NEXT:   3: A c;
// CHECK-NEXT:   4: UV
// CHECK-NEXT:   5: [B7.4] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B7.5]
// CHECK-NEXT:   Preds (1): B8
// CHECK-NEXT:   Succs (2): B6 B5
// CHECK:      [B8]
// CHECK-NEXT:   1: CFGScopeBegin(a)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B8.3], class A)
// CHECK-NEXT:   3: A a;
// CHECK-NEXT:   4: CFGScopeBegin(b)
// CHECK-NEXT:   5: a
// CHECK-NEXT:   6: [B8.5] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   7: [B8.6] (CXXConstructExpr, [B8.8], class A)
// CHECK-NEXT:   8: A b = a;
// CHECK-NEXT:   9: b
// CHECK-NEXT:  10: [B8.9] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:  11: [B8.10].operator int
// CHECK-NEXT:  12: [B8.10]
// CHECK-NEXT:  13: [B8.12] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK-NEXT:  14: [B8.13] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:   T: if [B8.14]
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
// CHECK-NEXT:   1: [B4.5].~A() (Implicit destructor)
// CHECK-NEXT:   2: CFGScopeEnd(b)
// CHECK-NEXT:   3: [B5.3].~A() (Implicit destructor)
// CHECK-NEXT:   4: CFGScopeEnd(a)
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   Preds (1): B3
// CHECK-NEXT:   Succs (1): B4
// CHECK:      [B3]
// CHECK-NEXT:   1: CFGScopeBegin(c)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B3.3], class A)
// CHECK-NEXT:   3: A c;
// CHECK-NEXT:   4: [B3.3].~A() (Implicit destructor)
// CHECK-NEXT:   5: CFGScopeEnd(c)
// CHECK-NEXT:   6: [B4.5].~A() (Implicit destructor)
// CHECK-NEXT:   7: CFGScopeEnd(b)
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B4]
// CHECK-NEXT:   1: CFGScopeBegin(b)
// CHECK-NEXT:   2: a
// CHECK-NEXT:   3: [B4.2] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   4: [B4.3] (CXXConstructExpr, class A)
// CHECK-NEXT:   5: A b = a;
// CHECK-NEXT:   6: b
// CHECK-NEXT:   7: [B4.6] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   8: [B4.7].operator int
// CHECK-NEXT:   9: [B4.7]
// CHECK-NEXT:  10: [B4.9] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK-NEXT:  11: [B4.10] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:   T: while [B4.11]
// CHECK-NEXT:   Preds (2): B2 B5
// CHECK-NEXT:   Succs (2): B3 B1
// CHECK:      [B5]
// CHECK-NEXT:   1: CFGScopeBegin(a)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B5.3], class A)
// CHECK-NEXT:   3: A a;
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
// CHECK-NEXT:   1: [B10.5].~A() (Implicit destructor)
// CHECK-NEXT:   2: CFGScopeEnd(b)
// CHECK-NEXT:   3:  (CXXConstructExpr, [B1.4], class A)
// CHECK-NEXT:   4: A e;
// CHECK-NEXT:   5: [B1.4].~A() (Implicit destructor)
// CHECK-NEXT:   6: [B11.3].~A() (Implicit destructor)
// CHECK-NEXT:   7: CFGScopeEnd(a)
// CHECK-NEXT:   Preds (2): B8 B10
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   Preds (2): B3 B6
// CHECK-NEXT:   Succs (1): B10
// CHECK:      [B3]
// CHECK-NEXT:   1:  (CXXConstructExpr, [B3.2], class A)
// CHECK-NEXT:   2: A d;
// CHECK-NEXT:   3: [B3.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B9.3].~A() (Implicit destructor)
// CHECK-NEXT:   5: CFGScopeEnd(c)
// CHECK-NEXT:   6: [B10.5].~A() (Implicit destructor)
// CHECK-NEXT:   7: CFGScopeEnd(b)
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B4]
// CHECK-NEXT:   1: return;
// CHECK-NEXT:   2: [B9.3].~A() (Implicit destructor)
// CHECK-NEXT:   3: CFGScopeEnd(c)
// CHECK-NEXT:   4: [B10.5].~A() (Implicit destructor)
// CHECK-NEXT:   5: CFGScopeEnd(b)
// CHECK-NEXT:   6: [B11.3].~A() (Implicit destructor)
// CHECK-NEXT:   7: CFGScopeEnd(a)
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B5]
// CHECK-NEXT:   1: UV
// CHECK-NEXT:   2: [B5.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B5.2]
// CHECK-NEXT:   Preds (1): B7
// CHECK-NEXT:   Succs (2): B4 B3
// CHECK:      [B6]
// CHECK-NEXT:   1: [B9.3].~A() (Implicit destructor)
// CHECK-NEXT:   2: CFGScopeEnd(c)
// CHECK-NEXT:   3: [B10.5].~A() (Implicit destructor)
// CHECK-NEXT:   4: CFGScopeEnd(b)
// CHECK-NEXT:   T: continue;
// CHECK-NEXT:   Preds (1): B7
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B7]
// CHECK-NEXT:   1: UV
// CHECK-NEXT:   2: [B7.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B7.2]
// CHECK-NEXT:   Preds (1): B9
// CHECK-NEXT:   Succs (2): B6 B5
// CHECK:      [B8]
// CHECK-NEXT:   1: [B9.3].~A() (Implicit destructor)
// CHECK-NEXT:   2: CFGScopeEnd(c)
// CHECK-NEXT:   T: break;
// CHECK-NEXT:   Preds (1): B9
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B9]
// CHECK-NEXT:   1: CFGScopeBegin(c)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B9.3], class A)
// CHECK-NEXT:   3: A c;
// CHECK-NEXT:   4: UV
// CHECK-NEXT:   5: [B9.4] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B9.5]
// CHECK-NEXT:   Preds (1): B10
// CHECK-NEXT:   Succs (2): B8 B7
// CHECK:      [B10]
// CHECK-NEXT:   1: CFGScopeBegin(b)
// CHECK-NEXT:   2: a
// CHECK-NEXT:   3: [B10.2] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   4: [B10.3] (CXXConstructExpr, class A)
// CHECK-NEXT:   5: A b = a;
// CHECK-NEXT:   6: b
// CHECK-NEXT:   7: [B10.6] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   8: [B10.7].operator int
// CHECK-NEXT:   9: [B10.7]
// CHECK-NEXT:  10: [B10.9] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK-NEXT:  11: [B10.10] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:   T: while [B10.11]
// CHECK-NEXT:   Preds (2): B2 B11
// CHECK-NEXT:   Succs (2): B9 B1
// CHECK:      [B11]
// CHECK-NEXT:   1: CFGScopeBegin(a)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B11.3], class A)
// CHECK-NEXT:   3: A a;
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

// CHECK:      [B12 (ENTRY)]
// CHECK-NEXT:   Succs (1): B11
// CHECK:      [B1]
// CHECK-NEXT:   1:  (CXXConstructExpr, [B1.2], class A)
// CHECK-NEXT:   2: A d;
// CHECK-NEXT:   3: [B1.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B11.3].~A() (Implicit destructor)
// CHECK-NEXT:   5: CFGScopeEnd(a)
// CHECK-NEXT:   Preds (2): B8 B2
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   1: UV
// CHECK-NEXT:   2: [B2.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: do ... while [B2.2]
// CHECK-NEXT:   Preds (2): B3 B6
// CHECK-NEXT:   Succs (2): B10 B1
// CHECK:      [B3]
// CHECK-NEXT:   1:  (CXXConstructExpr, [B3.2], class A)
// CHECK-NEXT:   2: A c;
// CHECK-NEXT:   3: [B3.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B9.3].~A() (Implicit destructor)
// CHECK-NEXT:   5: CFGScopeEnd(b)
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B4]
// CHECK-NEXT:   1: return;
// CHECK-NEXT:   2: [B9.3].~A() (Implicit destructor)
// CHECK-NEXT:   3: CFGScopeEnd(b)
// CHECK-NEXT:   4: [B11.3].~A() (Implicit destructor)
// CHECK-NEXT:   5: CFGScopeEnd(a)
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B5]
// CHECK-NEXT:   1: UV
// CHECK-NEXT:   2: [B5.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B5.2]
// CHECK-NEXT:   Preds (1): B7
// CHECK-NEXT:   Succs (2): B4 B3
// CHECK:      [B6]
// CHECK-NEXT:   1: [B9.3].~A() (Implicit destructor)
// CHECK-NEXT:   2: CFGScopeEnd(b)
// CHECK-NEXT:   T: continue;
// CHECK-NEXT:   Preds (1): B7
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B7]
// CHECK-NEXT:   1: UV
// CHECK-NEXT:   2: [B7.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B7.2]
// CHECK-NEXT:   Preds (1): B9
// CHECK-NEXT:   Succs (2): B6 B5
// CHECK:      [B8]
// CHECK-NEXT:   1: [B9.3].~A() (Implicit destructor)
// CHECK-NEXT:   2: CFGScopeEnd(b)
// CHECK-NEXT:   T: break;
// CHECK-NEXT:   Preds (1): B9
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B9]
// CHECK-NEXT:   1: CFGScopeBegin(b)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B9.3], class A)
// CHECK-NEXT:   3: A b;
// CHECK-NEXT:   4: UV
// CHECK-NEXT:   5: [B9.4] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B9.5]
// CHECK-NEXT:   Preds (2): B10 B11
// CHECK-NEXT:   Succs (2): B8 B7
// CHECK:      [B10]
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B9
// CHECK:      [B11]
// CHECK-NEXT:   1: CFGScopeBegin(a)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B11.3], class A)
// CHECK-NEXT:   3: A a;
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

// CHECK:      [B6 (ENTRY)]
// CHECK-NEXT:   Succs (1): B5
// CHECK:      [B1]
// CHECK-NEXT:   1: [B4.5].~A() (Implicit destructor)
// CHECK-NEXT:   2: CFGScopeEnd(b)
// CHECK-NEXT:   3: [B5.3].~A() (Implicit destructor)
// CHECK-NEXT:   4: CFGScopeEnd(a)
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   Preds (1): B3
// CHECK-NEXT:   Succs (1): B4
// CHECK:      [B3]
// CHECK-NEXT:   1: CFGScopeBegin(c)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B3.3], class A)
// CHECK-NEXT:   3: A c;
// CHECK-NEXT:   4: [B3.3].~A() (Implicit destructor)
// CHECK-NEXT:   5: CFGScopeEnd(c)
// CHECK-NEXT:   6: [B4.5].~A() (Implicit destructor)
// CHECK-NEXT:   7: CFGScopeEnd(b)
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B4]
// CHECK-NEXT:   1: CFGScopeBegin(b)
// CHECK-NEXT:   2: a
// CHECK-NEXT:   3: [B4.2] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   4: [B4.3] (CXXConstructExpr, class A)
// CHECK-NEXT:   5: A b = a;
// CHECK-NEXT:   6: b
// CHECK-NEXT:   7: [B4.6] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   8: [B4.7].operator int
// CHECK-NEXT:   9: [B4.7]
// CHECK-NEXT:  10: [B4.9] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK-NEXT:  11: [B4.10] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:   T: for (...; [B4.11]; )
// CHECK-NEXT:   Preds (2): B2 B5
// CHECK-NEXT:   Succs (2): B3 B1
// CHECK:      [B5]
// CHECK-NEXT:   1: CFGScopeBegin(a)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B5.3], class A)
// CHECK-NEXT:   3: A a;
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
// CHECK-NEXT:   1: [B10.5].~A() (Implicit destructor)
// CHECK-NEXT:   2: CFGScopeEnd(c)
// CHECK-NEXT:   3: [B11.6].~A() (Implicit destructor)
// CHECK-NEXT:   4: CFGScopeEnd(b)
// CHECK-NEXT:   5:  (CXXConstructExpr, [B1.6], class A)
// CHECK-NEXT:   6: A f;
// CHECK-NEXT:   7: [B1.6].~A() (Implicit destructor)
// CHECK-NEXT:   8: [B11.3].~A() (Implicit destructor)
// CHECK-NEXT:   9: CFGScopeEnd(a)
// CHECK-NEXT:   Preds (2): B8 B10
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   Preds (2): B3 B6
// CHECK-NEXT:   Succs (1): B10
// CHECK:      [B3]
// CHECK-NEXT:   1:  (CXXConstructExpr, [B3.2], class A)
// CHECK-NEXT:   2: A e;
// CHECK-NEXT:   3: [B3.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B9.3].~A() (Implicit destructor)
// CHECK-NEXT:   5: CFGScopeEnd(d)
// CHECK-NEXT:   6: [B10.5].~A() (Implicit destructor)
// CHECK-NEXT:   7: CFGScopeEnd(c)
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B4]
// CHECK-NEXT:   1: return;
// CHECK-NEXT:   2: [B9.3].~A() (Implicit destructor)
// CHECK-NEXT:   3: CFGScopeEnd(d)
// CHECK-NEXT:   4: [B10.5].~A() (Implicit destructor)
// CHECK-NEXT:   5: CFGScopeEnd(c)
// CHECK-NEXT:   6: [B11.6].~A() (Implicit destructor)
// CHECK-NEXT:   7: CFGScopeEnd(b)
// CHECK-NEXT:   8: [B11.3].~A() (Implicit destructor)
// CHECK-NEXT:   9: CFGScopeEnd(a)
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B5]
// CHECK-NEXT:   1: UV
// CHECK-NEXT:   2: [B5.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B5.2]
// CHECK-NEXT:   Preds (1): B7
// CHECK-NEXT:   Succs (2): B4 B3
// CHECK:      [B6]
// CHECK-NEXT:   1: [B9.3].~A() (Implicit destructor)
// CHECK-NEXT:   2: CFGScopeEnd(d)
// CHECK-NEXT:   T: continue;
// CHECK-NEXT:   Preds (1): B7
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B7]
// CHECK-NEXT:   1: UV
// CHECK-NEXT:   2: [B7.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B7.2]
// CHECK-NEXT:   Preds (1): B9
// CHECK-NEXT:   Succs (2): B6 B5
// CHECK:      [B8]
// CHECK-NEXT:   1: [B9.3].~A() (Implicit destructor)
// CHECK-NEXT:   2: CFGScopeEnd(d)
// CHECK-NEXT:   T: break;
// CHECK-NEXT:   Preds (1): B9
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B9]
// CHECK-NEXT:   1: CFGScopeBegin(d)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B9.3], class A)
// CHECK-NEXT:   3: A d;
// CHECK-NEXT:   4: UV
// CHECK-NEXT:   5: [B9.4] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B9.5]
// CHECK-NEXT:   Preds (1): B10
// CHECK-NEXT:   Succs (2): B8 B7
// CHECK:      [B10]
// CHECK-NEXT:   1: CFGScopeBegin(c)
// CHECK-NEXT:   2: b
// CHECK-NEXT:   3: [B10.2] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   4: [B10.3] (CXXConstructExpr, class A)
// CHECK-NEXT:   5: A c = b;
// CHECK-NEXT:   6: c
// CHECK-NEXT:   7: [B10.6] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   8: [B10.7].operator int
// CHECK-NEXT:   9: [B10.7]
// CHECK-NEXT:  10: [B10.9] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK-NEXT:  11: [B10.10] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:   T: for (...; [B10.11]; )
// CHECK-NEXT:   Preds (2): B2 B11
// CHECK-NEXT:   Succs (2): B9 B1
// CHECK:      [B11]
// CHECK-NEXT:   1: CFGScopeBegin(a)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B11.3], class A)
// CHECK-NEXT:   3: A a;
// CHECK-NEXT:   4: CFGScopeBegin(b)
// CHECK-NEXT:   5:  (CXXConstructExpr, [B11.6], class A)
// CHECK-NEXT:   6: A b;
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

// CHECK:      [B8 (ENTRY)]
// CHECK-NEXT:   Succs (1): B7
// CHECK:      [B1]
// CHECK-NEXT:  l1:
// CHECK-NEXT:   1:  (CXXConstructExpr, [B1.2], class A)
// CHECK-NEXT:   2: A c;
// CHECK-NEXT:   3: [B1.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B6.5].~A() (Implicit destructor)
// CHECK-NEXT:   5: [B6.3].~A() (Implicit destructor)
// CHECK-NEXT:   6: [B7.3].~A() (Implicit destructor)
// CHECK-NEXT:   7: CFGScopeEnd(a)
// CHECK-NEXT:   Preds (2): B2 B3
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   1:  (CXXConstructExpr, [B2.2], class A)
// CHECK-NEXT:   2: A b;
// CHECK-NEXT:   3: [B2.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B6.8].~A() (Implicit destructor)
// CHECK-NEXT:   5: CFGScopeEnd(a)
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B3]
// CHECK-NEXT:   1: [B6.8].~A() (Implicit destructor)
// CHECK-NEXT:   2: CFGScopeEnd(a)
// CHECK-NEXT:   T: goto l1;
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B4]
// CHECK-NEXT:   1: UV
// CHECK-NEXT:   2: [B4.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B4.2]
// CHECK-NEXT:   Preds (1): B6
// CHECK-NEXT:   Succs (2): B3 B2
// CHECK:      [B5]
// CHECK-NEXT:   1: [B6.8].~A() (Implicit destructor)
// CHECK-NEXT:   2: [B6.5].~A() (Implicit destructor)
// CHECK-NEXT:   3: [B6.3].~A() (Implicit destructor)
// CHECK-NEXT:   4: CFGScopeEnd(cb)
// CHECK-NEXT:   T: goto l0;
// CHECK-NEXT:   Preds (1): B6
// CHECK-NEXT:   Succs (1): B6
// CHECK:      [B6]
// CHECK-NEXT:  l0:
// CHECK-NEXT:   1: CFGScopeBegin(cb)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B6.3], class A)
// CHECK-NEXT:   3: A cb;
// CHECK-NEXT:   4:  (CXXConstructExpr, [B6.5], class A)
// CHECK-NEXT:   5: A b;
// CHECK-NEXT:   6: CFGScopeBegin(a)
// CHECK-NEXT:   7:  (CXXConstructExpr, [B6.8], class A)
// CHECK-NEXT:   8: A a;
// CHECK-NEXT:   9: UV
// CHECK-NEXT:  10: [B6.9] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:   T: if [B6.10]
// CHECK-NEXT:   Preds (2): B7 B5
// CHECK-NEXT:   Succs (2): B5 B4
// CHECK:      [B7]
// CHECK-NEXT:   1: CFGScopeBegin(a)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B7.3], class A)
// CHECK-NEXT:   3: A a;
// CHECK-NEXT:   Preds (1): B8
// CHECK-NEXT:   Succs (1): B6
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void test_goto() {
  A a;
l0:
  A cb;
  A b;
  { A a;
    if (UV) goto l0;
    if (UV) goto l1;
    A b;
  }
l1:
  A c;
}

// CHECK:      [B7 (ENTRY)]
// CHECK-NEXT:   Succs (1): B6
// CHECK:      [B1]
// CHECK-NEXT:   1: CFGScopeEnd(i)
// CHECK-NEXT:   2: CFGScopeBegin(unused2)
// CHECK-NEXT:   3: int unused2;
// CHECK-NEXT:   4: CFGScopeEnd(unused2)
// CHECK-NEXT:   Preds (2): B4 B5
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   1: i
// CHECK-NEXT:   2: ++[B2.1]
// CHECK-NEXT:   Preds (1): B3
// CHECK-NEXT:   Succs (1): B5
// CHECK:      [B3]
// CHECK-NEXT:   1: CFGScopeEnd(unused1)
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B4]
// CHECK-NEXT:   1: CFGScopeBegin(unused1)
// CHECK-NEXT:   2: int unused1;
// CHECK-NEXT:   3: CFGScopeEnd(unused1)
// CHECK-NEXT:   T: break;
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B5]
// CHECK-NEXT:   1: i
// CHECK-NEXT:   2: [B5.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: 3
// CHECK-NEXT:   4: [B5.2] < [B5.3]
// CHECK-NEXT:   T: for (...; [B5.4]; ...)
// CHECK-NEXT:   Preds (2): B2 B6
// CHECK-NEXT:   Succs (2): B4 B1
// CHECK:      [B6]
// CHECK-NEXT:   1: CFGScopeBegin(i)
// CHECK-NEXT:   2: 0
// CHECK-NEXT:   3: int i = 0;
// CHECK-NEXT:   Preds (1): B7
// CHECK-NEXT:   Succs (1): B5
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void test_for_compound_and_break() {
  for (int i = 0; i < 3; ++i) {
    {
      int unused1;
      break;
    }
  }
  {
    int unused2;
  }
}

// CHECK:      [B6 (ENTRY)]
// CHECK-NEXT:   Succs (1): B5
// CHECK:      [B1]
// CHECK-NEXT:   1: CFGScopeEnd(__end1)
// CHECK-NEXT:   2: CFGScopeEnd(__begin1)
// CHECK-NEXT:   3: CFGScopeEnd(__range1)
// CHECK-NEXT:   4: [B5.3].~A() (Implicit destructor)
// CHECK-NEXT:   5: CFGScopeEnd(a)
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   1: __begin1
// CHECK-NEXT:   2: [B2.1] (ImplicitCastExpr, LValueToRValue, class A *)
// CHECK-NEXT:   3: __end1
// CHECK-NEXT:   4: [B2.3] (ImplicitCastExpr, LValueToRValue, class A *)
// CHECK-NEXT:   5: [B2.2] != [B2.4]
// CHECK-NEXT:   T: for (auto &i : [B5.4]) {
// CHECK:         [B4.11];
// CHECK-NEXT:}
// CHECK-NEXT:   Preds (2): B3 B5
// CHECK-NEXT:   Succs (2): B4 B1
// CHECK:      [B3]
// CHECK-NEXT:   1: __begin1
// CHECK-NEXT:   2: ++[B3.1]
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B4]
// CHECK-NEXT:   1: CFGScopeBegin(i)
// CHECK-NEXT:   2: __begin1
// CHECK-NEXT:   3: [B4.2] (ImplicitCastExpr, LValueToRValue, class A *)
// CHECK-NEXT:   4: *[B4.3]
// CHECK-NEXT:   5: auto &i = *__begin1;
// CHECK-NEXT:   6: operator=
// CHECK-NEXT:   7: [B4.6] (ImplicitCastExpr, FunctionToPointerDecay, class A &(*)(const class A &)
// CHECK-NEXT:   8: i
// CHECK-NEXT:   9: b
// CHECK-NEXT:  10: [B4.9] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:  11: [B4.8] = [B4.10] (OperatorCall)
// CHECK-NEXT:  12: CFGScopeEnd(i)
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B3
// CHECK:      [B5]
// CHECK-NEXT:   1: CFGScopeBegin(a)
// CHECK-NEXT:   2:  (CXXConstructExpr, [B5.3], class A [10])
// CHECK-NEXT:   3: A a[10];
// CHECK-NEXT:   4: a
// CHECK-NEXT:   5: auto &&__range1 = a;
// CHECK-NEXT:   6: CFGScopeBegin(__end1)
// CHECK-NEXT:   7: __range1
// CHECK-NEXT:   8: [B5.7] (ImplicitCastExpr, ArrayToPointerDecay, class A *)
// CHECK-NEXT:   9: 10
// CHECK-NEXT:  10: [B5.8] + [B5.9]
// CHECK-NEXT:  11: auto __end1 = __range1 + 10
// CHECK-NEXT:  12: __range1
// CHECK-NEXT:  13: [B5.12] (ImplicitCastExpr, ArrayToPointerDecay, class A *)
// CHECK-NEXT:  14: auto __begin1 = __range1;
// CHECK-NEXT:   Preds (1): B6
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void test_range_for(A &b) {
  A a[10];
  for (auto &i : a)
    i = b;
}

// CHECK:      [B8 (ENTRY)]
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B1]
// CHECK-NEXT:   1: CFGScopeEnd(i)
// CHECK-NEXT:   2: 1
// CHECK-NEXT:   3: int k = 1;
// CHECK-NEXT:   4: CFGScopeEnd(c)
// CHECK-NEXT:   Preds (3): B3 B5 B6
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   1: CFGScopeBegin(c)
// CHECK-NEXT:   2: '1'
// CHECK-NEXT:   3: char c = '1';
// CHECK-NEXT:   4: CFGScopeBegin(i)
// CHECK-NEXT:   5: getX
// CHECK-NEXT:   6: [B2.5] (ImplicitCastExpr, FunctionToPointerDecay, int (*)(void))
// CHECK-NEXT:   7: [B2.6]()
// CHECK-NEXT:   8: int i = getX();
// CHECK-NEXT:   9: i
// CHECK-NEXT:  10: [B2.9] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   T: switch [B2.10]
// CHECK-NEXT:   Preds (1): B8
// CHECK-NEXT:   Succs (5): B4 B5 B6 B7 B3
// CHECK:      [B3]
// CHECK-NEXT:  default:
// CHECK-NEXT:   1: CFGScopeBegin(a)
// CHECK-NEXT:   2: 0
// CHECK-NEXT:   3: int a = 0;
// CHECK-NEXT:   4: i
// CHECK-NEXT:   5: ++[B3.4]
// CHECK-NEXT:   6: CFGScopeEnd(a)
// CHECK-NEXT:   Preds (2): B4 B2
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B4]
// CHECK-NEXT:  case 3:
// CHECK-NEXT:   1: '2'
// CHECK-NEXT:   2: c
// CHECK-NEXT:   3: [B4.2] = [B4.1]
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B3
// CHECK:      [B5]
// CHECK-NEXT:  case 2:
// CHECK-NEXT:   1: '2'
// CHECK-NEXT:   2: c
// CHECK-NEXT:   3: [B5.2] = [B5.1]
// CHECK-NEXT:   T: break;
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B6]
// CHECK-NEXT:  case 1:
// CHECK-NEXT:   1: '3'
// CHECK-NEXT:   2: c
// CHECK-NEXT:   3: [B6.2] = [B6.1]
// CHECK-NEXT:   T: break;
// CHECK-NEXT:   Preds (2): B2 B7
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B7]
// CHECK-NEXT:  case 0:
// CHECK-NEXT:   1: '2'
// CHECK-NEXT:   2: c
// CHECK-NEXT:   3: [B7.2] = [B7.1]
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B6
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void test_switch_with_compound_with_default() {
  char c = '1';
  switch (int i = getX()) {
    case 0:
      c = '2';
    case 1:
      c = '3';
      break;
    case 2: {
      c = '2';
      break;
    }
    case 3:
      c = '2';
    default: {
      int a = 0;
      ++i;
    }
    }
  int k = 1;
}

// CHECK:      [B6 (ENTRY)]
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B1]
// CHECK-NEXT:   1: CFGScopeEnd(i)
// CHECK-NEXT:   2: 3
// CHECK-NEXT:   3: int k = 3;
// CHECK-NEXT:   4: CFGScopeEnd(c)
// CHECK-NEXT:   Preds (3): B3 B4 B2
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   1: CFGScopeBegin(c)
// CHECK-NEXT:   2: '1'
// CHECK-NEXT:   3: char c = '1';
// CHECK-NEXT:   4: CFGScopeBegin(i)
// CHECK-NEXT:   5: getX
// CHECK-NEXT:   6: [B2.5] (ImplicitCastExpr, FunctionToPointerDecay, int (*)(void))
// CHECK-NEXT:   7: [B2.6]()
// CHECK-NEXT:   8: int i = getX();
// CHECK-NEXT:   9: i
// CHECK-NEXT:  10: [B2.9] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   T: switch [B2.10]
// CHECK-NEXT:   Preds (1): B6
// CHECK-NEXT:   Succs (4): B3 B4 B5 B1
// CHECK:      [B3]
// CHECK-NEXT:  case 2:
// CHECK-NEXT:   1: '3'
// CHECK-NEXT:   2: c
// CHECK-NEXT:   3: [B3.2] = [B3.1]
// CHECK-NEXT:   T: break;
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B4]
// CHECK-NEXT:  case 1:
// CHECK-NEXT:   1: '1'
// CHECK-NEXT:   2: c
// CHECK-NEXT:   3: [B4.2] = [B4.1]
// CHECK-NEXT:   T: break;
// CHECK-NEXT:   Preds (2): B2 B5
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B5]
// CHECK-NEXT:  case 0:
// CHECK-NEXT:   1: '2'
// CHECK-NEXT:   2: c
// CHECK-NEXT:   3: [B5.2] = [B5.1]
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B4
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
int test_switch_with_compound_without_default() {
  char c = '1';
  switch (int i = getX()) {
    case 0:
      c = '2';
    case 1:
      c = '1';
      break;
    case 2:
      c = '3';
      break;
   }
  int k = 3;
}

// CHECK:      [B5 (ENTRY)]
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B1]
// CHECK-NEXT:   1: CFGScopeEnd(i)
// CHECK-NEXT:   2: 1
// CHECK-NEXT:   3: int k = 1;
// CHECK-NEXT:   4: CFGScopeEnd(s)
// CHECK-NEXT:   Preds (1): B3
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   1: CFGScopeBegin(s)
// CHECK-NEXT:   2: '1'
// CHECK-NEXT:   3: char s = '1';
// CHECK-NEXT:   4: CFGScopeBegin(i)
// CHECK-NEXT:   5: getX
// CHECK-NEXT:   6: [B2.5] (ImplicitCastExpr, FunctionToPointerDecay, int (*)(void))
// CHECK-NEXT:   7: [B2.6]()
// CHECK-NEXT:   8: int i = getX();
// CHECK-NEXT:   9: i
// CHECK-NEXT:  10: [B2.9] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   T: switch [B2.10]
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (2): B4 B3
// CHECK:      [B3]
// CHECK-NEXT:  default:
// CHECK-NEXT:   1: CFGScopeBegin(a)
// CHECK-NEXT:   2: 0
// CHECK-NEXT:   3: int a = 0;
// CHECK-NEXT:   4: i
// CHECK-NEXT:   5: ++[B3.4]
// CHECK-NEXT:   6: CFGScopeEnd(a)
// CHECK-NEXT:   Preds (2): B4 B2
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B4]
// CHECK-NEXT:  case 0:
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B3
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void test_without_compound() {
  char s = '1';
  switch (int i = getX())
    case 0:
    default: {
      int a = 0;
      ++i;
    }
  int k = 1;
}

// CHECK:      [B12 (ENTRY)]
// CHECK-NEXT:   Succs (1): B11
// CHECK:      [B1]
// CHECK-NEXT:   1: CFGScopeEnd(i)
// CHECK-NEXT:   Preds (2): B4 B10
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   1: i
// CHECK-NEXT:   2: ++[B2.1]
// CHECK-NEXT:   Preds (2): B3 B7
// CHECK-NEXT:   Succs (1): B10
// CHECK:      [B3]
// CHECK-NEXT:   1: CFGScopeEnd(z)
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B4]
// CHECK-NEXT:   1: CFGScopeBegin(z)
// CHECK-NEXT:   2: 5
// CHECK-NEXT:   3: int z = 5;
// CHECK-NEXT:   4: CFGScopeEnd(z)
// CHECK-NEXT:   T: break;
// CHECK-NEXT:   Preds (2): B6 B8
// CHECK-NEXT:   Succs (1): B1
// CHECK:      [B5]
// CHECK-NEXT:   1: x
// CHECK-NEXT:   2: [B5.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   T: switch [B5.2]
// CHECK-NEXT:   Preds (1): B10
// CHECK-NEXT:   Succs (4): B7 B8 B9 B6
// CHECK:      [B6]
// CHECK-NEXT:  default:
// CHECK-NEXT:   1: 3
// CHECK-NEXT:   2: y
// CHECK-NEXT:   3: [B6.2] = [B6.1]
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (1): B4
// CHECK:      [B7]
// CHECK-NEXT:  case 2:
// CHECK-NEXT:   1: 4
// CHECK-NEXT:   2: y
// CHECK-NEXT:   3: [B7.2] = [B7.1]
// CHECK-NEXT:   T: continue;
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B8]
// CHECK-NEXT:  case 1:
// CHECK-NEXT:   1: 2
// CHECK-NEXT:   2: y
// CHECK-NEXT:   3: [B8.2] = [B8.1]
// CHECK-NEXT:   T: break;
// CHECK-NEXT:   Preds (2): B5 B9
// CHECK-NEXT:   Succs (1): B4
// CHECK:      [B9]
// CHECK-NEXT:  case 0:
// CHECK-NEXT:   1: 1
// CHECK-NEXT:   2: y
// CHECK-NEXT:   3: [B9.2] = [B9.1]
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (1): B8
// CHECK:      [B10]
// CHECK-NEXT:   1: i
// CHECK-NEXT:   2: [B10.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: 1000
// CHECK-NEXT:   4: [B10.2] < [B10.3]
// CHECK-NEXT:   T: for (...; [B10.4]; ...)
// CHECK-NEXT:   Preds (2): B2 B11
// CHECK-NEXT:   Succs (2): B5 B1
// CHECK:      [B11]
// CHECK-NEXT:   1: CFGScopeBegin(i)
// CHECK-NEXT:   2: int i;
// CHECK-NEXT:   3: int x;
// CHECK-NEXT:   4: int y;
// CHECK-NEXT:   5: 0
// CHECK-NEXT:   6: i
// CHECK-NEXT:   7: [B11.6] = [B11.5]
// CHECK-NEXT:   Preds (1): B12
// CHECK-NEXT:   Succs (1): B10
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void test_for_switch_in_for() {
  int i, x, y;
  for (i = 0; i < 1000; ++i) {
    switch (x) {
    case 0:
      y = 1;
    case 1:
      y = 2;
      break; // break from switch
    case 2:
      y = 4;
      continue; // continue in loop
    default:
      y = 3;
    }
    {
      int z = 5;
      break; // break from loop
    }
  }
}
