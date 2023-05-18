// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -analyze -analyzer-checker=debug.DumpCFG -analyzer-config cfg-lifetime=true %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s

// FIXME Most of the cases in this file test only with the objects of type `A`,
// that has a non-trivial destructor. The types with a trivial destructor
// ends their lifetime when the underlying memory is released, i.e. they
// are destroyed last, after all objects with non-trivial destructors.
// Consequently, they are handled differently in code, and we should make 
// sure that test cover also trivially destructible types.

extern bool UV;
class A {
public:
  // CHECK:       [B2 (ENTRY)]
  // CHECK-NEXT:    Succs (1): B1
  // CHECK:       [B1]
  // CHECK-NEXT:    1: true
  // CHECK-NEXT:    2: UV
  // CHECK-NEXT:    3: [B1.2] = [B1.1]
  // CHECK-NEXT:    Preds (1): B2
  // CHECK-NEXT:    Succs (1): B0
  // CHECK:       [B0 (EXIT)]
  // CHECK-NEXT:    Preds (1): B1
  A() {
    UV = true;
  }
  // CHECK:       [B3 (ENTRY)]
  // CHECK-NEXT:    Succs (1): B2
  // CHECK:       [B1]
  // CHECK-NEXT:    1: 0
  // CHECK-NEXT:    2: this
  // CHECK-NEXT:    3: [B1.2]->p
  // CHECK-NEXT:    4: [B1.3] (ImplicitCastExpr, LValueToRValue, int *)
  // CHECK-NEXT:    5: *[B1.4]
  // CHECK-NEXT:    6: [B1.5] = [B1.1]
  // CHECK-NEXT:    Preds (1): B2
  // CHECK-NEXT:    Succs (1): B0
  // CHECK:       [B2]
  // CHECK-NEXT:    1: this
  // CHECK-NEXT:    2: [B2.1]->p
  // CHECK-NEXT:    3: [B2.2] (ImplicitCastExpr, LValueToRValue, int *)
  // CHECK-NEXT:    4: [B2.3] (ImplicitCastExpr, PointerToBoolean, _Bool)
  // CHECK-NEXT:    T: if [B2.4]
  // CHECK-NEXT:    Preds (1): B3
  // CHECK-NEXT:    Succs (2): B1 B0
  // CHECK:       [B0 (EXIT)]
  // CHECK-NEXT:    Preds (2): B1 B2
  ~A() {
    if (p)
      *p = 0;
  }
  // CHECK:       [B2 (ENTRY)]
  // CHECK-NEXT:    Succs (1): B1
  // CHECK:       [B1]
  // CHECK-NEXT:    1: 1
  // CHECK-NEXT:    2: return [B1.1];
  // CHECK-NEXT:    Preds (1): B2
  // CHECK-NEXT:    Succs (1): B0
  // CHECK:       [B0 (EXIT)]
  // CHECK-NEXT:    Preds (1): B1
  operator int() const { return 1; }
  int *p;
};

// CHECK:       [B2 (ENTRY)]
// CHECK-NEXT:    Succs (1): B1
// CHECK:       [B1]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B1.2], A)
// CHECK-NEXT:    2: A a;
// CHECK-NEXT:    3: a
// CHECK-NEXT:    4: [B1.3] (ImplicitCastExpr, NoOp, const A)
// CHECK-NEXT:    5: const A &b = a;
// CHECK-NEXT:    6: A() (CXXConstructExpr, [B1.9], A)
// CHECK-NEXT:    7: [B1.6] (BindTemporary)
// CHECK-NEXT:    8: [B1.7] (ImplicitCastExpr, NoOp, const A)
// CHECK-NEXT:    9: [B1.8]
// CHECK-NEXT:   10: const A &c = A();
// CHECK-NEXT:   11: [B1.10].~A() (Implicit destructor)
// CHECK-NEXT:   12: [B1.10] (Lifetime ends)
// CHECK-NEXT:   13: [B1.2].~A() (Implicit destructor)
// CHECK-NEXT:   14: [B1.2] (Lifetime ends)
// CHECK-NEXT:   15: [B1.5] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B0
// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
void test_const_ref() {
  A a;
  const A &b = a;
  const A &c = A();
}

// CHECK:      [B2 (ENTRY)]
// CHECK-NEXT:   Succs (1): B1
// CHECK:       [B1]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B1.2], A[2])
// CHECK-NEXT:    2: A a[2];
// CHECK-NEXT:    3:  (CXXConstructExpr, [B1.4], A[0])
// CHECK-NEXT:    4: A b[0];
// CHECK-NEXT:    5: [B1.2].~A[2]() (Implicit destructor)
// lifetime of a ends when its destructors are run
// CHECK-NEXT:    6: [B1.2] (Lifetime ends)
// lifetime of b ends when its storage duration ends
// CHECK-NEXT:    7: [B1.4] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B0
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void test_array() {
  A a[2];
  A b[0];
}

// CHECK:      [B2 (ENTRY)]
// CHECK-NEXT:   Succs (1): B1
// CHECK:       [B1]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B1.2], A)
// CHECK-NEXT:    2: A a;
// CHECK-NEXT:    3:  (CXXConstructExpr, [B1.4], A)
// CHECK-NEXT:    4: A c;
// CHECK-NEXT:    5:  (CXXConstructExpr, [B1.6], A)
// CHECK-NEXT:    6: A d;
// CHECK-NEXT:    7: [B1.6].~A() (Implicit destructor)
// CHECK-NEXT:    8: [B1.6] (Lifetime ends)
// CHECK-NEXT:    9: [B1.4].~A() (Implicit destructor)
// CHECK-NEXT:   10: [B1.4] (Lifetime ends)
// CHECK-NEXT:   11:  (CXXConstructExpr, [B1.12], A)
// CHECK-NEXT:   12: A b;
// CHECK-NEXT:   13: [B1.12].~A() (Implicit destructor)
// CHECK-NEXT:   14: [B1.12] (Lifetime ends)
// CHECK-NEXT:   15: [B1.2].~A() (Implicit destructor)
// CHECK-NEXT:   16: [B1.2] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B0
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void test_scope() {
  A a;
  {
    A c;
    A d;
  }
  A b;
}

// CHECK:      [B4 (ENTRY)]
// CHECK-NEXT:   Succs (1): B3
// CHECK:       [B1]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B1.2], A)
// CHECK-NEXT:    2: A c;
// CHECK-NEXT:    3: [B1.2].~A() (Implicit destructor)
// CHECK-NEXT:    4: [B1.2] (Lifetime ends)
// CHECK-NEXT:    5: [B3.4].~A() (Implicit destructor)
// CHECK-NEXT:    6: [B3.4] (Lifetime ends)
// CHECK-NEXT:    7: [B3.2].~A() (Implicit destructor)
// CHECK-NEXT:    8: [B3.2] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B3
// CHECK-NEXT:    Succs (1): B0
// CHECK:       [B2]
// CHECK-NEXT:    1: return;
// CHECK-NEXT:    2: [B3.4].~A() (Implicit destructor)
// CHECK-NEXT:    3: [B3.4] (Lifetime ends)
// CHECK-NEXT:    4: [B3.2].~A() (Implicit destructor)
// CHECK-NEXT:    5: [B3.2] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B3
// CHECK-NEXT:    Succs (1): B0
// CHECK:       [B3]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B3.2], A)
// CHECK-NEXT:    2: A a;
// CHECK-NEXT:    3:  (CXXConstructExpr, [B3.4], A)
// CHECK-NEXT:    4: A b;
// CHECK-NEXT:    5: UV
// CHECK-NEXT:    6: [B3.5] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:    T: if [B3.6]
// CHECK-NEXT:    Preds (1): B4
// CHECK-NEXT:    Succs (2): B2 B1
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:   Preds (2): B1 B2
void test_return() {
  A a;
  A b;
  if (UV)
    return;
  A c;
}

// CHECK:       [B5 (ENTRY)]
// CHECK-NEXT:    Succs (1): B4
// CHECK:       [B1]
// CHECK-NEXT:    1: [B4.6].~A() (Implicit destructor)
// CHECK-NEXT:    2: [B4.6] (Lifetime ends)
// CHECK-NEXT:    3: [B4.2].~A() (Implicit destructor)
// CHECK-NEXT:    4: [B4.2] (Lifetime ends)
// CHECK-NEXT:    Preds (2): B2 B3
// CHECK-NEXT:    Succs (1): B0
// CHECK:       [B2]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B2.2], A)
// CHECK-NEXT:    2: A c;
// CHECK-NEXT:    3: [B2.2].~A() (Implicit destructor)
// CHECK-NEXT:    4: [B2.2] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B4
// CHECK-NEXT:    Succs (1): B1
// CHECK:       [B3]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B3.2], A)
// CHECK-NEXT:    2: A c;
// CHECK-NEXT:    3: [B3.2].~A() (Implicit destructor)
// CHECK-NEXT:    4: [B3.2] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B4
// CHECK-NEXT:    Succs (1): B1
// CHECK:       [B4]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B4.2], A)
// CHECK-NEXT:    2: A a;
// CHECK-NEXT:    3: a
// CHECK-NEXT:    4: [B4.3] (ImplicitCastExpr, NoOp, const A)
// CHECK-NEXT:    5: [B4.4] (CXXConstructExpr, [B4.6], A)
// CHECK-NEXT:    6: A b = a;
// CHECK-NEXT:    7: b
// CHECK-NEXT:    8: [B4.7] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:    9: [B4.8].operator int
// CHECK-NEXT:   10: [B4.8]
// CHECK-NEXT:   11: [B4.10] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK-NEXT:   12: [B4.11] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:    T: if [B4.12]
// CHECK-NEXT:    Preds (1): B5
// CHECK-NEXT:    Succs (2): B3 B2
// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
void test_if_implicit_scope() {
  A a;
  if (A b = a)
    A c;
  else
    A c;
}

// CHECK:       [B9 (ENTRY)]
// CHECK-NEXT:    Succs (1): B8
// CHECK:       [B1]
// CHECK-NEXT:    1: [B8.6].~A() (Implicit destructor)
// CHECK-NEXT:    2: [B8.6] (Lifetime ends)
// CHECK-NEXT:    3:  (CXXConstructExpr, [B1.4], A)
// CHECK-NEXT:    4: A e;
// CHECK-NEXT:    5: [B1.4].~A() (Implicit destructor)
// CHECK-NEXT:    6: [B1.4] (Lifetime ends)
// CHECK-NEXT:    7: [B8.2].~A() (Implicit destructor)
// CHECK-NEXT:    8: [B8.2] (Lifetime ends)
// CHECK-NEXT:    Preds (2): B2 B5
// CHECK-NEXT:    Succs (1): B0
// CHECK:       [B2]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B2.2], A)
// CHECK-NEXT:    2: A d;
// CHECK-NEXT:    3: [B2.2].~A() (Implicit destructor)
// CHECK-NEXT:    4: [B2.2] (Lifetime ends)
// CHECK-NEXT:    5: [B4.2].~A() (Implicit destructor)
// CHECK-NEXT:    6: [B4.2] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B4
// CHECK-NEXT:    Succs (1): B1
// CHECK:       [B3]
// CHECK-NEXT:    1: return;
// CHECK-NEXT:    2: [B4.2].~A() (Implicit destructor)
// CHECK-NEXT:    3: [B4.2] (Lifetime ends)
// CHECK-NEXT:    4: [B8.6].~A() (Implicit destructor)
// CHECK-NEXT:    5: [B8.6] (Lifetime ends)
// CHECK-NEXT:    6: [B8.2].~A() (Implicit destructor)
// CHECK-NEXT:    7: [B8.2] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B4
// CHECK-NEXT:    Succs (1): B0
// CHECK:       [B4]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B4.2], A)
// CHECK-NEXT:    2: A c;
// CHECK-NEXT:    3: UV
// CHECK-NEXT:    4: [B4.3] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:    T: if [B4.4]
// CHECK-NEXT:    Preds (1): B8
// CHECK-NEXT:    Succs (2): B3 B2
// CHECK:       [B5]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B5.2], A)
// CHECK-NEXT:    2: A d;
// CHECK-NEXT:    3: [B5.2].~A() (Implicit destructor)
// CHECK-NEXT:    4: [B5.2] (Lifetime ends)
// CHECK-NEXT:    5: [B7.2].~A() (Implicit destructor)
// CHECK-NEXT:    6: [B7.2] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B7
// CHECK-NEXT:    Succs (1): B1
// CHECK:       [B6]
// CHECK-NEXT:    1: return;
// CHECK-NEXT:    2: [B7.2].~A() (Implicit destructor)
// CHECK-NEXT:    3: [B7.2] (Lifetime ends)
// CHECK-NEXT:    4: [B8.6].~A() (Implicit destructor)
// CHECK-NEXT:    5: [B8.6] (Lifetime ends)
// CHECK-NEXT:    6: [B8.2].~A() (Implicit destructor)
// CHECK-NEXT:    7: [B8.2] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B7
// CHECK-NEXT:    Succs (1): B0
// CHECK:       [B7]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B7.2], A)
// CHECK-NEXT:    2: A c;
// CHECK-NEXT:    3: UV
// CHECK-NEXT:    4: [B7.3] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:    T: if [B7.4]
// CHECK-NEXT:    Preds (1): B8
// CHECK-NEXT:    Succs (2): B6 B5
// CHECK:       [B8]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B8.2], A)
// CHECK-NEXT:    2: A a;
// CHECK-NEXT:    3: a
// CHECK-NEXT:    4: [B8.3] (ImplicitCastExpr, NoOp, const A)
// CHECK-NEXT:    5: [B8.4] (CXXConstructExpr, [B8.6], A)
// CHECK-NEXT:    6: A b = a;
// CHECK-NEXT:    7: b
// CHECK-NEXT:    8: [B8.7] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:    9: [B8.8].operator int
// CHECK-NEXT:   10: [B8.8]
// CHECK-NEXT:   11: [B8.10] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK-NEXT:   12: [B8.11] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:    T: if [B8.12]
// CHECK-NEXT:    Preds (1): B9
// CHECK-NEXT:    Succs (2): B7 B4
// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:    Preds (3): B1 B3 B6
void test_if_jumps() {
  A a;
  if (A b = a) {
    A c;
    if (UV)
      return;
    A d;
  } else {
    A c;
    if (UV)
      return;
    A d;
  }
  A e;
}

// CHECK:       [B6 (ENTRY)]
// CHECK-NEXT:    Succs (1): B5
// CHECK:       [B1]
// CHECK-NEXT:    1: [B4.4].~A() (Implicit destructor)
// CHECK-NEXT:    2: [B4.4] (Lifetime ends)
// CHECK-NEXT:    3: [B5.2].~A() (Implicit destructor)
// CHECK-NEXT:    4: [B5.2] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B4
// CHECK-NEXT:    Succs (1): B0
// CHECK:       [B2]
// CHECK-NEXT:    Preds (1): B3
// CHECK-NEXT:    Succs (1): B4
// CHECK:       [B3]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B3.2], A)
// CHECK-NEXT:    2: A c;
// CHECK-NEXT:    3: [B3.2].~A() (Implicit destructor)
// CHECK-NEXT:    4: [B3.2] (Lifetime ends)
// CHECK-NEXT:    5: [B4.4].~A() (Implicit destructor)
// CHECK-NEXT:    6: [B4.4] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B4
// CHECK-NEXT:    Succs (1): B2
// CHECK:       [B4]
// CHECK-NEXT:    1: a
// CHECK-NEXT:    2: [B4.1] (ImplicitCastExpr, NoOp, const A)
// CHECK-NEXT:    3: [B4.2] (CXXConstructExpr, [B4.4], A)
// CHECK-NEXT:    4: A b = a;
// CHECK-NEXT:    5: b
// CHECK-NEXT:    6: [B4.5] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:    7: [B4.6].operator int
// CHECK-NEXT:    8: [B4.6]
// CHECK-NEXT:    9: [B4.8] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK-NEXT:   10: [B4.9] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:    T: while [B4.10]
// CHECK-NEXT:    Preds (2): B2 B5
// CHECK-NEXT:    Succs (2): B3 B1
// CHECK:       [B5]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B5.2], A)
// CHECK-NEXT:    2: A a;
// CHECK-NEXT:    Preds (1): B6
// CHECK-NEXT:    Succs (1): B4
// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
void test_while_implicit_scope() {
  A a;
  while (A b = a)
    A c;
}

// CHECK:       [B12 (ENTRY)]
// CHECK-NEXT:    Succs (1): B11
// CHECK:       [B1]
// CHECK-NEXT:    1: [B10.4].~A() (Implicit destructor)
// CHECK-NEXT:    2: [B10.4] (Lifetime ends)
// CHECK-NEXT:    3:  (CXXConstructExpr, [B1.4], A)
// CHECK-NEXT:    4: A e;
// CHECK-NEXT:    5: [B1.4].~A() (Implicit destructor)
// CHECK-NEXT:    6: [B1.4] (Lifetime ends)
// CHECK-NEXT:    7: [B11.2].~A() (Implicit destructor)
// CHECK-NEXT:    8: [B11.2] (Lifetime ends)
// CHECK-NEXT:    Preds (2): B8 B10
// CHECK-NEXT:    Succs (1): B0
// CHECK:       [B2]
// CHECK-NEXT:    Preds (2): B3 B6
// CHECK-NEXT:    Succs (1): B10
// CHECK:       [B3]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B3.2], A)
// CHECK-NEXT:    2: A d;
// CHECK-NEXT:    3: [B3.2].~A() (Implicit destructor)
// CHECK-NEXT:    4: [B3.2] (Lifetime ends)
// CHECK-NEXT:    5: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:    6: [B9.2] (Lifetime ends)
// CHECK-NEXT:    7: [B10.4].~A() (Implicit destructor)
// CHECK-NEXT:    8: [B10.4] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B5
// CHECK-NEXT:    Succs (1): B2
// CHECK:       [B4]
// CHECK-NEXT:    1: return;
// CHECK-NEXT:    2: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:    3: [B9.2] (Lifetime ends)
// CHECK-NEXT:    4: [B10.4].~A() (Implicit destructor)
// CHECK-NEXT:    5: [B10.4] (Lifetime ends)
// CHECK-NEXT:    6: [B11.2].~A() (Implicit destructor)
// CHECK-NEXT:    7: [B11.2] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B5
// CHECK-NEXT:    Succs (1): B0
// CHECK:       [B5]
// CHECK-NEXT:    1: UV
// CHECK-NEXT:    2: [B5.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:    T: if [B5.2]
// CHECK-NEXT:    Preds (1): B7
// CHECK-NEXT:    Succs (2): B4 B3
// CHECK:       [B6]
// CHECK-NEXT:    1: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:    2: [B9.2] (Lifetime ends)
// CHECK-NEXT:    3: [B10.4].~A() (Implicit destructor)
// CHECK-NEXT:    4: [B10.4] (Lifetime ends)
// CHECK-NEXT:    T: continue;
// CHECK-NEXT:    Preds (1): B7
// CHECK-NEXT:    Succs (1): B2
// CHECK:       [B7]
// CHECK-NEXT:    1: UV
// CHECK-NEXT:    2: [B7.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:    T: if [B7.2]
// CHECK-NEXT:    Preds (1): B9
// CHECK-NEXT:    Succs (2): B6 B5
// CHECK:       [B8]
// CHECK-NEXT:    1: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:    2: [B9.2] (Lifetime ends)
// CHECK-NEXT:    T: break;
// CHECK-NEXT:    Preds (1): B9
// CHECK-NEXT:    Succs (1): B1
// CHECK:       [B9]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B9.2], A)
// CHECK-NEXT:    2: A c;
// CHECK-NEXT:    3: UV
// CHECK-NEXT:    4: [B9.3] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:    T: if [B9.4]
// CHECK-NEXT:    Preds (1): B10
// CHECK-NEXT:    Succs (2): B8 B7
// CHECK:       [B10]
// CHECK-NEXT:    1: a
// CHECK-NEXT:    2: [B10.1] (ImplicitCastExpr, NoOp, const A)
// CHECK-NEXT:    3: [B10.2] (CXXConstructExpr, [B10.4], A)
// CHECK-NEXT:    4: A b = a;
// CHECK-NEXT:    5: b
// CHECK-NEXT:    6: [B10.5] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:    7: [B10.6].operator int
// CHECK-NEXT:    8: [B10.6]
// CHECK-NEXT:    9: [B10.8] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK-NEXT:   10: [B10.9] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:    T: while [B10.10]
// CHECK-NEXT:    Preds (2): B2 B11
// CHECK-NEXT:    Succs (2): B9 B1
// CHECK:       [B11]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B11.2], A)
// CHECK-NEXT:    2: A a;
// CHECK-NEXT:    Preds (1): B12
// CHECK-NEXT:    Succs (1): B10
// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:    Preds (2): B1 B4
void test_while_jumps() {
  A a;
  while (A b = a) {
    A c;
    if (UV)
      break;
    if (UV)
      continue;
    if (UV)
      return;
    A d;
  }
  A e;
}

// CHECK:       [B12 (ENTRY)]
// CHECK-NEXT:    Succs (1): B11
// CHECK:       [B1]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B1.2], A)
// CHECK-NEXT:    2: A d;
// CHECK-NEXT:    3: [B1.2].~A() (Implicit destructor)
// CHECK-NEXT:    4: [B1.2] (Lifetime ends)
// CHECK-NEXT:    5: [B11.2].~A() (Implicit destructor)
// CHECK-NEXT:    6: [B11.2] (Lifetime ends)
// CHECK-NEXT:    Preds (2): B8 B2
// CHECK-NEXT:    Succs (1): B0
// CHECK:       [B2]
// CHECK-NEXT:    1: UV
// CHECK-NEXT:    2: [B2.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:    T: do ... while [B2.2]
// CHECK-NEXT:    Preds (2): B3 B6
// CHECK-NEXT:    Succs (2): B10 B1
// CHECK:       [B3]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B3.2], A)
// CHECK-NEXT:    2: A c;
// CHECK-NEXT:    3: [B3.2].~A() (Implicit destructor)
// CHECK-NEXT:    4: [B3.2] (Lifetime ends)
// CHECK-NEXT:    5: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:    6: [B9.2] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B5
// CHECK-NEXT:    Succs (1): B2
// CHECK:       [B4]
// CHECK-NEXT:    1: return;
// CHECK-NEXT:    2: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:    3: [B9.2] (Lifetime ends)
// CHECK-NEXT:    4: [B11.2].~A() (Implicit destructor)
// CHECK-NEXT:    5: [B11.2] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B5
// CHECK-NEXT:    Succs (1): B0
// CHECK:       [B5]
// CHECK-NEXT:    1: UV
// CHECK-NEXT:    2: [B5.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:    T: if [B5.2]
// CHECK-NEXT:    Preds (1): B7
// CHECK-NEXT:    Succs (2): B4 B3
// CHECK:       [B6]
// CHECK-NEXT:    1: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:    2: [B9.2] (Lifetime ends)
// CHECK-NEXT:    T: continue;
// CHECK-NEXT:    Preds (1): B7
// CHECK-NEXT:    Succs (1): B2
// CHECK:       [B7]
// CHECK-NEXT:    1: UV
// CHECK-NEXT:    2: [B7.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:    T: if [B7.2]
// CHECK-NEXT:    Preds (1): B9
// CHECK-NEXT:    Succs (2): B6 B5
// CHECK:       [B8]
// CHECK-NEXT:    1: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:    2: [B9.2] (Lifetime ends)
// CHECK-NEXT:    T: break;
// CHECK-NEXT:    Preds (1): B9
// CHECK-NEXT:    Succs (1): B1
// CHECK:       [B9]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B9.2], A)
// CHECK-NEXT:    2: A b;
// CHECK-NEXT:    3: UV
// CHECK-NEXT:    4: [B9.3] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:    T: if [B9.4]
// CHECK-NEXT:    Preds (2): B10 B11
// CHECK-NEXT:    Succs (2): B8 B7
// CHECK:       [B10]
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B9
// CHECK:       [B11]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B11.2], A)
// CHECK-NEXT:    2: A a;
// CHECK-NEXT:    Preds (1): B12
// CHECK-NEXT:    Succs (1): B9
// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:    Preds (2): B1 B4
void test_do_jumps() {
  A a;
  do {
    A b;
    if (UV)
      break;
    if (UV)
      continue;
    if (UV)
      return;
    A c;
  } while (UV);
  A d;
}

// CHECK:       [B6 (ENTRY)]
// CHECK-NEXT:    Succs (1): B5
// CHECK:       [B1]
// CHECK-NEXT:    1: [B4.4].~A() (Implicit destructor)
// CHECK-NEXT:    2: [B4.4] (Lifetime ends)
// CHECK-NEXT:    3: [B5.2].~A() (Implicit destructor)
// CHECK-NEXT:    4: [B5.2] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B4
// CHECK-NEXT:    Succs (1): B0
// CHECK:       [B2]
// CHECK-NEXT:    1: b
// CHECK-NEXT:    2: [B2.1].p
// CHECK-NEXT:    3: [B4.4].~A() (Implicit destructor)
// CHECK-NEXT:    4: [B4.4] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B3
// CHECK-NEXT:    Succs (1): B4
// CHECK:       [B3]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B3.2], A)
// CHECK-NEXT:    2: A c;
// CHECK-NEXT:    3: [B3.2].~A() (Implicit destructor)
// CHECK-NEXT:    4: [B3.2] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B4
// CHECK-NEXT:    Succs (1): B2
// CHECK:       [B4]
// CHECK-NEXT:    1: a
// CHECK-NEXT:    2: [B4.1] (ImplicitCastExpr, NoOp, const A)
// CHECK-NEXT:    3: [B4.2] (CXXConstructExpr, [B4.4], A)
// CHECK-NEXT:    4: A b = a;
// CHECK-NEXT:    5: b
// CHECK-NEXT:    6: [B4.5] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:    7: [B4.6].operator int
// CHECK-NEXT:    8: [B4.6]
// CHECK-NEXT:    9: [B4.8] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK-NEXT:   10: [B4.9] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:    T: for (...; [B4.10]; ...)
// CHECK-NEXT:    Preds (2): B2 B5
// CHECK-NEXT:    Succs (2): B3 B1
// CHECK:       [B5]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B5.2], A)
// CHECK-NEXT:    2: A a;
// CHECK-NEXT:    Preds (1): B6
// CHECK-NEXT:    Succs (1): B4
// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
void test_for_implicit_scope() {
  for (A a; A b = a; b.p)
    A c;
}

// CHECK:       [B12 (ENTRY)]
// CHECK-NEXT:    Succs (1): B11
// CHECK:       [B1]
// CHECK-NEXT:    1: [B10.4].~A() (Implicit destructor)
// CHECK-NEXT:    2: [B10.4] (Lifetime ends)
// CHECK-NEXT:    3: [B11.4].~A() (Implicit destructor)
// CHECK-NEXT:    4: [B11.4] (Lifetime ends)
// CHECK-NEXT:    5:  (CXXConstructExpr, [B1.6], A)
// CHECK-NEXT:    6: A f;
// CHECK-NEXT:    7: [B1.6].~A() (Implicit destructor)
// CHECK-NEXT:    8: [B1.6] (Lifetime ends)
// CHECK-NEXT:    9: [B11.2].~A() (Implicit destructor)
// CHECK-NEXT:   10: [B11.2] (Lifetime ends)
// CHECK-NEXT:    Preds (2): B8 B10
// CHECK-NEXT:    Succs (1): B0
// CHECK:       [B2]
// CHECK-NEXT:    1: [B10.4].~A() (Implicit destructor)
// CHECK-NEXT:    2: [B10.4] (Lifetime ends)
// CHECK-NEXT:    Preds (2): B3 B6
// CHECK-NEXT:    Succs (1): B10
// CHECK:       [B3]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B3.2], A)
// CHECK-NEXT:    2: A e;
// CHECK-NEXT:    3: [B3.2].~A() (Implicit destructor)
// CHECK-NEXT:    4: [B3.2] (Lifetime ends)
// CHECK-NEXT:    5: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:    6: [B9.2] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B5
// CHECK-NEXT:    Succs (1): B2
// CHECK:       [B4]
// CHECK-NEXT:    1: return;
// CHECK-NEXT:    2: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:    3: [B9.2] (Lifetime ends)
// CHECK-NEXT:    4: [B10.4].~A() (Implicit destructor)
// CHECK-NEXT:    5: [B10.4] (Lifetime ends)
// CHECK-NEXT:    6: [B11.4].~A() (Implicit destructor)
// CHECK-NEXT:    7: [B11.4] (Lifetime ends)
// CHECK-NEXT:    8: [B11.2].~A() (Implicit destructor)
// CHECK-NEXT:    9: [B11.2] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B5
// CHECK-NEXT:    Succs (1): B0
// CHECK:       [B5]
// CHECK-NEXT:    1: UV
// CHECK-NEXT:    2: [B5.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:    T: if [B5.2]
// CHECK-NEXT:    Preds (1): B7
// CHECK-NEXT:    Succs (2): B4 B3
// CHECK:       [B6]
// CHECK-NEXT:    1: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:    2: [B9.2] (Lifetime ends)
// CHECK-NEXT:    T: continue;
// CHECK-NEXT:    Preds (1): B7
// CHECK-NEXT:    Succs (1): B2
// CHECK:       [B7]
// CHECK-NEXT:    1: UV
// CHECK-NEXT:    2: [B7.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:    T: if [B7.2]
// CHECK-NEXT:    Preds (1): B9
// CHECK-NEXT:    Succs (2): B6 B5
// CHECK:       [B8]
// CHECK-NEXT:    1: [B9.2].~A() (Implicit destructor)
// CHECK-NEXT:    2: [B9.2] (Lifetime ends)
// CHECK-NEXT:    T: break;
// CHECK-NEXT:    Preds (1): B9
// CHECK-NEXT:    Succs (1): B1
// CHECK:       [B9]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B9.2], A)
// CHECK-NEXT:    2: A d;
// CHECK-NEXT:    3: UV
// CHECK-NEXT:    4: [B9.3] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:    T: if [B9.4]
// CHECK-NEXT:    Preds (1): B10
// CHECK-NEXT:    Succs (2): B8 B7
// CHECK:       [B10]
// CHECK-NEXT:    1: b
// CHECK-NEXT:    2: [B10.1] (ImplicitCastExpr, NoOp, const A)
// CHECK-NEXT:    3: [B10.2] (CXXConstructExpr, [B10.4], A)
// CHECK-NEXT:    4: A c = b;
// CHECK-NEXT:    5: c
// CHECK-NEXT:    6: [B10.5] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:    7: [B10.6].operator int
// CHECK-NEXT:    8: [B10.6]
// CHECK-NEXT:    9: [B10.8] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK-NEXT:   10: [B10.9] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:    T: for (...; [B10.10]; )
// CHECK-NEXT:    Preds (2): B2 B11
// CHECK-NEXT:    Succs (2): B9 B1
// CHECK:       [B11]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B11.2], A)
// CHECK-NEXT:    2: A a;
// CHECK-NEXT:    3:  (CXXConstructExpr, [B11.4], A)
// CHECK-NEXT:    4: A b;
// CHECK-NEXT:    Preds (1): B12
// CHECK-NEXT:    Succs (1): B10
// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:    Preds (2): B1 B4
void test_for_jumps() {
  A a;
  for (A b; A c = b; ) {
    A d;
    if (UV)
      break;
    if (UV)
      continue;
    if (UV)
      return;
    A e;
  }
  A f;
}

// CHECK:      [B9 (ENTRY)]
// CHECK-NEXT:   Succs (1): B8
// CHECK:      [B1]
// CHECK-NEXT:   1: [B7.4].~A() (Implicit destructor)
// CHECK-NEXT:   2: [B7.4] (Lifetime ends)
// CHECK-NEXT:   3: [B8.2].~A() (Implicit destructor)
// CHECK-NEXT:   4: [B8.2] (Lifetime ends)
// CHECK-NEXT:   Preds (1): B7
// CHECK-NEXT:   Succs (1): B0
// CHECK:      [B2]
// CHECK-NEXT:   1: [B5.4] ? [B3.3] : [B4.2]
// CHECK-NEXT:   2: [B7.4].~A() (Implicit destructor)
// CHECK-NEXT:   3: [B7.4] (Lifetime ends)
// CHECK-NEXT:   Preds (2): B3 B4
// CHECK-NEXT:   Succs (1): B7
// CHECK:      [B3]
// CHECK-NEXT:   1: b
// CHECK-NEXT:   2: [B3.1].p
// CHECK-NEXT:   3: [B3.2]++
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B4]
// CHECK-NEXT:   1: 0
// CHECK-NEXT:   2: [B4.1] (ImplicitCastExpr, NullToPointer, int *)
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (1): B2
// CHECK:      [B5]
// CHECK-NEXT:   1: b
// CHECK-NEXT:   2: [B5.1].p
// CHECK-NEXT:   3: [B5.2] (ImplicitCastExpr, LValueToRValue, int *)
// CHECK-NEXT:   4: [B5.3] (ImplicitCastExpr, PointerToBoolean, _Bool)
// CHECK-NEXT:   T: [B5.4] ? ... : ...
// CHECK-NEXT:   Preds (1): B6
// CHECK-NEXT:   Succs (2): B3 B4
// CHECK:      [B6]
// CHECK-NEXT:   1: 0
// CHECK-NEXT:   2: (void)[B6.1] (CStyleCastExpr, ToVoid, void)
// CHECK-NEXT:   Preds (1): B7
// CHECK-NEXT:   Succs (1): B5
// CHECK:      [B7]
// CHECK-NEXT:   1: a
// CHECK-NEXT:   2: [B7.1] (ImplicitCastExpr, NoOp, const A)
// CHECK-NEXT:   3: [B7.2] (CXXConstructExpr, [B7.4],  A)
// CHECK-NEXT:   4: A b = a;
// CHECK-NEXT:   5: b
// CHECK-NEXT:   6: [B7.5] (ImplicitCastExpr, NoOp, const class A)
// CHECK-NEXT:   7: [B7.6].operator int
// CHECK-NEXT:   8: [B7.6]
// CHECK-NEXT:   9: [B7.8] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK-NEXT:  10: [B7.9] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:   T: for (...; [B7.10]; ...)
// CHECK-NEXT:   Preds (2): B2 B8
// CHECK-NEXT:   Succs (2): B6 B1
// CHECK:      [B8]
// CHECK-NEXT:   1:  (CXXConstructExpr, [B8.2], A)
// CHECK-NEXT:   2: A a;
// CHECK-NEXT:   Preds (1): B9
// CHECK-NEXT:   Succs (1): B7
void test_for_inc_conditional() {
  for (A a; A b = a; b.p ? b.p++ : 0)
    (void)0;
}

// CHECK:       [B2 (ENTRY)]
// CHECK-NEXT:    Succs (1): B1
// CHECK:       [B1]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B1.2], A)
// CHECK-NEXT:    2: A a;
// CHECK-NEXT:    3: int n;
// CHECK-NEXT:    4: n
// CHECK-NEXT:    5: &[B1.4]
// CHECK-NEXT:    6: a
// CHECK-NEXT:    7: [B1.6].p
// CHECK-NEXT:    8: [B1.7] = [B1.5]
// CHECK-NEXT:    9: [B1.2].~A() (Implicit destructor)
// CHECK-NEXT:   10: [B1.2] (Lifetime ends)
// CHECK-NEXT:   11: [B1.3] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B0
// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
void test_trivial_vs_non_trivial_order() {
  A a;
  int n;
  a.p = &n;
}

// CHECK:       [B4 (ENTRY)]
// CHECK-NEXT:    Succs (1): B3
// CHECK:       [B1]
// CHECK-NEXT:   a:
// CHECK-NEXT:    1: 1
// CHECK-NEXT:    2: i
// CHECK-NEXT:    3: [B1.2] = [B1.1]
// CHECK-NEXT:    4: [B2.1] (Lifetime ends)
// CHECK-NEXT:    Preds (2): B2 B3
// CHECK-NEXT:    Succs (1): B0
// CHECK:       [B2]
// CHECK-NEXT:    1: int i;
// CHECK-NEXT:    Succs (1): B1
// CHECK:       [B3]
// CHECK-NEXT:    T: goto a;
// CHECK-NEXT:    Preds (1): B4
// CHECK-NEXT:    Succs (1): B1
// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
void goto_past_declaration() {
  goto a;
  int i;
a:
  i = 1;
}

// CHECK:       [B4 (ENTRY)]
// CHECK-NEXT:    Succs (1): B3
// CHECK:       [B1]
// CHECK-NEXT:   a:
// CHECK-NEXT:    1: 1
// CHECK-NEXT:    2: k
// CHECK-NEXT:    3: [B1.2] = [B1.1]
// CHECK-NEXT:    4: [B2.4] (Lifetime ends)
// CHECK-NEXT:    Preds (2): B2 B3
// CHECK-NEXT:    Succs (1): B0
// CHECK:       [B2]
// CHECK-NEXT:    1: int j;
// CHECK-NEXT:    2: [B2.1] (Lifetime ends)
// CHECK-NEXT:    3: [B3.1] (Lifetime ends)
// CHECK-NEXT:    4: int k;
// CHECK-NEXT:    Succs (1): B1
// CHECK:       [B3]
// CHECK-NEXT:    1: int i;
// CHECK-NEXT:    2: [B3.1] (Lifetime ends)
// CHECK-NEXT:    T: goto a;
// CHECK-NEXT:    Preds (1): B4
// CHECK-NEXT:    Succs (1): B1
// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
void goto_past_declaration2() {
  {
    int i;
    goto a;
    int j;
  }
  {
    int k;
  a:
    k = 1;
  }
}

struct B {
  ~B();
};

// CHECK:       [B5 (ENTRY)]
// CHECK-NEXT:    Succs (1): B3
// CHECK:       [B1]
// CHECK-NEXT:    1: i
// CHECK-NEXT:    2: [B1.1]++
// CHECK-NEXT:    3: [B2.4].~B() (Implicit destructor)
// CHECK-NEXT:    4: [B2.4] (Lifetime ends)
// CHECK-NEXT:    5: [B2.2].~B() (Implicit destructor)
// CHECK-NEXT:    6: [B2.2] (Lifetime ends)
// CHECK-NEXT:    7: [B3.1] (Lifetime ends)
// CHECK-NEXT:    Succs (1): B0
// CHECK:       [B2]
// CHECK-NEXT:   label:
// CHECK-NEXT:    1:  (CXXConstructExpr, [B2.2], B)
// CHECK-NEXT:    2: B b1;
// CHECK-NEXT:    3:  (CXXConstructExpr, [B2.4], B)
// CHECK-NEXT:    4: B b2;
// CHECK-NEXT:    Preds (2): B3 B4
// CHECK-NEXT:    Succs (1): B4
// CHECK:       [B3]
// CHECK-NEXT:    1: int i;
// CHECK-NEXT:    Preds (1): B5
// CHECK-NEXT:    Succs (1): B2
// CHECK:       [B4]
// CHECK-NEXT:    1: [B2.4].~B() (Implicit destructor)
// CHECK-NEXT:    2: [B2.4] (Lifetime ends)
// CHECK-NEXT:    3: [B2.2].~B() (Implicit destructor)
// CHECK-NEXT:    4: [B2.2] (Lifetime ends)
// CHECK-NEXT:    T: goto label;
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B2
// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
int backpatched_goto() {
  int i;
label:
  B b1;
  B b2;
  goto label;
  i++;
}
