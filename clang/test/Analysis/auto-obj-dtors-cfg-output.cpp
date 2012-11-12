// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -analyze -analyzer-checker=debug.DumpCFG %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s
// XPASS: *

class A {
public:
  A() {}
  ~A() {}
  operator int() const { return 1; }
};

extern const bool UV;

void test_const_ref() {
  A a;
  const A& b = a;
  const A& c = A();
}

void test_array() {
  A a[2];
  A b[0];
}

void test_scope() {
  A a;
  { A c;
    A d;
  }
  A b;
}

void test_return() {
  A a;
  A b;
  if (UV) return;
  A c;
}

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

void test_if_implicit_scope() {
  A a;
  if (A b = a)
    A c;
  else A c;
}

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

void test_while_implicit_scope() {
  A a;
  while (A b = a)
    A c;
}

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

void test_do_implicit_scope() {
  do A a;
  while (UV);
}

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

void test_switch_implicit_scope() {
  A a;
  switch (A b = a)
    A c;
}

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

void test_for_implicit_scope() {
  for (A a; A b = a; )
    A c;
}

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

void test_catch_const_ref() {
  try {
  } catch (const A& e) {
  }
}

void test_catch_copy() {
  try {
  } catch (A e) {
  }
}

// CHECK:  [B1 (ENTRY)]
// CHECK:    Succs (1): B0
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (1): B1
// CHECK:  [B1 (ENTRY)]
// CHECK:    Succs (1): B0
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (1): B1
// CHECK:  [B2 (ENTRY)]
// CHECK:    Succs (1): B1
// CHECK:  [B1]
// CHECK:    1: 1
// CHECK:    2: return [B1.1];
// CHECK:    Preds (1): B2
// CHECK:    Succs (1): B0
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (1): B1
// CHECK:  [B2 (ENTRY)]
// CHECK:    Succs (1): B1
// CHECK:  [B1]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A a;
// CHECK:    3: a
// CHECK:    4: [B1.3] (ImplicitCastExpr, NoOp, const class A)
// CHECK:    5: const A &b = a;
// CHECK:    6: A() (CXXConstructExpr, class A)
// CHECK:    7: [B1.6] (BindTemporary)
// CHECK:    8: [B1.7] (ImplicitCastExpr, NoOp, const class A)
// CHECK:    9: [B1.8]
// CHECK:   10: const A &c = A();
// CHECK:   11: [B1.10].~A() (Implicit destructor)
// CHECK:   12: [B1.2].~A() (Implicit destructor)
// CHECK:    Preds (1): B2
// CHECK:    Succs (1): B0
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (1): B1
// CHECK:  [B2 (ENTRY)]
// CHECK:    Succs (1): B1
// CHECK:  [B1]
// CHECK:    1:  (CXXConstructExpr, class A [2])
// CHECK:    2: A a[2];
// CHECK:    3:  (CXXConstructExpr, class A [0])
// CHECK:    4: A b[0];
// CHECK:    5: [B1.2].~A() (Implicit destructor)
// CHECK:    Preds (1): B2
// CHECK:    Succs (1): B0
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (1): B1
// CHECK:  [B2 (ENTRY)]
// CHECK:    Succs (1): B1
// CHECK:  [B1]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A a;
// CHECK:    3:  (CXXConstructExpr, class A)
// CHECK:    4: A c;
// CHECK:    5:  (CXXConstructExpr, class A)
// CHECK:    6: A d;
// CHECK:    7: [B1.6].~A() (Implicit destructor)
// CHECK:    8: [B1.4].~A() (Implicit destructor)
// CHECK:    9:  (CXXConstructExpr, class A)
// CHECK:   10: A b;
// CHECK:   11: [B1.10].~A() (Implicit destructor)
// CHECK:   12: [B1.2].~A() (Implicit destructor)
// CHECK:    Preds (1): B2
// CHECK:    Succs (1): B0
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (1): B1
// CHECK:  [B4 (ENTRY)]
// CHECK:    Succs (1): B3
// CHECK:  [B1]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A c;
// CHECK:    3: [B1.2].~A() (Implicit destructor)
// CHECK:    4: [B3.4].~A() (Implicit destructor)
// CHECK:    5: [B3.2].~A() (Implicit destructor)
// CHECK:    Preds (1): B3
// CHECK:    Succs (1): B0
// CHECK:  [B2]
// CHECK:    1: return;
// CHECK:    2: [B3.4].~A() (Implicit destructor)
// CHECK:    3: [B3.2].~A() (Implicit destructor)
// CHECK:    Preds (1): B3
// CHECK:    Succs (1): B0
// CHECK:  [B3]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A a;
// CHECK:    3:  (CXXConstructExpr, class A)
// CHECK:    4: A b;
// CHECK:    5: UV
// CHECK:    6: [B3.5] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:    T: if [B3.6]
// CHECK:    Preds (1): B4
// CHECK:    Succs (2): B2 B1
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (2): B1 B2
// CHECK:  [B8 (ENTRY)]
// CHECK:    Succs (1): B7
// CHECK:  [B1]
// CHECK:   l1:
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A c;
// CHECK:    3: [B1.2].~A() (Implicit destructor)
// CHECK:    4: [B6.2].~A() (Implicit destructor)
// CHECK:    5: [B7.2].~A() (Implicit destructor)
// CHECK:    Preds (2): B2 B3
// CHECK:    Succs (1): B0
// CHECK:  [B2]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A b;
// CHECK:    3: [B2.2].~A() (Implicit destructor)
// CHECK:    4: [B6.4].~A() (Implicit destructor)
// CHECK:    Preds (1): B4
// CHECK:    Succs (1): B1
// CHECK:  [B3]
// CHECK:    1: [B6.4].~A() (Implicit destructor)
// CHECK:    T: goto l1;
// CHECK:    Preds (1): B4
// CHECK:    Succs (1): B1
// CHECK:  [B4]
// CHECK:    1: UV
// CHECK:    2: [B4.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:    T: if [B4.2]
// CHECK:    Preds (1): B6
// CHECK:    Succs (2): B3 B2
// CHECK:  [B5]
// CHECK:    1: [B6.4].~A() (Implicit destructor)
// CHECK:    2: [B6.2].~A() (Implicit destructor)
// CHECK:    T: goto l0;
// CHECK:    Preds (1): B6
// CHECK:    Succs (1): B6
// CHECK:  [B6]
// CHECK:   l0:
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A b;
// CHECK:    3:  (CXXConstructExpr, class A)
// CHECK:    4: A a;
// CHECK:    5: UV
// CHECK:    6: [B6.5] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:    T: if [B6.6]
// CHECK:    Preds (2): B7 B5
// CHECK:    Succs (2): B5 B4
// CHECK:  [B7]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A a;
// CHECK:    Preds (1): B8
// CHECK:    Succs (1): B6
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (1): B1
// CHECK:  [B5 (ENTRY)]
// CHECK:    Succs (1): B4
// CHECK:  [B1]
// CHECK:    1: [B4.6].~A() (Implicit destructor)
// CHECK:    2: [B4.2].~A() (Implicit destructor)
// CHECK:    Preds (2): B2 B3
// CHECK:    Succs (1): B0
// CHECK:  [B2]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A c;
// CHECK:    3: [B2.2].~A() (Implicit destructor)
// CHECK:    Preds (1): B4
// CHECK:    Succs (1): B1
// CHECK:  [B3]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A c;
// CHECK:    3: [B3.2].~A() (Implicit destructor)
// CHECK:    Preds (1): B4
// CHECK:    Succs (1): B1
// CHECK:  [B4]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A a;
// CHECK:    3: a
// CHECK:    4: [B4.3] (ImplicitCastExpr, NoOp, const class A)
// CHECK:    5: [B4.4] (CXXConstructExpr, class A)
// CHECK:    6: A b = a;
// CHECK:    7: b
// CHECK:    8: [B4.7] (ImplicitCastExpr, NoOp, const class A)
// CHECK:    9: [B4.8].operator int
// CHECK:   10: [B4.9]()
// CHECK:   11: [B4.10] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:   12: [B4.11] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK:    T: if [B4.12]
// CHECK:    Preds (1): B5
// CHECK:    Succs (2): B3 B2
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (1): B1
// CHECK:  [B9 (ENTRY)]
// CHECK:    Succs (1): B8
// CHECK:  [B1]
// CHECK:    1: [B8.6].~A() (Implicit destructor)
// CHECK:    2:  (CXXConstructExpr, class A)
// CHECK:    3: A e;
// CHECK:    4: [B1.3].~A() (Implicit destructor)
// CHECK:    5: [B8.2].~A() (Implicit destructor)
// CHECK:    Preds (2): B2 B5
// CHECK:    Succs (1): B0
// CHECK:  [B2]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A d;
// CHECK:    3: [B2.2].~A() (Implicit destructor)
// CHECK:    4: [B4.2].~A() (Implicit destructor)
// CHECK:    Preds (1): B4
// CHECK:    Succs (1): B1
// CHECK:  [B3]
// CHECK:    1: return;
// CHECK:    2: [B4.2].~A() (Implicit destructor)
// CHECK:    3: [B8.6].~A() (Implicit destructor)
// CHECK:    4: [B8.2].~A() (Implicit destructor)
// CHECK:    Preds (1): B4
// CHECK:    Succs (1): B0
// CHECK:  [B4]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A c;
// CHECK:    3: UV
// CHECK:    4: [B4.3] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:    T: if [B4.4]
// CHECK:    Preds (1): B8
// CHECK:    Succs (2): B3 B2
// CHECK:  [B5]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A d;
// CHECK:    3: [B5.2].~A() (Implicit destructor)
// CHECK:    4: [B7.2].~A() (Implicit destructor)
// CHECK:    Preds (1): B7
// CHECK:    Succs (1): B1
// CHECK:  [B6]
// CHECK:    1: return;
// CHECK:    2: [B7.2].~A() (Implicit destructor)
// CHECK:    3: [B8.6].~A() (Implicit destructor)
// CHECK:    4: [B8.2].~A() (Implicit destructor)
// CHECK:    Preds (1): B7
// CHECK:    Succs (1): B0
// CHECK:  [B7]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A c;
// CHECK:    3: UV
// CHECK:    4: [B7.3] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:    T: if [B7.4]
// CHECK:    Preds (1): B8
// CHECK:    Succs (2): B6 B5
// CHECK:  [B8]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A a;
// CHECK:    3: a
// CHECK:    4: [B8.3] (ImplicitCastExpr, NoOp, const class A)
// CHECK:    5: [B8.4] (CXXConstructExpr, class A)
// CHECK:    6: A b = a;
// CHECK:    7: b
// CHECK:    8: [B8.7] (ImplicitCastExpr, NoOp, const class A)
// CHECK:    9: [B8.8].operator int
// CHECK:   10: [B8.9]()
// CHECK:   11: [B8.10] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:   12: [B8.11] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK:    T: if [B8.12]
// CHECK:    Preds (1): B9
// CHECK:    Succs (2): B7 B4
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (3): B1 B3 B6
// CHECK:  [B6 (ENTRY)]
// CHECK:    Succs (1): B5
// CHECK:  [B1]
// CHECK:    1: [B4.4].~A() (Implicit destructor)
// CHECK:    2: [B5.2].~A() (Implicit destructor)
// CHECK:    Preds (1): B4
// CHECK:    Succs (1): B0
// CHECK:  [B2]
// CHECK:    Preds (1): B3
// CHECK:    Succs (1): B4
// CHECK:  [B3]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A c;
// CHECK:    3: [B3.2].~A() (Implicit destructor)
// CHECK:    4: [B4.4].~A() (Implicit destructor)
// CHECK:    Preds (1): B4
// CHECK:    Succs (1): B2
// CHECK:  [B4]
// CHECK:    1: a
// CHECK:    2: [B4.1] (ImplicitCastExpr, NoOp, const class A)
// CHECK:    3: [B4.2] (CXXConstructExpr, class A)
// CHECK:    4: A b = a;
// CHECK:    5: b
// CHECK:    6: [B4.5] (ImplicitCastExpr, NoOp, const class A)
// CHECK:    7: [B4.6].operator int
// CHECK:    8: [B4.7]()
// CHECK:    9: [B4.8] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:   10: [B4.9] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK:    T: while [B4.10]
// CHECK:    Preds (2): B2 B5
// CHECK:    Succs (2): B3 B1
// CHECK:  [B5]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A a;
// CHECK:    Preds (1): B6
// CHECK:    Succs (1): B4
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (1): B1
// CHECK:  [B12 (ENTRY)]
// CHECK:    Succs (1): B11
// CHECK:  [B1]
// CHECK:    1: [B10.4].~A() (Implicit destructor)
// CHECK:    2:  (CXXConstructExpr, class A)
// CHECK:    3: A e;
// CHECK:    4: [B1.3].~A() (Implicit destructor)
// CHECK:    5: [B11.2].~A() (Implicit destructor)
// CHECK:    Preds (2): B8 B10
// CHECK:    Succs (1): B0
// CHECK:  [B2]
// CHECK:    Preds (2): B3 B6
// CHECK:    Succs (1): B10
// CHECK:  [B3]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A d;
// CHECK:    3: [B3.2].~A() (Implicit destructor)
// CHECK:    4: [B9.2].~A() (Implicit destructor)
// CHECK:    5: [B10.4].~A() (Implicit destructor)
// CHECK:    Preds (1): B5
// CHECK:    Succs (1): B2
// CHECK:  [B4]
// CHECK:    1: return;
// CHECK:    2: [B9.2].~A() (Implicit destructor)
// CHECK:    3: [B10.4].~A() (Implicit destructor)
// CHECK:    4: [B11.2].~A() (Implicit destructor)
// CHECK:    Preds (1): B5
// CHECK:    Succs (1): B0
// CHECK:  [B5]
// CHECK:    1: UV
// CHECK:    2: [B5.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:    T: if [B5.2]
// CHECK:    Preds (1): B7
// CHECK:    Succs (2): B4 B3
// CHECK:  [B6]
// CHECK:    1: [B9.2].~A() (Implicit destructor)
// CHECK:    2: [B10.4].~A() (Implicit destructor)
// CHECK:    T: continue;
// CHECK:    Preds (1): B7
// CHECK:    Succs (1): B2
// CHECK:  [B7]
// CHECK:    1: UV
// CHECK:    2: [B7.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:    T: if [B7.2]
// CHECK:    Preds (1): B9
// CHECK:    Succs (2): B6 B5
// CHECK:  [B8]
// CHECK:    1: [B9.2].~A() (Implicit destructor)
// CHECK:    T: break;
// CHECK:    Preds (1): B9
// CHECK:    Succs (1): B1
// CHECK:  [B9]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A c;
// CHECK:    3: UV
// CHECK:    4: [B9.3] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:    T: if [B9.4]
// CHECK:    Preds (1): B10
// CHECK:    Succs (2): B8 B7
// CHECK:  [B10]
// CHECK:    1: a
// CHECK:    2: [B10.1] (ImplicitCastExpr, NoOp, const class A)
// CHECK:    3: [B10.2] (CXXConstructExpr, class A)
// CHECK:    4: A b = a;
// CHECK:    5: b
// CHECK:    6: [B10.5] (ImplicitCastExpr, NoOp, const class A)
// CHECK:    7: [B10.6].operator int
// CHECK:    8: [B10.7]()
// CHECK:    9: [B10.8] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:   10: [B10.9] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK:    T: while [B10.10]
// CHECK:    Preds (2): B2 B11
// CHECK:    Succs (2): B9 B1
// CHECK:  [B11]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A a;
// CHECK:    Preds (1): B12
// CHECK:    Succs (1): B10
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (2): B1 B4
// CHECK:  [B4 (ENTRY)]
// CHECK:    Succs (1): B2
// CHECK:  [B1]
// CHECK:    1: UV
// CHECK:    2: [B1.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:    T: do ... while [B1.2]
// CHECK:    Preds (1): B2
// CHECK:    Succs (2): B3 B0
// CHECK:  [B2]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A a;
// CHECK:    3: [B2.2].~A() (Implicit destructor)
// CHECK:    Preds (2): B3 B4
// CHECK:    Succs (1): B1
// CHECK:  [B3]
// CHECK:    Preds (1): B1
// CHECK:    Succs (1): B2
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (1): B1
// CHECK:  [B12 (ENTRY)]
// CHECK:    Succs (1): B11
// CHECK:  [B1]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A d;
// CHECK:    3: [B1.2].~A() (Implicit destructor)
// CHECK:    4: [B11.2].~A() (Implicit destructor)
// CHECK:    Preds (2): B8 B2
// CHECK:    Succs (1): B0
// CHECK:  [B2]
// CHECK:    1: UV
// CHECK:    2: [B2.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:    T: do ... while [B2.2]
// CHECK:    Preds (2): B3 B6
// CHECK:    Succs (2): B10 B1
// CHECK:  [B3]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A c;
// CHECK:    3: [B3.2].~A() (Implicit destructor)
// CHECK:    4: [B9.2].~A() (Implicit destructor)
// CHECK:    Preds (1): B5
// CHECK:    Succs (1): B2
// CHECK:  [B4]
// CHECK:    1: return;
// CHECK:    2: [B9.2].~A() (Implicit destructor)
// CHECK:    3: [B11.2].~A() (Implicit destructor)
// CHECK:    Preds (1): B5
// CHECK:    Succs (1): B0
// CHECK:  [B5]
// CHECK:    1: UV
// CHECK:    2: [B5.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:    T: if [B5.2]
// CHECK:    Preds (1): B7
// CHECK:    Succs (2): B4 B3
// CHECK:  [B6]
// CHECK:    1: [B9.2].~A() (Implicit destructor)
// CHECK:    T: continue;
// CHECK:    Preds (1): B7
// CHECK:    Succs (1): B2
// CHECK:  [B7]
// CHECK:    1: UV
// CHECK:    2: [B7.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:    T: if [B7.2]
// CHECK:    Preds (1): B9
// CHECK:    Succs (2): B6 B5
// CHECK:  [B8]
// CHECK:    1: [B9.2].~A() (Implicit destructor)
// CHECK:    T: break;
// CHECK:    Preds (1): B9
// CHECK:    Succs (1): B1
// CHECK:  [B9]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A b;
// CHECK:    3: UV
// CHECK:    4: [B9.3] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:    T: if [B9.4]
// CHECK:    Preds (2): B10 B11
// CHECK:    Succs (2): B8 B7
// CHECK:  [B10]
// CHECK:    Preds (1): B2
// CHECK:    Succs (1): B9
// CHECK:  [B11]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A a;
// CHECK:    Preds (1): B12
// CHECK:    Succs (1): B9
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (2): B1 B4
// CHECK:  [B4 (ENTRY)]
// CHECK:    Succs (1): B2
// CHECK:  [B1]
// CHECK:    1: [B2.6].~A() (Implicit destructor)
// CHECK:    2: [B2.2].~A() (Implicit destructor)
// CHECK:    Preds (2): B3 B2
// CHECK:    Succs (1): B0
// CHECK:  [B2]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A a;
// CHECK:    3: a
// CHECK:    4: [B2.3] (ImplicitCastExpr, NoOp, const class A)
// CHECK:    5: [B2.4] (CXXConstructExpr, class A)
// CHECK:    6: A b = a;
// CHECK:    7: b
// CHECK:    8: [B2.7] (ImplicitCastExpr, NoOp, const class A)
// CHECK:    9: [B2.8].operator int
// CHECK:   10: [B2.9]()
// CHECK:   11: [B2.10] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:    T: switch [B2.11]
// CHECK:    Preds (1): B4
// CHECK:    Succs (1): B1
// CHECK:  [B3]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A c;
// CHECK:    3: [B3.2].~A() (Implicit destructor)
// CHECK:    Succs (1): B1
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (1): B1
// CHECK:  [B9 (ENTRY)]
// CHECK:    Succs (1): B2
// CHECK:  [B1]
// CHECK:    1: [B2.6].~A() (Implicit destructor)
// CHECK:    2:  (CXXConstructExpr, class A)
// CHECK:    3: A g;
// CHECK:    4: [B1.3].~A() (Implicit destructor)
// CHECK:    5: [B2.2].~A() (Implicit destructor)
// CHECK:    Preds (3): B3 B7 B2
// CHECK:    Succs (1): B0
// CHECK:  [B2]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A a;
// CHECK:    3: a
// CHECK:    4: [B2.3] (ImplicitCastExpr, NoOp, const class A)
// CHECK:    5: [B2.4] (CXXConstructExpr, class A)
// CHECK:    6: A b = a;
// CHECK:    7: b
// CHECK:    8: [B2.7] (ImplicitCastExpr, NoOp, const class A)
// CHECK:    9: [B2.8].operator int
// CHECK:   10: [B2.9]()
// CHECK:   11: [B2.10] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:    T: switch [B2.11]
// CHECK:    Preds (1): B9
// CHECK:    Succs (3): B3 B8
// CHECK:      B1
// CHECK:  [B3]
// CHECK:   case 1:
// CHECK:    T: break;
// CHECK:    Preds (2): B2 B4
// CHECK:    Succs (1): B1
// CHECK:  [B4]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A f;
// CHECK:    3: [B4.2].~A() (Implicit destructor)
// CHECK:    4: [B8.2].~A() (Implicit destructor)
// CHECK:    Preds (1): B6
// CHECK:    Succs (1): B3
// CHECK:  [B5]
// CHECK:    1: return;
// CHECK:    2: [B8.2].~A() (Implicit destructor)
// CHECK:    3: [B2.6].~A() (Implicit destructor)
// CHECK:    4: [B2.2].~A() (Implicit destructor)
// CHECK:    Preds (1): B6
// CHECK:    Succs (1): B0
// CHECK:  [B6]
// CHECK:    1: UV
// CHECK:    2: [B6.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:    T: if [B6.2]
// CHECK:    Preds (1): B8
// CHECK:    Succs (2): B5 B4
// CHECK:  [B7]
// CHECK:    1: [B8.2].~A() (Implicit destructor)
// CHECK:    T: break;
// CHECK:    Preds (1): B8
// CHECK:    Succs (1): B1
// CHECK:  [B8]
// CHECK:   case 0:
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A c;
// CHECK:    3: UV
// CHECK:    4: [B8.3] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:    T: if [B8.4]
// CHECK:    Preds (1): B2
// CHECK:    Succs (2): B7 B6
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (2): B1 B5
// CHECK:  [B6 (ENTRY)]
// CHECK:    Succs (1): B5
// CHECK:  [B1]
// CHECK:    1: [B4.4].~A() (Implicit destructor)
// CHECK:    2: [B5.2].~A() (Implicit destructor)
// CHECK:    Preds (1): B4
// CHECK:    Succs (1): B0
// CHECK:  [B2]
// CHECK:    Preds (1): B3
// CHECK:    Succs (1): B4
// CHECK:  [B3]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A c;
// CHECK:    3: [B3.2].~A() (Implicit destructor)
// CHECK:    4: [B4.4].~A() (Implicit destructor)
// CHECK:    Preds (1): B4
// CHECK:    Succs (1): B2
// CHECK:  [B4]
// CHECK:    1: a
// CHECK:    2: [B4.1] (ImplicitCastExpr, NoOp, const class A)
// CHECK:    3: [B4.2] (CXXConstructExpr, class A)
// CHECK:    4: A b = a;
// CHECK:    5: b
// CHECK:    6: [B4.5] (ImplicitCastExpr, NoOp, const class A)
// CHECK:    7: [B4.6].operator int
// CHECK:    8: [B4.7]()
// CHECK:    9: [B4.8] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:   10: [B4.9] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK:    T: for (...; [B4.10]; )
// CHECK:    Preds (2): B2 B5
// CHECK:    Succs (2): B3 B1
// CHECK:  [B5]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A a;
// CHECK:    Preds (1): B6
// CHECK:    Succs (1): B4
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (1): B1
// CHECK:  [B12 (ENTRY)]
// CHECK:    Succs (1): B11
// CHECK:  [B1]
// CHECK:    1: [B10.4].~A() (Implicit destructor)
// CHECK:    2: [B11.4].~A() (Implicit destructor)
// CHECK:    3:  (CXXConstructExpr, class A)
// CHECK:    4: A f;
// CHECK:    5: [B1.4].~A() (Implicit destructor)
// CHECK:    6: [B11.2].~A() (Implicit destructor)
// CHECK:    Preds (2): B8 B10
// CHECK:    Succs (1): B0
// CHECK:  [B2]
// CHECK:    Preds (2): B3 B6
// CHECK:    Succs (1): B10
// CHECK:  [B3]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A e;
// CHECK:    3: [B3.2].~A() (Implicit destructor)
// CHECK:    4: [B9.2].~A() (Implicit destructor)
// CHECK:    5: [B10.4].~A() (Implicit destructor)
// CHECK:    Preds (1): B5
// CHECK:    Succs (1): B2
// CHECK:  [B4]
// CHECK:    1: return;
// CHECK:    2: [B9.2].~A() (Implicit destructor)
// CHECK:    3: [B10.4].~A() (Implicit destructor)
// CHECK:    4: [B11.4].~A() (Implicit destructor)
// CHECK:    5: [B11.2].~A() (Implicit destructor)
// CHECK:    Preds (1): B5
// CHECK:    Succs (1): B0
// CHECK:  [B5]
// CHECK:    1: UV
// CHECK:    2: [B5.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:    T: if [B5.2]
// CHECK:    Preds (1): B7
// CHECK:    Succs (2): B4 B3
// CHECK:  [B6]
// CHECK:    1: [B9.2].~A() (Implicit destructor)
// CHECK:    T: continue;
// CHECK:    Preds (1): B7
// CHECK:    Succs (1): B2
// CHECK:  [B7]
// CHECK:    1: UV
// CHECK:    2: [B7.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:    T: if [B7.2]
// CHECK:    Preds (1): B9
// CHECK:    Succs (2): B6 B5
// CHECK:  [B8]
// CHECK:    1: [B9.2].~A() (Implicit destructor)
// CHECK:    T: break;
// CHECK:    Preds (1): B9
// CHECK:    Succs (1): B1
// CHECK:  [B9]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A d;
// CHECK:    3: UV
// CHECK:    4: [B9.3] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:    T: if [B9.4]
// CHECK:    Preds (1): B10
// CHECK:    Succs (2): B8 B7
// CHECK:  [B10]
// CHECK:    1: b
// CHECK:    2: [B10.1] (ImplicitCastExpr, NoOp, const class A)
// CHECK:    3: [B10.2] (CXXConstructExpr, class A)
// CHECK:    4: A c = b;
// CHECK:    5: c
// CHECK:    6: [B10.5] (ImplicitCastExpr, NoOp, const class A)
// CHECK:    7: [B10.6].operator int
// CHECK:    8: [B10.7]()
// CHECK:    9: [B10.8] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:   10: [B10.9] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK:    T: for (...; [B10.10]; )
// CHECK:    Preds (2): B2 B11
// CHECK:    Succs (2): B9 B1
// CHECK:  [B11]
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A a;
// CHECK:    3:  (CXXConstructExpr, class A)
// CHECK:    4: A b;
// CHECK:    Preds (1): B12
// CHECK:    Succs (1): B10
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (2): B1 B4
// CHECK:  [B3 (ENTRY)]
// CHECK:    Succs (1): B0
// CHECK:  [B1]
// CHECK:    T: try ...
// CHECK:    Succs (2): B2 B0
// CHECK:  [B2]
// CHECK:   catch (const A &e):
// CHECK:    1: catch (const A &e) {
// CHECK: }
// CHECK:    Preds (1): B1
// CHECK:    Succs (1): B0
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (3): B2 B1 B3
// CHECK:  [B3 (ENTRY)]
// CHECK:    Succs (1): B0
// CHECK:  [B1]
// CHECK:    T: try ...
// CHECK:    Succs (2): B2 B0
// CHECK:  [B2]
// CHECK:   catch (A e):
// CHECK:    1: catch (A e) {
// CHECK: }
// CHECK:    2: [B2.1].~A() (Implicit destructor)
// CHECK:    Preds (1): B1
// CHECK:    Succs (1): B0
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (3): B2 B1 B3

