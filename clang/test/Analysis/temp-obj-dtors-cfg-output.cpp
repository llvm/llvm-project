// RUN: rm -f %t
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -analyzer-config cfg-rich-constructors=false -std=c++98 %s > %t 2>&1
// RUN: FileCheck --input-file=%t -check-prefixes=CHECK,CXX98,WARNINGS,CXX98-WARNINGS %s
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -analyzer-config cfg-rich-constructors=false -std=c++11 %s > %t 2>&1
// RUN: FileCheck --input-file=%t -check-prefixes=CHECK,CXX11,WARNINGS,CXX11-WARNINGS %s
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -analyzer-config cfg-rich-constructors=true -std=c++98 %s > %t 2>&1
// RUN: FileCheck --input-file=%t -check-prefixes=CHECK,CXX98,ANALYZER,CXX98-ANALYZER %s
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -analyzer-config cfg-rich-constructors=true -std=c++11 %s > %t 2>&1
// RUN: FileCheck --input-file=%t -check-prefixes=CHECK,CXX11,ANALYZER,CXX11-ANALYZER %s

// This file tests how we construct two different flavors of the Clang CFG -
// the CFG used by the Sema analysis-based warnings and the CFG used by the
// static analyzer. The difference in the behavior is checked via FileCheck
// prefixes (WARNINGS and ANALYZER respectively). When introducing new analyzer
// flags, no new run lines should be added - just these flags would go to the
// respective line depending on where is it turned on and where is it turned
// off. Feel free to add tests that test only one of the CFG flavors if you're
// not sure how the other flavor is supposed to work in your case.

// Additionally, different C++ standards are checked.

class A {
public:
  A() {}
  ~A() {}

  static A make() { return A(); }

  operator bool() { return false; }
  operator int() { return 0; }
};

class B {
public:
  B() {}
  ~B() {}

  operator bool() { return true; }
  operator int() { return 1; }
  operator A() { return A(); }
};

void foo(int);
void foo(bool);
void foo(const A&);

void test_binary() {
  int a = int(A()) + int(B());
  foo(int(A()) + int(B()));
  int b;
}

void test_and() {
  bool a = A() && B();
  foo(A() && B());
  int b;
}

void test_or() {
  bool a = A() || B();
  foo(A() || B());
  int b;
}

void test_cond() {
  A a = B() ? A() : A(B());
  if (B()) { foo(0); } else { foo(0); }
  int b;
}

struct C {
  C():b_(true) {}
  ~C() {}

  operator bool() { return b_; }
  bool b_;
};

struct D {
  D():b_(true) {}

  operator bool() { return b_; }
  bool b_;
};

int test_cond_unnamed_custom_destructor() {
  if (C()) { return 1; } else { return 0; }
}

int test_cond_named_custom_destructor() {
  if (C c = C()) { return 1; } else { return 0; }
}

int test_cond_unnamed_auto_destructor() {
  if (D()) { return 1; } else { return 0; }
}

int test_cond_named_auto_destructor() {
  if (D d = D()) { return 1; } else { return 0; }
}

void test_cond_cref() {
  const A& a = B() ? A() : A(B());
  foo(B() ? A() : A(B()));
  int b;
}

void test_cond_implicit() {
  A a = A() ?: A();
  int b;
}

void test_cond_implicit_cref() {
  const A& a = A() ?: A();
  foo(A() ?: A());
  int b;
}

void test_copy_init() {
  A a = A();
  int b;
}

void test_cref_init() {
  const A& a = A();
  foo(A());
  int b;
}

void test_call_copy_init() {
  A a = A::make();
  int b;
}

void test_call_cref_init() {
  const A& a = A::make();
  foo(A::make());
  int b;
}

void test_assign() {
  int a;
  a = A();
  int b;
}

class TestCtorInits {
  int a;
  int b;
public:
  TestCtorInits();
};

TestCtorInits::TestCtorInits()
  : a(int(A()) + int(B()))
  , b() {}

class NoReturn {
public:
  ~NoReturn() __attribute__((noreturn));
  void f();
};

void test_noreturn1() {
  int a;
  NoReturn().f();
  int b;
}

void test_noreturn2() {
  int a;
  NoReturn(), 47;
  int b;
}

extern bool check(const NoReturn&);

// PR16664 and PR18159
int testConsistencyNestedSimple(bool value) {
  if (value) {
    if (!value || check(NoReturn())) {
      return 1;
    }
  }
  return 0;
}

// PR16664 and PR18159
int testConsistencyNestedComplex(bool value) {
  if (value) {
    if (!value || !value || check(NoReturn())) {
      return 1;
    }
  }
  return 0;
}

// PR16664 and PR18159
int testConsistencyNestedNormalReturn(bool value) {
  if (value) {
    if (!value || value || check(NoReturn())) {
      return 1;
    }
  }
  return 0;
}

namespace pass_references_through {
class C {
public:
  ~C() {}
};

const C &foo1();
C &&foo2();

// In these examples the foo() expression has record type, not reference type.
// Don't try to figure out how to perform construction of the record here.
const C &bar1() { return foo1(); } // no-crash
C &&bar2() { return foo2(); } // no-crash
const C &bar3(bool coin) {
  return coin ? foo1() : foo1(); // no-crash
}
} // end namespace pass_references_through

// CHECK:   [B1 (ENTRY)]
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B1 (ENTRY)]
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B2 (ENTRY)]
// CHECK:     Succs (1): B1
// CHECK:   [B1]
// WARNINGS:     1: A() (CXXConstructExpr, A)
// ANALYZER:     1: A() (CXXConstructExpr, [B1.2], [B1.4], [B1.5], A)
// CHECK:     2: [B1.1] (BindTemporary)
// CHECK:     3: [B1.2] (ImplicitCastExpr, NoOp, const A)
// CHECK:     4: [B1.3]
// WARNINGS:     5: [B1.4] (CXXConstructExpr, A)
// ANALYZER:     5: [B1.4] (CXXConstructExpr, [B1.7], A)
// CHECK:     6: ~A() (Temporary object destructor)
// CHECK:     7: return [B1.5];
// CHECK:     Preds (1): B2
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B2 (ENTRY)]
// CHECK:     Succs (1): B1
// CHECK:   [B1]
// CHECK:     1: false
// CHECK:     2: return [B1.1];
// CHECK:     Preds (1): B2
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B2 (ENTRY)]
// CHECK:     Succs (1): B1
// CHECK:   [B1]
// CHECK:     1: 0
// CHECK:     2: return [B1.1];
// CHECK:     Preds (1): B2
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B1 (ENTRY)]
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B1 (ENTRY)]
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B2 (ENTRY)]
// CHECK:     Succs (1): B1
// CHECK:   [B1]
// CHECK:     1: true
// CHECK:     2: return [B1.1];
// CHECK:     Preds (1): B2
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B2 (ENTRY)]
// CHECK:     Succs (1): B1
// CHECK:   [B1]
// CHECK:     1: 1
// CHECK:     2: return [B1.1];
// CHECK:     Preds (1): B2
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B2 (ENTRY)]
// CHECK:     Succs (1): B1
// CHECK:   [B1]
// WARNINGS:     1: A() (CXXConstructExpr, A)
// ANALYZER:     1: A() (CXXConstructExpr, [B1.2], [B1.4], [B1.5], A)
// CHECK:     2: [B1.1] (BindTemporary)
// CHECK:     3: [B1.2] (ImplicitCastExpr, NoOp, const A)
// CHECK:     4: [B1.3]
// WARNINGS:     5: [B1.4] (CXXConstructExpr, A)
// ANALYZER:     5: [B1.4] (CXXConstructExpr, [B1.7], A)
// CHECK:     6: ~A() (Temporary object destructor)
// CHECK:     7: return [B1.5];
// CHECK:     Preds (1): B2
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B2 (ENTRY)]
// CHECK:     Succs (1): B1
// CHECK:   [B1]
// WARNINGS:     1: A() (CXXConstructExpr, A)
// ANALYZER:     1: A() (CXXConstructExpr, [B1.2], [B1.3], A)
// CHECK:     2: [B1.1] (BindTemporary)
// CHECK:     3: [B1.2]
// CHECK:     4: [B1.3].operator int
// CHECK:     5: [B1.3]
// CHECK:     6: [B1.5] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:     7: int([B1.6]) (CXXFunctionalCastExpr, NoOp, int)
// WARNINGS:     8: B() (CXXConstructExpr, B)
// ANALYZER:     8: B() (CXXConstructExpr, [B1.9], [B1.10], B)
// CHECK:     9: [B1.8] (BindTemporary)
// CHECK:    10: [B1.9]
// CHECK:    11: [B1.10].operator int
// CHECK:    12: [B1.10]
// CHECK:    13: [B1.12] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:    14: int([B1.13]) (CXXFunctionalCastExpr, NoOp, int)
// CHECK:    15: [B1.7] + [B1.14]
// CHECK:    16: int a = int(A()) + int(B());
// CHECK:    17: ~B() (Temporary object destructor)
// CHECK:    18: ~A() (Temporary object destructor)
// CHECK:    19: foo
// CHECK:    20: [B1.19] (ImplicitCastExpr, FunctionToPointerDecay, void (*)(int))
// WARNINGS:    21: A() (CXXConstructExpr, A)
// ANALYZER:    21: A() (CXXConstructExpr, [B1.22], [B1.23], A)
// CHECK:    22: [B1.21] (BindTemporary)
// CHECK:    23: [B1.22]
// CHECK:    24: [B1.23].operator int
// CHECK:    25: [B1.23]
// CHECK:    26: [B1.25] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:    27: int([B1.26]) (CXXFunctionalCastExpr, NoOp, int)
// WARNINGS:    28: B() (CXXConstructExpr, B)
// ANALYZER:    28: B() (CXXConstructExpr, [B1.29], [B1.30], B)
// CHECK:    29: [B1.28] (BindTemporary)
// CHECK:    30: [B1.29]
// CHECK:    31: [B1.30].operator int
// CHECK:    32: [B1.30]
// CHECK:    33: [B1.32] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:    34: int([B1.33]) (CXXFunctionalCastExpr, NoOp, int)
// CHECK:    35: [B1.27] + [B1.34]
// CHECK:    36: [B1.20]([B1.35])
// CHECK:    37: ~B() (Temporary object destructor)
// CHECK:    38: ~A() (Temporary object destructor)
// CHECK:    39: int b;
// CHECK:     Preds (1): B2
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B10 (ENTRY)]
// CHECK:     Succs (1): B9
// CHECK:   [B1]
// CHECK:     1: ~A() (Temporary object destructor)
// CHECK:     2: int b;
// CHECK:     Preds (2): B2 B3
// CHECK:     Succs (1): B0
// CHECK:   [B2]
// CHECK:     1: ~B() (Temporary object destructor)
// CHECK:     Preds (1): B3
// CHECK:     Succs (1): B1
// CHECK:   [B3]
// CHECK:     1: [B5.9] && [B4.6]
// CHECK:     2: [B5.3]([B3.1])
// CHECK:     T: (Temp Dtor) [B4.2]
// CHECK:     Preds (2): B4 B5
// CHECK:     Succs (2): B2 B1
// CHECK:   [B4]
// WARNINGS:     1: B() (CXXConstructExpr, B)
// ANALYZER:     1: B() (CXXConstructExpr, [B4.2], [B4.3], B)
// CHECK:     2: [B4.1] (BindTemporary)
// CHECK:     3: [B4.2]
// CHECK:     4: [B4.3].operator bool
// CHECK:     5: [B4.3]
// CHECK:     6: [B4.5] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CHECK:     Preds (1): B5
// CHECK:     Succs (1): B3
// CHECK:   [B5]
// CHECK:     1: ~A() (Temporary object destructor)
// CHECK:     2: foo
// CHECK:     3: [B5.2] (ImplicitCastExpr, FunctionToPointerDecay, void (*)(_Bool))
// WARNINGS:     4: A() (CXXConstructExpr, A)
// ANALYZER:     4: A() (CXXConstructExpr, [B5.5], [B5.6], A)
// CHECK:     5: [B5.4] (BindTemporary)
// CHECK:     6: [B5.5]
// CHECK:     7: [B5.6].operator bool
// CHECK:     8: [B5.6]
// CHECK:     9: [B5.8] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CHECK:     T: [B5.9] && ...
// CHECK:     Preds (2): B6 B7
// CHECK:     Succs (2): B4 B3
// CHECK:   [B6]
// CHECK:     1: ~B() (Temporary object destructor)
// CHECK:     Preds (1): B7
// CHECK:     Succs (1): B5
// CHECK:   [B7]
// CHECK:     1: [B9.6] && [B8.6]
// CHECK:     2: bool a = A() && B();
// CHECK:     T: (Temp Dtor) [B8.2]
// CHECK:     Preds (2): B8 B9
// CHECK:     Succs (2): B6 B5
// CHECK:   [B8]
// WARNINGS:     1: B() (CXXConstructExpr, B)
// ANALYZER:     1: B() (CXXConstructExpr, [B8.2], [B8.3], B)
// CHECK:     2: [B8.1] (BindTemporary)
// CHECK:     3: [B8.2]
// CHECK:     4: [B8.3].operator bool
// CHECK:     5: [B8.3]
// CHECK:     6: [B8.5] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CHECK:     Preds (1): B9
// CHECK:     Succs (1): B7
// CHECK:   [B9]
// WARNINGS:     1: A() (CXXConstructExpr, A)
// ANALYZER:     1: A() (CXXConstructExpr, [B9.2], [B9.3], A)
// CHECK:     2: [B9.1] (BindTemporary)
// CHECK:     3: [B9.2]
// CHECK:     4: [B9.3].operator bool
// CHECK:     5: [B9.3]
// CHECK:     6: [B9.5] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CHECK:     T: [B9.6] && ...
// CHECK:     Preds (1): B10
// CHECK:     Succs (2): B8 B7
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B10 (ENTRY)]
// CHECK:     Succs (1): B9
// CHECK:   [B1]
// CHECK:     1: ~A() (Temporary object destructor)
// CHECK:     2: int b;
// CHECK:     Preds (2): B2 B3
// CHECK:     Succs (1): B0
// CHECK:   [B2]
// CHECK:     1: ~B() (Temporary object destructor)
// CHECK:     Preds (1): B3
// CHECK:     Succs (1): B1
// CHECK:   [B3]
// CHECK:     1: [B5.9] || [B4.6]
// CHECK:     2: [B5.3]([B3.1])
// CHECK:     T: (Temp Dtor) [B4.2]
// CHECK:     Preds (2): B4 B5
// CHECK:     Succs (2): B2 B1
// CHECK:   [B4]
// WARNINGS:     1: B() (CXXConstructExpr, B)
// ANALYZER:     1: B() (CXXConstructExpr, [B4.2], [B4.3], B)
// CHECK:     2: [B4.1] (BindTemporary)
// CHECK:     3: [B4.2]
// CHECK:     4: [B4.3].operator bool
// CHECK:     5: [B4.3]
// CHECK:     6: [B4.5] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CHECK:     Preds (1): B5
// CHECK:     Succs (1): B3
// CHECK:   [B5]
// CHECK:     1: ~A() (Temporary object destructor)
// CHECK:     2: foo
// CHECK:     3: [B5.2] (ImplicitCastExpr, FunctionToPointerDecay, void (*)(_Bool))
// WARNINGS:     4: A() (CXXConstructExpr, A)
// ANALYZER:     4: A() (CXXConstructExpr, [B5.5], [B5.6], A)
// CHECK:     5: [B5.4] (BindTemporary)
// CHECK:     6: [B5.5]
// CHECK:     7: [B5.6].operator bool
// CHECK:     8: [B5.6]
// CHECK:     9: [B5.8] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CHECK:     T: [B5.9] || ...
// CHECK:     Preds (2): B6 B7
// CHECK:     Succs (2): B3 B4
// CHECK:   [B6]
// CHECK:     1: ~B() (Temporary object destructor)
// CHECK:     Preds (1): B7
// CHECK:     Succs (1): B5
// CHECK:   [B7]
// CHECK:     1: [B9.6] || [B8.6]
// CHECK:     2: bool a = A() || B();
// CHECK:     T: (Temp Dtor) [B8.2]
// CHECK:     Preds (2): B8 B9
// CHECK:     Succs (2): B6 B5
// CHECK:   [B8]
// WARNINGS:     1: B() (CXXConstructExpr, B)
// ANALYZER:     1: B() (CXXConstructExpr, [B8.2], [B8.3], B)
// CHECK:     2: [B8.1] (BindTemporary)
// CHECK:     3: [B8.2]
// CHECK:     4: [B8.3].operator bool
// CHECK:     5: [B8.3]
// CHECK:     6: [B8.5] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CHECK:     Preds (1): B9
// CHECK:     Succs (1): B7
// CHECK:   [B9]
// WARNINGS:     1: A() (CXXConstructExpr, A)
// ANALYZER:     1: A() (CXXConstructExpr, [B9.2], [B9.3], A)
// CHECK:     2: [B9.1] (BindTemporary)
// CHECK:     3: [B9.2]
// CHECK:     4: [B9.3].operator bool
// CHECK:     5: [B9.3]
// CHECK:     6: [B9.5] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CHECK:     T: [B9.6] || ...
// CHECK:     Preds (1): B10
// CHECK:     Succs (2): B7 B8
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B11 (ENTRY)]
// CHECK:     Succs (1): B10
// CHECK:   [B1]
// CHECK:     1: int b;
// CHECK:     2: [B7.5].~A() (Implicit destructor)
// CHECK:     Preds (2): B2 B3
// CHECK:     Succs (1): B0
// CHECK:   [B2]
// CHECK:     1: foo
// CHECK:     2: [B2.1] (ImplicitCastExpr, FunctionToPointerDecay, void (*)(int))
// CHECK:     3: 0
// CHECK:     4: [B2.2]([B2.3])
// CHECK:     Preds (1): B4
// CHECK:     Succs (1): B1
// CHECK:   [B3]
// CHECK:     1: foo
// CHECK:     2: [B3.1] (ImplicitCastExpr, FunctionToPointerDecay, void (*)(int))
// CHECK:     3: 0
// CHECK:     4: [B3.2]([B3.3])
// CHECK:     Preds (1): B4
// CHECK:     Succs (1): B1
// CHECK:   [B4]
// CHECK:     1: ~B() (Temporary object destructor)
// WARNINGS:     2: B() (CXXConstructExpr, B)
// ANALYZER:     2: B() (CXXConstructExpr, [B4.3], [B4.4], B)
// CHECK:     3: [B4.2] (BindTemporary)
// CHECK:     4: [B4.3]
// CHECK:     5: [B4.4].operator bool
// CHECK:     6: [B4.4]
// CHECK:     7: [B4.6] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CHECK:     8: ~B() (Temporary object destructor)
// CHECK:     T: if [B4.7]
// CHECK:     Preds (2): B5 B6
// CHECK:     Succs (2): B3 B2
// CHECK:   [B5]
// CHECK:     1: ~A() (Temporary object destructor)
// CHECK:     2: ~A() (Temporary object destructor)
// CHECK:     Preds (1): B7
// CHECK:     Succs (1): B4
// CHECK:   [B6]
// CHECK:     1: ~A() (Temporary object destructor)
// CHECK:     2: ~A() (Temporary object destructor)
// CHECK:     3: ~A() (Temporary object destructor)
// CHECK:     4: ~B() (Temporary object destructor)
// CHECK:     Preds (1): B7
// CHECK:     Succs (1): B4
// CHECK:   [B7]
// CHECK:     1: [B10.6] ? [B8.6] : [B9.16]
// CHECK:     2: [B7.1] (ImplicitCastExpr, NoOp, const A)
// CHECK:     3: [B7.2]
// WARNINGS:     4: [B7.3] (CXXConstructExpr, A)
// ANALYZER:     4: [B7.3] (CXXConstructExpr, [B7.5], A)
// CHECK:     5: A a = B() ? A() : A(B());
// CHECK:     T: (Temp Dtor) [B9.2]
// CHECK:     Preds (2): B8 B9
// CHECK:     Succs (2): B6 B5
// CHECK:   [B8]
// WARNINGS:     1: A() (CXXConstructExpr, A)
// ANALYZER:     1: A() (CXXConstructExpr, [B8.2], [B8.4], [B8.5], A)
// CHECK:     2: [B8.1] (BindTemporary)
// CHECK:     3: [B8.2] (ImplicitCastExpr, NoOp, const A)
// CHECK:     4: [B8.3]
// WARNINGS:     5: [B8.4] (CXXConstructExpr, A)
// ANALYZER:     5: [B8.4] (CXXConstructExpr, [B8.6], [B7.3], [B7.4], A)
// CHECK:     6: [B8.5] (BindTemporary)
// CHECK:     Preds (1): B10
// CHECK:     Succs (1): B7
// CHECK:   [B9]
// WARNINGS:     1: B() (CXXConstructExpr, B)
// ANALYZER:     1: B() (CXXConstructExpr, [B9.2], [B9.3], B)
// CHECK:     2: [B9.1] (BindTemporary)
// CHECK:     3: [B9.2]
// CHECK:     4: [B9.3].operator A
// CHECK:     5: [B9.3]
// CHECK:     6: [B9.5] (ImplicitCastExpr, UserDefinedConversion, A)
// CHECK:     7: [B9.6] (BindTemporary)
// CHECK:     8: [B9.7] (ImplicitCastExpr, NoOp, const A)
// CHECK:     9: [B9.8]
// WARNINGS:     10: [B9.9] (CXXConstructExpr, A)
// ANALYZER:     10: [B9.9] (CXXConstructExpr, [B9.11], [B9.14], [B9.15], A)
// CHECK:    11: [B9.10] (BindTemporary)
// CHECK:    12: A([B9.11]) (CXXFunctionalCastExpr, ConstructorConversion, A)
// CHECK:    13: [B9.12] (ImplicitCastExpr, NoOp, const A)
// CHECK:    14: [B9.13]
// WARNINGS:    15: [B9.14] (CXXConstructExpr, A)
// ANALYZER:    15: [B9.14] (CXXConstructExpr, [B9.16], [B7.3], [B7.4], A)
// CHECK:    16: [B9.15] (BindTemporary)
// CHECK:     Preds (1): B10
// CHECK:     Succs (1): B7
// CHECK:   [B10]
// WARNINGS:     1: B() (CXXConstructExpr, B)
// ANALYZER:     1: B() (CXXConstructExpr, [B10.2], [B10.3], B)
// CHECK:     2: [B10.1] (BindTemporary)
// CHECK:     3: [B10.2]
// CHECK:     4: [B10.3].operator bool
// CHECK:     5: [B10.3]
// CHECK:     6: [B10.5] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CHECK:     T: [B10.6] ? ... : ...
// CHECK:     Preds (1): B11
// CHECK:     Succs (2): B8 B9
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B2 (ENTRY)]
// CHECK:     Succs (1): B1
// CHECK:   [B1]
// CHECK:     1: true
// CHECK:     2: b_([B1.1]) (Member initializer)
// CHECK:     Preds (1): B2
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B1 (ENTRY)]
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B2 (ENTRY)]
// CHECK:     Succs (1): B1
// CHECK:   [B1]
// CHECK:     1: this
// CHECK:     2: [B1.1]->b_
// CHECK:     3: [B1.2] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:     4: return [B1.3];
// CHECK:     Preds (1): B2
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B2 (ENTRY)]
// CHECK:     Succs (1): B1
// CHECK:   [B1]
// CHECK:     1: true
// CHECK:     2: b_([B1.1]) (Member initializer)
// CHECK:     Preds (1): B2
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B2 (ENTRY)]
// CHECK:     Succs (1): B1
// CHECK:   [B1]
// CHECK:     1: this
// CHECK:     2: [B1.1]->b_
// CHECK:     3: [B1.2] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:     4: return [B1.3];
// CHECK:     Preds (1): B2
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B4 (ENTRY)]
// CHECK:     Succs (1): B3
// CHECK:   [B1]
// CHECK:     1: 0
// CHECK:     2: return [B1.1];
// CHECK:     Preds (1): B3
// CHECK:     Succs (1): B0
// CHECK:   [B2]
// CHECK:     1: 1
// CHECK:     2: return [B2.1];
// CHECK:     Preds (1): B3
// CHECK:     Succs (1): B0
// CHECK:   [B3]
// WARNINGS:     1: C() (CXXConstructExpr, C)
// ANALYZER:     1: C() (CXXConstructExpr, [B3.2], [B3.3], C)
// CHECK:     2: [B3.1] (BindTemporary)
// CHECK:     3: [B3.2]
// CHECK:     4: [B3.3].operator bool
// CHECK:     5: [B3.3]
// CHECK:     6: [B3.5] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CHECK:     7: ~C() (Temporary object destructor)
// CHECK:     T: if [B3.6]
// CHECK:     Preds (1): B4
// CHECK:     Succs (2): B2 B1
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (2): B1 B2
// CHECK:   [B5 (ENTRY)]
// CHECK:     Succs (1): B4
// CHECK:   [B1]
// CHECK:     1: [B4.6].~C() (Implicit destructor)
// CHECK:     Succs (1): B0
// CHECK:   [B2]
// CHECK:     1: 0
// CHECK:     2: return [B2.1];
// CHECK:     3: [B4.6].~C() (Implicit destructor)
// CHECK:     Preds (1): B4
// CHECK:     Succs (1): B0
// CHECK:   [B3]
// CHECK:     1: 1
// CHECK:     2: return [B3.1];
// CHECK:     3: [B4.6].~C() (Implicit destructor)
// CHECK:     Preds (1): B4
// CHECK:     Succs (1): B0
// CHECK:   [B4]
// WARNINGS:     1: C() (CXXConstructExpr, C)
// ANALYZER:     1: C() (CXXConstructExpr, [B4.2], [B4.4], [B4.5], C)
// CHECK:     2: [B4.1] (BindTemporary)
// CHECK:     3: [B4.2] (ImplicitCastExpr, NoOp, const C)
// CHECK:     4: [B4.3]
// WARNINGS:     5: [B4.4] (CXXConstructExpr, C)
// ANALYZER:     5: [B4.4] (CXXConstructExpr, [B4.6], C)
// CHECK:     6: C c = C();
// CHECK:     7: ~C() (Temporary object destructor)
// CHECK:     8: c
// CHECK:     9: [B4.8].operator bool
// CHECK:    10: [B4.8]
// CHECK:    11: [B4.10] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CHECK:     T: if [B4.11]
// CHECK:     Preds (1): B5
// CHECK:     Succs (2): B3 B2
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (3): B1 B2 B3
// CHECK:   [B4 (ENTRY)]
// CHECK:     Succs (1): B3
// CHECK:   [B1]
// CHECK:     1: 0
// CHECK:     2: return [B1.1];
// CHECK:     Preds (1): B3
// CHECK:     Succs (1): B0
// CHECK:   [B2]
// CHECK:     1: 1
// CHECK:     2: return [B2.1];
// CHECK:     Preds (1): B3
// CHECK:     Succs (1): B0
// CHECK:   [B3]
// WARNINGS:  1: D() (CXXConstructExpr, D)
// ANALYZER:  1: D() (CXXConstructExpr, [B3.2], D)
// CHECK:     2: [B3.1]
// CHECK:     3: [B3.2].operator bool
// CHECK:     4: [B3.2]
// CHECK:     5: [B3.4] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CHECK:     T: if [B3.5]
// CHECK:     Preds (1): B4
// CHECK:     Succs (2): B2 B1
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (2): B1 B2
// CHECK:   [B4 (ENTRY)]
// CHECK:     Succs (1): B3
// CHECK:   [B1]
// CHECK:     1: 0
// CHECK:     2: return [B1.1];
// CHECK:     Preds (1): B3
// CHECK:     Succs (1): B0
// CHECK:   [B2]
// CHECK:     1: 1
// CHECK:     2: return [B2.1];
// CHECK:     Preds (1): B3
// CHECK:     Succs (1): B0
// CHECK:   [B3]
// CXX98-WARNINGS:     1: D() (CXXConstructExpr, D)
// CXX98-ANALYZER:     1: D() (CXXConstructExpr, [B3.3], [B3.4], D)
// CXX98:     2: [B3.1] (ImplicitCastExpr, NoOp, const D)
// CXX98:     3: [B3.2]
// CXX98-WARNINGS:     4: [B3.3] (CXXConstructExpr, D)
// CXX98-ANALYZER:     4: [B3.3] (CXXConstructExpr, [B3.5], D)
// CXX98:     5: D d = D();
// CXX98:     6: d
// CXX98:     7: [B3.6].operator bool
// CXX98:     8: [B3.6]
// CXX98:     9: [B3.8] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CXX98:     T: if [B3.9]
// CXX11-WARNINGS:     1: D() (CXXConstructExpr, D)
// CXX11-ANALYZER:     1: D() (CXXConstructExpr, [B3.2], [B3.3], D)
// CXX11:     2: [B3.1]
// CXX11-WARNINGS:     3: [B3.2] (CXXConstructExpr, D)
// CXX11-ANALYZER:     3: [B3.2] (CXXConstructExpr, [B3.4], D)
// CXX11:     4: D d = D();
// CXX11:     5: d
// CXX11:     6: [B3.5].operator bool
// CXX11:     7: [B3.5]
// CXX11:     8: [B3.7] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CXX11:     T: if [B3.8]
// CHECK:     Preds (1): B4
// CHECK:     Succs (2): B2 B1
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (2): B1 B2
// CHECK:   [B14 (ENTRY)]
// CHECK:     Succs (1): B13
// CHECK:   [B1]
// CHECK:     1: ~B() (Temporary object destructor)
// CHECK:     2: int b;
// CHECK:     3: [B10.4].~A() (Implicit destructor)
// CHECK:     Preds (2): B2 B3
// CHECK:     Succs (1): B0
// CHECK:   [B2]
// CHECK:     1: ~A() (Temporary object destructor)
// CHECK:     2: ~A() (Temporary object destructor)
// CHECK:     Preds (1): B4
// CHECK:     Succs (1): B1
// CHECK:   [B3]
// CHECK:     1: ~A() (Temporary object destructor)
// CHECK:     2: ~A() (Temporary object destructor)
// CHECK:     3: ~A() (Temporary object destructor)
// CHECK:     4: ~B() (Temporary object destructor)
// CHECK:     Preds (1): B4
// CHECK:     Succs (1): B1
// CHECK:   [B4]
// CHECK:     1: [B7.9] ? [B5.6] : [B6.16]
// CHECK:     2: [B4.1] (ImplicitCastExpr, NoOp, const A)
// CHECK:     3: [B4.2]
// CHECK:     4: [B7.3]([B4.3])
// CHECK:     T: (Temp Dtor) [B6.2]
// CHECK:     Preds (2): B5 B6
// CHECK:     Succs (2): B3 B2
// CHECK:   [B5]
// WARNINGS:     1: A() (CXXConstructExpr, A)
// ANALYZER:     1: A() (CXXConstructExpr, [B5.2], [B5.4], [B5.5], A)
// CHECK:     2: [B5.1] (BindTemporary)
// CHECK:     3: [B5.2] (ImplicitCastExpr, NoOp, const A)
// CHECK:     4: [B5.3]
// WARNINGS:     5: [B5.4] (CXXConstructExpr, A)
// ANALYZER:     5: [B5.4] (CXXConstructExpr, [B5.6], [B4.3], A)
// CHECK:     6: [B5.5] (BindTemporary)
// CHECK:     Preds (1): B7
// CHECK:     Succs (1): B4
// CHECK:   [B6]
// WARNINGS:     1: B() (CXXConstructExpr, B)
// ANALYZER:     1: B() (CXXConstructExpr, [B6.2], [B6.3], B)
// CHECK:     2: [B6.1] (BindTemporary)
// CHECK:     3: [B6.2]
// CHECK:     4: [B6.3].operator A
// CHECK:     5: [B6.3]
// CHECK:     6: [B6.5] (ImplicitCastExpr, UserDefinedConversion, A)
// CHECK:     7: [B6.6] (BindTemporary)
// CHECK:     8: [B6.7] (ImplicitCastExpr, NoOp, const A)
// CHECK:     9: [B6.8]
// WARNINGS:     10: [B6.9] (CXXConstructExpr, A)
// ANALYZER:     10: [B6.9] (CXXConstructExpr, [B6.11], [B6.14], [B6.15], A)
// CHECK:    11: [B6.10] (BindTemporary)
// CHECK:    12: A([B6.11]) (CXXFunctionalCastExpr, ConstructorConversion, A)
// CHECK:    13: [B6.12] (ImplicitCastExpr, NoOp, const A)
// CHECK:    14: [B6.13]
// WARNINGS:    15: [B6.14] (CXXConstructExpr, A)
// ANALYZER:    15: [B6.14] (CXXConstructExpr, [B6.16], [B4.3], A)
// CHECK:    16: [B6.15] (BindTemporary)
// CHECK:     Preds (1): B7
// CHECK:     Succs (1): B4
// CHECK:   [B7]
// CHECK:     1: ~B() (Temporary object destructor)
// CHECK:     2: foo
// CHECK:     3: [B7.2] (ImplicitCastExpr, FunctionToPointerDecay, void (*)(const A &))
// WARNINGS:     4: B() (CXXConstructExpr, B)
// ANALYZER:     4: B() (CXXConstructExpr, [B7.5], [B7.6], B)
// CHECK:     5: [B7.4] (BindTemporary)
// CHECK:     6: [B7.5]
// CHECK:     7: [B7.6].operator bool
// CHECK:     8: [B7.6]
// CHECK:     9: [B7.8] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CHECK:     T: [B7.9] ? ... : ...
// CHECK:     Preds (2): B8 B9
// CHECK:     Succs (2): B5 B6
// CHECK:   [B8]
// CHECK:     1: ~A() (Temporary object destructor)
// CHECK:     Preds (1): B10
// CHECK:     Succs (1): B7
// CHECK:   [B9]
// CHECK:     1: ~A() (Temporary object destructor)
// CHECK:     2: ~A() (Temporary object destructor)
// CHECK:     3: ~B() (Temporary object destructor)
// CHECK:     Preds (1): B10
// CHECK:     Succs (1): B7
// CHECK:   [B10]
// CHECK:     1: [B13.6] ? [B11.6] : [B12.16]
// CHECK:     2: [B10.1] (ImplicitCastExpr, NoOp, const A)
// CHECK:     3: [B10.2]
// CHECK:     4: const A &a = B() ? A() : A(B());
// CHECK:     T: (Temp Dtor) [B12.2]
// CHECK:     Preds (2): B11 B12
// CHECK:     Succs (2): B9 B8
// CHECK:   [B11]
// WARNINGS:     1: A() (CXXConstructExpr, A)
// ANALYZER:     1: A() (CXXConstructExpr, [B11.2], [B11.4], [B11.5], A)
// CHECK:     2: [B11.1] (BindTemporary)
// CHECK:     3: [B11.2] (ImplicitCastExpr, NoOp, const A)
// CHECK:     4: [B11.3]
// WARNINGS:     5: [B11.4] (CXXConstructExpr, A)
// ANALYZER:     5: [B11.4] (CXXConstructExpr, [B10.3], A)
// CHECK:     6: [B11.5] (BindTemporary)
// CHECK:     Preds (1): B13
// CHECK:     Succs (1): B10
// CHECK:   [B12]
// WARNINGS:     1: B() (CXXConstructExpr, B)
// ANALYZER:     1: B() (CXXConstructExpr, [B12.2], [B12.3], B)
// CHECK:     2: [B12.1] (BindTemporary)
// CHECK:     3: [B12.2]
// CHECK:     4: [B12.3].operator A
// CHECK:     5: [B12.3]
// CHECK:     6: [B12.5] (ImplicitCastExpr, UserDefinedConversion, A)
// CHECK:     7: [B12.6] (BindTemporary)
// CHECK:     8: [B12.7] (ImplicitCastExpr, NoOp, const A)
// CHECK:     9: [B12.8]
// WARNINGS:     10: [B12.9] (CXXConstructExpr, A)
// ANALYZER:     10: [B12.9] (CXXConstructExpr, [B12.11], [B12.14], [B12.15], A)
// CHECK:    11: [B12.10] (BindTemporary)
// CHECK:    12: A([B12.11]) (CXXFunctionalCastExpr, ConstructorConversion, A)
// CHECK:    13: [B12.12] (ImplicitCastExpr, NoOp, const A)
// CHECK:    14: [B12.13]
// WARNINGS:    15: [B12.14] (CXXConstructExpr, A)
// ANALYZER:    15: [B12.14] (CXXConstructExpr, [B10.3], A)
// CHECK:    16: [B12.15] (BindTemporary)
// CHECK:     Preds (1): B13
// CHECK:     Succs (1): B10
// CHECK:   [B13]
// WARNINGS:     1: B() (CXXConstructExpr, B)
// ANALYZER:     1: B() (CXXConstructExpr, [B13.2], [B13.3], B)
// CHECK:     2: [B13.1] (BindTemporary)
// CHECK:     3: [B13.2]
// CHECK:     4: [B13.3].operator bool
// CHECK:     5: [B13.3]
// CHECK:     6: [B13.5] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CHECK:     T: [B13.6] ? ... : ...
// CHECK:     Preds (1): B14
// CHECK:     Succs (2): B11 B12
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B8 (ENTRY)]
// CHECK:     Succs (1): B7
// CHECK:   [B1]
// CHECK:     1: int b;
// CHECK:     2: [B4.5].~A() (Implicit destructor)
// CHECK:     Preds (2): B2 B3
// CHECK:     Succs (1): B0
// CHECK:   [B2]
// CHECK:     1: ~A() (Temporary object destructor)
// CHECK:     Preds (1): B4
// CHECK:     Succs (1): B1
// CHECK:   [B3]
// CHECK:     1: ~A() (Temporary object destructor)
// CHECK:     2: ~A() (Temporary object destructor)
// CHECK:     Preds (1): B4
// CHECK:     Succs (1): B1
// CHECK:   [B4]
// CXX98:     1: [B7.2] ?: [B6.6]
// CXX11:     1: [B7.3] ?: [B6.6]
// CHECK:     2: [B4.1] (ImplicitCastExpr, NoOp, const A)
// CHECK:     3: [B4.2]
// WARNINGS:     4: [B4.3] (CXXConstructExpr, A)
// ANALYZER:     4: [B4.3] (CXXConstructExpr, [B4.5], A)
// CHECK:     5: A a = A() ?: A();
// CHECK:     T: (Temp Dtor) [B6.2]
// CHECK:     Preds (2): B5 B6
// CHECK:     Succs (2): B3 B2
// CHECK:   [B5]
// CXX98:     1: [B7.2] (ImplicitCastExpr, NoOp, const A)
// CXX98:     2: [B5.1]
// WARNINGS-CXX98:     3: [B5.2] (CXXConstructExpr, A)
// ANALYZER-CXX98:     3: [B5.2] (CXXConstructExpr, [B5.4], A)
// CXX98:     4: [B5.3] (BindTemporary)
// CXX11:     1: [B7.3] (ImplicitCastExpr, NoOp, const A)
// WARNINGS-CXX11:     2: [B5.1] (CXXConstructExpr, A)
// ANALYZER-CXX11:     2: [B5.1] (CXXConstructExpr, [B5.3], A)
// CXX11:     3: [B5.2] (BindTemporary)
// CHECK:     Preds (1): B7
// CHECK:     Succs (1): B4
// CHECK:   [B6]
// WARNINGS:     1: A() (CXXConstructExpr, A)
// ANALYZER:     1: A() (CXXConstructExpr, [B6.2], [B6.4], [B6.5], A)
// CHECK:     2: [B6.1] (BindTemporary)
// CHECK:     3: [B6.2] (ImplicitCastExpr, NoOp, const A)
// CHECK:     4: [B6.3]
// WARNINGS:     5: [B6.4] (CXXConstructExpr, A)
// ANALYZER:     5: [B6.4] (CXXConstructExpr, [B6.6], A)
// CHECK:     6: [B6.5] (BindTemporary)
// CHECK:     Preds (1): B7
// CHECK:     Succs (1): B4
// CHECK:   [B7]
// WARNINGS:     1: A() (CXXConstructExpr, A)
// ANALYZER-CXX98:     1: A() (CXXConstructExpr, [B7.2], [B7.3], A)
// ANALYZER-CXX11:     1: A() (CXXConstructExpr, [B7.2], A)
// CHECK:     2: [B7.1] (BindTemporary)
// CHECK:     3: [B7.2]
// CHECK:     4: [B7.3].operator bool
// CHECK:     5: [B7.3]
// CHECK:     6: [B7.5] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CHECK:     T: [B7.6] ? ... : ...
// CHECK:     Preds (1): B8
// CHECK:     Succs (2): B5 B6
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B13 (ENTRY)]
// CHECK:     Succs (1): B12
// CHECK:   [B1]
// CHECK:     1: int b;
// CHECK:     2: [B9.4].~A() (Implicit destructor)
// CHECK:     Preds (2): B2 B3
// CHECK:     Succs (1): B0
// CHECK:   [B2]
// CHECK:     1: ~A() (Temporary object destructor)
// CHECK:     Preds (1): B4
// CHECK:     Succs (1): B1
// CHECK:   [B3]
// CHECK:     1: ~A() (Temporary object destructor)
// CHECK:     2: ~A() (Temporary object destructor)
// CHECK:     Preds (1): B4
// CHECK:     Succs (1): B1
// CHECK:   [B4]
// CXX98:     1: [B7.4] ?: [B6.6]
// CXX11:     1: [B7.5] ?: [B6.6]
// CHECK:     2: [B4.1] (ImplicitCastExpr, NoOp, const A)
// CHECK:     3: [B4.2]
// CHECK:     4: [B7.2]([B4.3])
// CHECK:     T: (Temp Dtor) [B6.2]
// CHECK:     Preds (2): B5 B6
// CHECK:     Succs (2): B3 B2
// CHECK:   [B5]
// CXX98:     1: [B7.4] (ImplicitCastExpr, NoOp, const A)
// CXX98:     2: [B5.1]
// WARNINGS-CXX98:     3: [B5.2] (CXXConstructExpr, A)
// ANALYZER-CXX98:     3: [B5.2] (CXXConstructExpr, [B5.4], A)
// CXX98:     4: [B5.3] (BindTemporary)
// CXX11:     1: [B7.5] (ImplicitCastExpr, NoOp, const A)
// WARNINGS-CXX11:     2: [B5.1] (CXXConstructExpr, A)
// ANALYZER-CXX11:     2: [B5.1] (CXXConstructExpr, [B5.3], A)
// CXX11:     3: [B5.2] (BindTemporary)
// CHECK:     Preds (1): B7
// CHECK:     Succs (1): B4
// CHECK:   [B6]
// WARNINGS:     1: A() (CXXConstructExpr, A)
// ANALYZER:     1: A() (CXXConstructExpr, [B6.2], [B6.4], [B6.5], A)
// CHECK:     2: [B6.1] (BindTemporary)
// CHECK:     3: [B6.2] (ImplicitCastExpr, NoOp, const A)
// CHECK:     4: [B6.3]
// WARNINGS:     5: [B6.4] (CXXConstructExpr, A)
// ANALYZER:     5: [B6.4] (CXXConstructExpr, [B6.6], A)
// CHECK:     6: [B6.5] (BindTemporary)
// CHECK:     Preds (1): B7
// CHECK:     Succs (1): B4
// CHECK:   [B7]
// CHECK:     1: foo
// CHECK:     2: [B7.1] (ImplicitCastExpr, FunctionToPointerDecay, void (*)(const A &))
// WARNINGS:     3: A() (CXXConstructExpr, A)
// ANALYZER-CXX98:     3: A() (CXXConstructExpr, [B7.4], A)
// ANALYZER-CXX11:     3: A() (CXXConstructExpr, A)
// CHECK:     4: [B7.3] (BindTemporary)
// CHECK:     5: [B7.4]
// CHECK:     6: [B7.5].operator bool
// CHECK:     7: [B7.5]
// CHECK:     8: [B7.7] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CHECK:     T: [B7.8] ? ... : ...
// CHECK:     Preds (2): B8 B9
// CHECK:     Succs (2): B5 B6
// CHECK:   [B8]
// CHECK:     1: ~A() (Temporary object destructor)
// CHECK:     Preds (1): B9
// CHECK:     Succs (1): B7
// CHECK:   [B9]
// CXX98:     1: [B12.2] ?: [B11.6]
// CXX11:     1: [B12.3] ?: [B11.6]
// CHECK:     2: [B9.1] (ImplicitCastExpr, NoOp, const A)
// CHECK:     3: [B9.2]
// CHECK:     4: const A &a = A() ?: A();
// CHECK:     T: (Temp Dtor) [B11.2]
// CHECK:     Preds (2): B10 B11
// CHECK:     Succs (2): B8 B7
// CHECK:   [B10]
// CXX98:     1: [B12.2] (ImplicitCastExpr, NoOp, const A)
// CXX98:     2: [B10.1]
// WARNINGS-CXX98:     3: [B10.2] (CXXConstructExpr, A)
// ANALYZER-CXX98:     3: [B10.2] (CXXConstructExpr, [B10.4], A)
// CXX98:     4: [B10.3] (BindTemporary)
// CXX11:     1: [B12.3] (ImplicitCastExpr, NoOp, const A)
// WARNINGS-CXX11:     2: [B10.1] (CXXConstructExpr, A)
// ANALYZER-CXX11:     2: [B10.1] (CXXConstructExpr, [B10.3], A)
// CXX11:     3: [B10.2] (BindTemporary)
// CHECK:     Preds (1): B12
// CHECK:     Succs (1): B9
// CHECK:   [B11]
// WARNINGS-CHECK:     1: A() (CXXConstructExpr, A)
// ANALYZER-CHECK:     1: A() (CXXConstructExpr, [B11.2], A)
// CHECK:     2: [B11.1] (BindTemporary)
// CHECK:     3: [B11.2] (ImplicitCastExpr, NoOp, const A)
// CHECK:     4: [B11.3]
// WARNINGS:     5: [B11.4] (CXXConstructExpr, A)
// ANALYZER:     5: [B11.4] (CXXConstructExpr, [B11.6], A)
// CHECK:     6: [B11.5] (BindTemporary)
// CHECK:     Preds (1): B12
// CHECK:     Succs (1): B9
// CHECK:   [B12]
// WARNINGS:     1: A() (CXXConstructExpr, A)
// ANALYZER-CXX98:     1: A() (CXXConstructExpr, [B12.2], [B12.3], A)
// ANALYZER-CXX11:     1: A() (CXXConstructExpr, [B12.2], A)
// CHECK:     2: [B12.1] (BindTemporary)
// CHECK:     3: [B12.2]
// CHECK:     4: [B12.3].operator bool
// CHECK:     5: [B12.3]
// CHECK:     6: [B12.5] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CHECK:     T: [B12.6] ? ... : ...
// CHECK:     Preds (1): B13
// CHECK:     Succs (2): B10 B11
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B2 (ENTRY)]
// CHECK:     Succs (1): B1
// CHECK:   [B1]
// WARNINGS:     1: A() (CXXConstructExpr, A)
// ANALYZER:     1: A() (CXXConstructExpr, [B1.2], [B1.4], [B1.5], A)
// CHECK:     2: [B1.1] (BindTemporary)
// CHECK:     3: [B1.2] (ImplicitCastExpr, NoOp, const A)
// CHECK:     4: [B1.3]
// WARNINGS:     5: [B1.4] (CXXConstructExpr, A)
// ANALYZER:     5: [B1.4] (CXXConstructExpr, [B1.6], A)
// CHECK:     6: A a = A();
// CHECK:     7: ~A() (Temporary object destructor)
// CHECK:     8: int b;
// CHECK:     9: [B1.6].~A() (Implicit destructor)
// CHECK:     Preds (1): B2
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B2 (ENTRY)]
// CHECK:     Succs (1): B1
// CHECK:   [B1]
// WARNINGS:     1: A() (CXXConstructExpr, A)
// ANALYZER:     1: A() (CXXConstructExpr, [B1.4], A)
// CHECK:     2: [B1.1] (BindTemporary)
// CHECK:     3: [B1.2] (ImplicitCastExpr, NoOp, const A)
// CHECK:     4: [B1.3]
// CHECK:     5: const A &a = A();
// CHECK:     6: foo
// CHECK:     7: [B1.6] (ImplicitCastExpr, FunctionToPointerDecay, void (*)(const A &))
// WARNINGS:     8: A() (CXXConstructExpr, A)
// ANALYZER:     8: A() (CXXConstructExpr, [B1.9], [B1.11], A)
// CHECK:     9: [B1.8] (BindTemporary)
// CHECK:    10: [B1.9] (ImplicitCastExpr, NoOp, const A)
// CHECK:    11: [B1.10]
// CHECK:    12: [B1.7]([B1.11])
// CHECK:    13: ~A() (Temporary object destructor)
// CHECK:    14: int b;
// CHECK:    15: [B1.5].~A() (Implicit destructor)
// CHECK:     Preds (1): B2
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B2 (ENTRY)]
// CHECK:     Succs (1): B1
// CHECK:   [B1]
// CHECK:     1: A::make
// CHECK:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, A (*)(void))
// WARNINGS:     3: [B1.2]()
// ANALYZER:     3: [B1.2]() (CXXRecordTypedCall, [B1.4], [B1.6], [B1.7])
// CHECK:     4: [B1.3] (BindTemporary)
// CHECK:     5: [B1.4] (ImplicitCastExpr, NoOp, const A)
// CHECK:     6: [B1.5]
// WARNINGS:     7: [B1.6] (CXXConstructExpr, A)
// ANALYZER:     7: [B1.6] (CXXConstructExpr, [B1.8], A)
// CHECK:     8: A a = A::make();
// CHECK:     9: ~A() (Temporary object destructor)
// CHECK:    10: int b;
// CHECK:    11: [B1.8].~A() (Implicit destructor)
// CHECK:     Preds (1): B2
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B2 (ENTRY)]
// CHECK:     Succs (1): B1
// CHECK:   [B1]
// CHECK:     1: A::make
// CHECK:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, A (*)(void))
// WARNINGS:     3: [B1.2]()
// ANALYZER:     3: [B1.2]() (CXXRecordTypedCall, [B1.6])
// CHECK:     4: [B1.3] (BindTemporary)
// CHECK:     5: [B1.4] (ImplicitCastExpr, NoOp, const A)
// CHECK:     6: [B1.5]
// CHECK:     7: const A &a = A::make();
// CHECK:     8: foo
// CHECK:     9: [B1.8] (ImplicitCastExpr, FunctionToPointerDecay, void (*)(const A &))
// CHECK:    10: A::make
// CHECK:    11: [B1.10] (ImplicitCastExpr, FunctionToPointerDecay, A (*)(void))
// WARNINGS:    12: [B1.11]()
// ANALYZER:    12: [B1.11]() (CXXRecordTypedCall, [B1.13], [B1.15])
// CHECK:    13: [B1.12] (BindTemporary)
// CHECK:    14: [B1.13] (ImplicitCastExpr, NoOp, const A)
// CHECK:    15: [B1.14]
// CHECK:    16: [B1.9]([B1.15])
// CHECK:    17: ~A() (Temporary object destructor)
// CHECK:    18: int b;
// CHECK:    19: [B1.7].~A() (Implicit destructor)
// CHECK:     Preds (1): B2
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B2 (ENTRY)]
// CHECK:     Succs (1): B1
// CHECK:   [B1]
// CHECK:     1: int a;
// WARNINGS:     2: A() (CXXConstructExpr, A)
// ANALYZER:     2: A() (CXXConstructExpr, [B1.3], [B1.4], A)
// CHECK:     3: [B1.2] (BindTemporary)
// CHECK:     4: [B1.3]
// CHECK:     5: [B1.4].operator int
// CHECK:     6: [B1.4]
// CHECK:     7: [B1.6] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:     8: a
// CHECK:     9: [B1.8] = [B1.7]
// CHECK:    10: ~A() (Temporary object destructor)
// CHECK:    11: int b;
// CHECK:     Preds (1): B2
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B2 (ENTRY)]
// CHECK:     Succs (1): B1
// CHECK:   [B1]
// WARNINGS:     1: A() (CXXConstructExpr, A)
// ANALYZER:     1: A() (CXXConstructExpr, [B1.2], [B1.3], A)
// CHECK:     2: [B1.1] (BindTemporary)
// CHECK:     3: [B1.2]
// CHECK:     4: [B1.3].operator int
// CHECK:     5: [B1.3]
// CHECK:     6: [B1.5] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:     7: int([B1.6]) (CXXFunctionalCastExpr, NoOp, int)
// WARNINGS:     8: B() (CXXConstructExpr, B)
// ANALYZER:     8: B() (CXXConstructExpr, [B1.9], [B1.10], B)
// CHECK:     9: [B1.8] (BindTemporary)
// CHECK:    10: [B1.9]
// CHECK:    11: [B1.10].operator int
// CHECK:    12: [B1.10]
// CHECK:    13: [B1.12] (ImplicitCastExpr, UserDefinedConversion, int)
// CHECK:    14: int([B1.13]) (CXXFunctionalCastExpr, NoOp, int)
// CHECK:    15: [B1.7] + [B1.14]
// CHECK:    16: a([B1.15]) (Member initializer)
// CHECK:    17: ~B() (Temporary object destructor)
// CHECK:    18: ~A() (Temporary object destructor)
// CHECK:    19: b(/*implicit*/(int)0) (Member initializer)
// CHECK:     Preds (1): B2
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B3 (ENTRY)]
// CHECK:     Succs (1): B2
// CHECK:   [B1]
// CHECK:     1: int b;
// CHECK:     Preds (1): B2(Unreachable)
// CHECK:     Succs (1): B0
// CHECK:   [B2 (NORETURN)]
// CHECK:     1: int a;
// WARNINGS:     2: NoReturn() (CXXConstructExpr, NoReturn)
// ANALYZER-CXX98:     2: NoReturn() (CXXConstructExpr, [B2.3], [B2.4], NoReturn)
// ANALYZER-CXX11:     2: NoReturn() (CXXConstructExpr, [B2.3], NoReturn)
// CHECK:     3: [B2.2] (BindTemporary)
// CHECK:     [[MEMBER:[45]]]: [B2.{{[34]}}].f
// CHECK:     {{[56]}}: [B2.[[MEMBER]]]()
// CHECK:     {{[67]}}: ~NoReturn() (Temporary object destructor)
// CHECK:     Preds (1): B3
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (2): B1 B2
// CHECK:   [B3 (ENTRY)]
// CHECK:     Succs (1): B2
// CHECK:   [B1]
// CHECK:     1: int b;
// CHECK:     Preds (1): B2(Unreachable)
// CHECK:     Succs (1): B0
// CHECK:   [B2 (NORETURN)]
// CHECK:     1: int a;
// WARNINGS:     2: NoReturn() (CXXConstructExpr, NoReturn)
// ANALYZER:     2: NoReturn() (CXXConstructExpr, [B2.3], NoReturn)
// CHECK:     3: [B2.2] (BindTemporary)
// CHECK:     4: 47
// CHECK:     5: ... , [B2.4]
// CHECK:     6: ~NoReturn() (Temporary object destructor)
// CHECK:     Preds (1): B3
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (2): B1 B2
// CHECK:   [B9 (ENTRY)]
// CHECK:     Succs (1): B8
// CHECK:   [B1]
// CHECK:     1: 0
// CHECK:     2: return [B1.1];
// CHECK:     Preds (2): B3 B8
// CHECK:     Succs (1): B0
// CHECK:   [B2]
// CHECK:     1: 1
// CHECK:     2: return [B2.1];
// CHECK:     Preds (1): B3
// CHECK:     Succs (1): B0
// CHECK:   [B3]
// CHECK:     T: if [B5.1]
// CHECK:     Preds (2): B4(Unreachable) B5
// CHECK:     Succs (2): B2 B1
// CHECK:   [B4 (NORETURN)]
// CHECK:     1: ~NoReturn() (Temporary object destructor)
// CHECK:     Preds (1): B5
// CHECK:     Succs (1): B0
// CHECK:   [B5]
// CHECK:     1: [B7.3] || [B6.7]
// CHECK:     T: (Temp Dtor) [B6.4]
// CHECK:     Preds (2): B6 B7
// CHECK:     Succs (2): B4 B3
// CHECK:   [B6]
// CHECK:     1: check
// CHECK:     2: [B6.1] (ImplicitCastExpr, FunctionToPointerDecay, _Bool (*)(const NoReturn &))
// WARNINGS:     3: NoReturn() (CXXConstructExpr, NoReturn)
// ANALYZER:     3: NoReturn() (CXXConstructExpr, [B6.4], [B6.6], NoReturn)
// CHECK:     4: [B6.3] (BindTemporary)
// CHECK:     5: [B6.4] (ImplicitCastExpr, NoOp, const NoReturn)
// CHECK:     6: [B6.5]
// CHECK:     7: [B6.2]([B6.6])
// CHECK:     Preds (1): B7
// CHECK:     Succs (1): B5
// CHECK:   [B7]
// CHECK:     1: value
// CHECK:     2: [B7.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:     3: ![B7.2]
// CHECK:     T: [B7.3] || ...
// CHECK:     Preds (1): B8
// CHECK:     Succs (2): B5 B6
// CHECK:   [B8]
// CHECK:     1: value
// CHECK:     2: [B8.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:     T: if [B8.2]
// CHECK:     Preds (1): B9
// CHECK:     Succs (2): B7 B1
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (3): B1 B2 B4
// CHECK:   [B10 (ENTRY)]
// CHECK:     Succs (1): B9
// CHECK:   [B1]
// CHECK:     1: 0
// CHECK:     2: return [B1.1];
// CHECK:     Preds (2): B3 B9
// CHECK:     Succs (1): B0
// CHECK:   [B2]
// CHECK:     1: 1
// CHECK:     2: return [B2.1];
// CHECK:     Preds (1): B3
// CHECK:     Succs (1): B0
// CHECK:   [B3]
// CHECK:     T: if [B5.1]
// CHECK:     Preds (2): B4(Unreachable) B5
// CHECK:     Succs (2): B2 B1
// CHECK:   [B4 (NORETURN)]
// CHECK:     1: ~NoReturn() (Temporary object destructor)
// CHECK:     Preds (1): B5
// CHECK:     Succs (1): B0
// CHECK:   [B5]
// CHECK:     1: [B8.3] || [B7.3] || [B6.7]
// CHECK:     T: (Temp Dtor) [B6.4]
// CHECK:     Preds (3): B6 B7 B8
// CHECK:     Succs (2): B4 B3
// CHECK:   [B6]
// CHECK:     1: check
// CHECK:     2: [B6.1] (ImplicitCastExpr, FunctionToPointerDecay, _Bool (*)(const NoReturn &))
// WARNINGS:     3: NoReturn() (CXXConstructExpr, NoReturn)
// ANALYZER:     3: NoReturn() (CXXConstructExpr, [B6.4], [B6.6], NoReturn)
// CHECK:     4: [B6.3] (BindTemporary)
// CHECK:     5: [B6.4] (ImplicitCastExpr, NoOp, const NoReturn)
// CHECK:     6: [B6.5]
// CHECK:     7: [B6.2]([B6.6])
// CHECK:     Preds (1): B7
// CHECK:     Succs (1): B5
// CHECK:   [B7]
// CHECK:     1: value
// CHECK:     2: [B7.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:     3: ![B7.2]
// CHECK:     T: [B8.3] || [B7.3] || ...
// CHECK:     Preds (1): B8
// CHECK:     Succs (2): B5 B6
// CHECK:   [B8]
// CHECK:     1: value
// CHECK:     2: [B8.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:     3: ![B8.2]
// CHECK:     T: [B8.3] || ...
// CHECK:     Preds (1): B9
// CHECK:     Succs (2): B5 B7
// CHECK:   [B9]
// CHECK:     1: value
// CHECK:     2: [B9.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:     T: if [B9.2]
// CHECK:     Preds (1): B10
// CHECK:     Succs (2): B8 B1
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (3): B1 B2 B4
// CHECK:   [B10 (ENTRY)]
// CHECK:     Succs (1): B9
// CHECK:   [B1]
// CHECK:     1: 0
// CHECK:     2: return [B1.1];
// CHECK:     Preds (2): B3 B9
// CHECK:     Succs (1): B0
// CHECK:   [B2]
// CHECK:     1: 1
// CHECK:     2: return [B2.1];
// CHECK:     Preds (1): B3
// CHECK:     Succs (1): B0
// CHECK:   [B3]
// CHECK:     T: if [B5.1]
// CHECK:     Preds (2): B4(Unreachable) B5
// CHECK:     Succs (2): B2 B1
// CHECK:   [B4 (NORETURN)]
// CHECK:     1: ~NoReturn() (Temporary object destructor)
// CHECK:     Preds (1): B5(Unreachable)
// CHECK:     Succs (1): B0
// CHECK:   [B5]
// CHECK:     1: [B8.3] || [B7.2] || [B6.7]
// CHECK:     T: (Temp Dtor) [B6.4]
// CHECK:     Preds (3): B6 B7 B8
// CHECK:     Succs (2): B4(Unreachable) B3
// CHECK:   [B6]
// CHECK:     1: check
// CHECK:     2: [B6.1] (ImplicitCastExpr, FunctionToPointerDecay, _Bool (*)(const NoReturn &))
// WARNINGS:     3: NoReturn() (CXXConstructExpr, NoReturn)
// ANALYZER:     3: NoReturn() (CXXConstructExpr, [B6.4], [B6.6], NoReturn)
// CHECK:     4: [B6.3] (BindTemporary)
// CHECK:     5: [B6.4] (ImplicitCastExpr, NoOp, const NoReturn)
// CHECK:     6: [B6.5]
// CHECK:     7: [B6.2]([B6.6])
// CHECK:     Preds (1): B7
// CHECK:     Succs (1): B5
// CHECK:   [B7]
// CHECK:     1: value
// CHECK:     2: [B7.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:     T: [B8.3] || [B7.2] || ...
// CHECK:     Preds (1): B8
// CHECK:     Succs (2): B5 B6
// CHECK:   [B8]
// CHECK:     1: value
// CHECK:     2: [B8.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:     3: ![B8.2]
// CHECK:     T: [B8.3] || ...
// CHECK:     Preds (1): B9
// CHECK:     Succs (2): B5 B7
// CHECK:   [B9]
// CHECK:     1: value
// CHECK:     2: [B9.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:     T: if [B9.2]
// CHECK:     Preds (1): B10
// CHECK:     Succs (2): B8 B1
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (3): B1 B2 B4
// CHECK:   [B1 (ENTRY)]
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B2 (ENTRY)]
// CHECK:     Succs (1): B1
// CHECK:   [B1]
// CHECK:     1: foo1
// CHECK:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, const C &(*)(void))
// CHECK:     3: [B1.2]()
// CHECK:     4: return [B1.3];
// CHECK:     Preds (1): B2
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
// CHECK:   [B2 (ENTRY)]
// CHECK:     Succs (1): B1
// CHECK:   [B1]
// CHECK:     1: foo2
// CHECK:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, C &&(*)(void))
// CHECK:     3: [B1.2]()
// CHECK:     4: return [B1.3];
// CHECK:     Preds (1): B2
// CHECK:     Succs (1): B0
// CHECK:   [B0 (EXIT)]
// CHECK:     Preds (1): B1
