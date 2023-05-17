// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux -analyzer-checker=debug.DumpCFG -std=c++14 -analyzer-config eagerly-assume=false %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux -analyzer-checker=core,debug.ExprInspection -std=c++14 -verify -analyzer-config eagerly-assume=false %s

void clang_analyzer_eval(bool);

int global;

namespace variant_0 {
// This variant of the code works correctly. Function foo() is not a template
// function. Note that there are two destructors within foo().

class A {
public:
  ~A() { ++global; }
};

class B {
  A a;
};

// CHECK: void foo(int)
// CHECK:       [B1]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B1.2], B)
// CHECK-NEXT:    2: B i;
// CHECK-NEXT:    3: operator=
// CHECK-NEXT:    4: [B1.3] (ImplicitCastExpr, FunctionToPointerDecay, B &(*)(B &&) noexcept)
// CHECK-NEXT:    5: i
// CHECK-NEXT:    6: {} (CXXConstructExpr, [B1.7], [B1.8], B)
// CHECK-NEXT:    7: [B1.6] (BindTemporary)
// CHECK-NEXT:    8: [B1.7]
// CHECK-NEXT:    9: [B1.5] = [B1.8] (OperatorCall)
// CHECK-NEXT:   10: ~B() (Temporary object destructor)
// CHECK-NEXT:   11: [B1.2].~B() (Implicit destructor)
void foo(int) {
  B i;
  i = {};
}

void bar() {
  global = 0;
  foo(1);
  clang_analyzer_eval(global == 2); // expected-warning{{TRUE}}
}

} // end namespace variant_0

namespace variant_1 {
// Suddenly, if we turn foo() into a template, we are missing a
// CXXBindTemporaryExpr in the AST, and therefore we're missing a
// temporary destructor in the CFG.

class A {
public:
  ~A() { ++global; }
};

class B {
  A a;
};

// FIXME: Find the construction context for {} and enforce the temporary
// destructor.
// CHECK: template<> void foo<int>(int)
// CHECK:       [B1]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B1.2], B)
// CHECK-NEXT:    2: B i;
// CHECK-NEXT:    3: operator=
// CHECK-NEXT:    4: [B1.3] (ImplicitCastExpr, FunctionToPointerDecay, B &(*)(B &&) noexcept)
// CHECK-NEXT:    5: i
// CHECK-NEXT:    6: {} (CXXConstructExpr, B)
// CHECK-NEXT:    7: [B1.6]
// CHECK-NEXT:    8: [B1.5] = [B1.7] (OperatorCall)
// CHECK-NEXT:    9: [B1.2].~B() (Implicit destructor)
template <typename T> void foo(T) {
  B i;
  i = {};
}

void bar() {
  global = 0;
  foo(1);
  // FIXME: Should be TRUE, i.e. we should call (and inline) two destructors.
  clang_analyzer_eval(global == 2); // expected-warning{{UNKNOWN}}
}

} // end namespace variant_1

namespace variant_2 {
// Making field 'a' in class 'B' public turns the class into an aggregate.
// In this case there is no constructor at {} - only an aggregate
// initialization. Aggregate initialization is unsupported for now.

class A {
public:
  ~A() { ++global; }
};

class B {
public:
  A a;
};

// CHECK: template<> void foo<int>(int)
// CHECK:       [B1]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B1.2], B)
// CHECK-NEXT:    2: B i;
// CHECK-NEXT:    3: operator=
// CHECK-NEXT:    4: [B1.3] (ImplicitCastExpr, FunctionToPointerDecay, B &(*)(B &&) noexcept)
// CHECK-NEXT:    5: i
// CHECK-NEXT:    6: {}
// CHECK-NEXT:    7: {}
// CHECK-NEXT:    8: [B1.7] (BindTemporary)
// CHECK-NEXT:    9: [B1.8]
// CHECK-NEXT:   10: [B1.5] = [B1.9] (OperatorCall)
// CHECK-NEXT:   11: ~B() (Temporary object destructor)
// CHECK-NEXT:   12: [B1.2].~B() (Implicit destructor)
template <typename T> void foo(T) {
  B i;
  i = {};
}

void bar() {
  global = 0;
  foo(1);
  // FIXME: Should be TRUE, i.e. we should call (and inline) two destructors.
  clang_analyzer_eval(global == 2); // expected-warning{{UNKNOWN}}
}

} // end namespace variant_2
