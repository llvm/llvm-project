// RUN: %clang_cc1 -std=c++17 -fsyntax-only -fobjc-arc -verify %s
// RUN: %clang_cc1 -std=c++17 -fobjc-arc -ast-dump %s 2>&1 | FileCheck %s
// expected-no-diagnostics

// Test for binding ObjC ARC __strong rvalues to const __autoreleasing references.
// This previously caused an assertion failure in Qualifiers::addConsistentQualifiers
// when the compiler attempted to add conflicting ObjC lifetime qualifiers.

// The const id& parameter has implicit __autoreleasing lifetime.
void take(const id&);

// CHECK-LABEL: FunctionDecl {{.*}} test_rvalue_binding
// CHECK: CallExpr
// CHECK: ImplicitCastExpr {{.*}} 'const __autoreleasing id' xvalue <NoOp>
// CHECK-NEXT: CXXStaticCastExpr {{.*}} '__strong id' xvalue static_cast<__strong id &&> <NoOp>
void test_rvalue_binding() {
  id obj = nullptr;
  take(static_cast<id&&>(obj));
}

// CHECK-LABEL: FunctionDecl {{.*}} test_lvalue_binding
// CHECK: CallExpr
// CHECK: ImplicitCastExpr {{.*}} 'const __autoreleasing id' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} '__strong id' lvalue
void test_lvalue_binding() {
  id obj = nullptr;
  take(obj);
}

// Test with fold expressions and perfect forwarding (original crash case).
template <typename... Args>
void call(Args... args) {
  (take(static_cast<Args&&>(args)), ...);
}

// CHECK-LABEL: FunctionDecl {{.*}} test_fold_expression
void test_fold_expression() {
  call<id>(nullptr);
}
