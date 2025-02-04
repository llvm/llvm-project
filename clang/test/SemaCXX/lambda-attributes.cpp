// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-linux -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-linux -fsyntax-only -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-linux -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++23 -triple x86_64-unknown-linux -include-pch %t -ast-dump-all /dev/null | FileCheck %s
// expected-no-diagnostics

// Check that we both don't crash on transforming FunctionProtoType's
// wrapped in type sugar and that we don't drop it when performing
// instantiations either.

#define PRESERVE __attribute__((preserve_most))

// Skip to the instantiation of f().
// CHECK: FunctionDecl {{.*}} f 'void ()' implicit_instantiation
template <typename T>
void f() {
  // CHECK: CXXMethodDecl {{.*}} operator() 'void (int) const __attribute__((preserve_most))':'void (int) __attribute__((preserve_most)) const' implicit_instantiation
  (void) [] (T) __attribute__((preserve_most)) { };

  // CHECK: CXXMethodDecl {{.*}} operator() 'void (int) const {{\[}}[clang::annotate_type(...)]]':'void (int) const' implicit_instantiation
  (void) [] (T) [[clang::annotate_type("foo")]] { };

  // CHECK: CXXMethodDecl {{.*}} operator() 'void (int) const {{\[}}[clang::annotate_type(...)]] {{\[}}[clang::annotate_type(...)]] {{\[}}[clang::annotate_type(...)]]':'void (int) const' implicit_instantiation
  (void) [] (T) [[clang::annotate_type("foo")]]
                [[clang::annotate_type("foo")]]
                [[clang::annotate_type("foo")]] { };

  // CHECK: CXXMethodDecl {{.*}} operator() 'void (int) const __attribute__((preserve_most)) {{\[}}[clang::annotate_type(...)]]':'void (int) __attribute__((preserve_most)) const' implicit_instantiation
  (void) [] (T) __attribute__((preserve_most))
                [[clang::annotate_type("foo")]] { };

  // CHECK: CXXMethodDecl {{.*}} operator() 'void (int) const __attribute__((cdecl)) {{\[}}[clang::annotate_type(...)]]':'void (int) const' implicit_instantiation
  (void) [] (T) __attribute__((cdecl))
                [[clang::annotate_type("foo")]] { };

  // CHECK: CXXMethodDecl {{.*}} operator() 'void (int) const {{\[}}[clang::annotate_type(...)]]':'void (int) const' implicit_instantiation
  (void) [] (T t) [[clang::annotate_type("foo", t)]] { };

  // CHECK: CXXMethodDecl {{.*}} operator() 'void (int) const __attribute__((preserve_most)) {{\[}}[clang::annotate_type(...)]]':'void (int) __attribute__((preserve_most)) const' implicit_instantiation
  (void) [] (T t) __attribute__((preserve_most))
                [[clang::annotate_type("foo", t, t, t, t)]] { };

  // Check that the MacroQualifiedType is preserved.
  // CHECK: CXXMethodDecl {{.*}} operator() 'PRESERVE void (int) __attribute__((preserve_most)) const':'void (int) __attribute__((preserve_most)) const' implicit_instantiation
  (void) [] (T) PRESERVE { };

  // CHECK: CXXMethodDecl {{.*}} operator() 'PRESERVE void (int) __attribute__((preserve_most)) const {{\[}}[clang::annotate_type(...)]]':'void (int) __attribute__((preserve_most)) const' implicit_instantiation
  (void) [] (T) PRESERVE [[clang::annotate_type("foo")]] { };

  // CHECK: CXXMethodDecl {{.*}} operator() 'void (int) const {{\[}}[clang::annotate_type(...)]]':'void (int) const' implicit_instantiation
  (void) [] (T) [[clang::annotate_type("foo")]] {
    // CHECK: CXXMethodDecl {{.*}} operator() 'PRESERVE void (int) __attribute__((preserve_most)) const {{\[}}[clang::annotate_type(...)]]':'void (int) __attribute__((preserve_most)) const' implicit_instantiation
    auto l = []<typename U = T> (U u = {}) PRESERVE [[clang::annotate_type("foo", u)]] { };

    // CHECK: DeclRefExpr {{.*}} 'PRESERVE void (int) __attribute__((preserve_most)) const {{\[}}[clang::annotate_type(...)]]':'void (int) __attribute__((preserve_most)) const' lvalue CXXMethod
    l();
  };
}

void g() {
  f<int>();
}
