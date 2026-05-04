// RUN: %clang_cc1 -fsyntax-only -Wredundant-move -std=c++11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wredundant-move -std=c++11 -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++11 %s -ast-dump | FileCheck %s --check-prefix=CHECK-AST

// definitions for std::move
namespace std {
inline namespace foo {
template <class T> struct remove_reference { typedef T type; };
template <class T> struct remove_reference<T&> { typedef T type; };
template <class T> struct remove_reference<T&&> { typedef T type; };

template <class T> typename remove_reference<T>::type &&move(T &&t);
}
}

// test1 and test2 should not warn until after implementation of DR1579.
struct A {};
struct B : public A {};

A test1(B b1) {
  B b2;
  return b1;
  return b2;
  return std::move(b1);
  return std::move(b2);
}

struct C {
  C() {}
  C(A) {}
};

C test2(A a1, B b1) {
  A a2;
  B b2;

  return a1;
  return a2;
  return b1;
  return b2;

  return std::move(a1);
  return std::move(a2);
  return std::move(b1);
  return std::move(b2);
}

// Copy of tests above with types changed to reference types.
A test3(B& b1) {
  B& b2 = b1;
  return b1;
  return b2;
  return std::move(b1);
  return std::move(b2);
}

C test4(A& a1, B& b1) {
  A& a2 = a1;
  B& b2 = b1;

  return a1;
  return a2;
  return b1;
  return b2;

  return std::move(a1);
  return std::move(a2);
  return std::move(b1);
  return std::move(b2);
}

// PR23819, case 2
struct D {};
D test5(D d) {
  return d;
  // Verify the implicit move from the AST dump
  // CHECK-AST: ReturnStmt{{.*}}line:[[@LINE-2]]
  // CHECK-AST-NEXT: CXXConstructExpr{{.*}}D{{.*}}void (D &&)
  // CHECK-AST-NEXT: ImplicitCastExpr
  // CHECK-AST-NEXT: DeclRefExpr{{.*}}ParmVar{{.*}}'d'

  return std::move(d);
  // expected-warning@-1{{redundant move in return statement}}
  // expected-note@-2{{remove std::move call here}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:20}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:21-[[@LINE-4]]:22}:""
}

namespace templates {
  struct A {};
  struct B { B(A); };

  // Warn once here since the type is not dependent.
  template <typename T>
  A test1(A a) {
    return std::move(a);
    // expected-warning@-1{{redundant move in return statement}}
    // expected-note@-2{{remove std::move call here}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:22}:""
    // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:23-[[@LINE-4]]:24}:""
  }
  void run_test1() {
    test1<A>(A());
    test1<B>(A());
  }

  // T1 and T2 may not be the same, the warning may not always apply.
  template <typename T1, typename T2>
  T1 test2(T2 t) {
    return std::move(t);
  }
  void run_test2() {
    test2<A, A>(A());
    test2<B, A>(A());
  }
}
