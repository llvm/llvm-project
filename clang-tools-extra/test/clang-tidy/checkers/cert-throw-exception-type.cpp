// RUN: %check_clang_tidy -std=c++11,c++14 %s cert-err60-cpp %t -- -- -fcxx-exceptions
// FIXME: Split off parts of this test that rely on dynamic exception
// specifications, and run this test in all language modes.

struct S {};
struct T : S {};
struct U {
  U() = default;
  U(const U&) = default;
};

struct V {
  V() = default;
  V(const V&) noexcept;
};

struct W {
  W() = default;
  W(const W&) noexcept(false);
};

struct X {
  X() = default;
  X(const X&) {}
};

struct Y {
  Y() = default;
  Y(const Y&) throw();
};

struct Z {
  Z() = default;
  Z(const Z&) throw(int);
};

void g() noexcept(false);

struct A {
  A() = default;
  A(const A&) noexcept(noexcept(g()));
};

struct B {
  B() = default;
  B(const B&) = default;
  B(const A&) noexcept(false);
};

class C {
  W M; // W is not no-throw copy constructible
public:
  C() = default;
  C(const C&) = default;
};

struct D {
  D() = default;
  D(const D&) noexcept(false);
  D(D&) noexcept(true);
};

struct E {
  E() = default;
  E(E&) noexcept(true);
  E(const E&) noexcept(false);
};

struct Allocates {
  int *x;
  Allocates() : x(new int(0)) {}
  Allocates(const Allocates &other) : x(new int(*other.x)) {}
};

struct OptionallyAllocates {
  int *x;
  OptionallyAllocates() : x(new int(0)) {}
  OptionallyAllocates(const Allocates &other) noexcept(true) {
    try {
      x = new int(*other.x);
    } catch (...) {
      x = nullptr;
    }
  }
};

void f() {
  throw 12; // ok
  throw "test"; // ok
  throw S(); // ok
  throw T(); // ok
  throw U(); // ok
  throw V(); // ok
  throw W(); // match, noexcept(false)
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: thrown exception type is not nothrow copy constructible [cert-err60-cpp]
  throw X(); // match, no noexcept clause, nontrivial
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: thrown exception type is not nothrow copy constructible
  throw Y(); // ok
  throw Z(); // match, throw(int)
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: thrown exception type is not nothrow copy constructible
  throw A(); // match, noexcept(false)
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: thrown exception type is not nothrow copy constructible
  throw B(); // ok
  throw C(); // match, C has a member variable that makes it throwing on copy
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: thrown exception type is not nothrow copy constructible
  throw D(); // match, has throwing copy constructor
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: thrown exception type is not nothrow copy constructible
  throw E(); // match, has throwing copy constructor
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: thrown exception type is not nothrow copy constructible
  throw Allocates(); // match, copy constructor throws
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: thrown exception type is not nothrow copy constructible
  throw OptionallyAllocates(); // ok
}

namespace PR25574 {
struct B {
  B(const B&) noexcept;
};

struct D : B {
  D();
  virtual ~D() noexcept;
};

template <typename T>
void f() {
  throw D();
}
}
