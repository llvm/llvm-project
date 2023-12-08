// RUN: %check_clang_tidy %s cert-oop58-cpp %t

// Example test cases from CERT rule
// https://wiki.sei.cmu.edu/confluence/display/cplusplus/OOP58-CPP.+Copy+operations+must+not+mutate+the+source+object
namespace test_mutating_noncompliant_example {
class A {
  mutable int m;

public:
  A() : m(0) {}
  explicit A(int m) : m(m) {}

  A(const A &other) : m(other.m) {
    other.m = 0;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: mutating copied object
  }

  A &operator=(const A &other) {
    if (&other != this) {
      m = other.m;
      other.m = 0;
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: mutating copied object
    }
    return *this;
  }

  int get_m() const { return m; }
};
} // namespace test_mutating_noncompliant_example

namespace test_mutating_compliant_example {
class B {
  int m;

public:
  B() : m(0) {}
  explicit B(int m) : m(m) {}

  B(const B &other) : m(other.m) {}
  B(B &&other) : m(other.m) {
    other.m = 0; //no-warning: mutation allowed in move constructor
  }

  B &operator=(const B &other) {
    if (&other != this) {
      m = other.m;
    }
    return *this;
  }

  B &operator=(B &&other) {
    m = other.m;
    other.m = 0; //no-warning: mutation allowed in move assignment operator
    return *this;
  }

  int get_m() const { return m; }
};
} // namespace test_mutating_compliant_example

namespace test_mutating_pointer {
class C {
  C *ptr;
  int value;

  C();
  C(C &other) {
    other = {};
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: mutating copied object
    other.ptr = nullptr;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: mutating copied object
    other.value = 0;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: mutating copied object

    // no-warning: mutating a pointee is allowed
    other.ptr->value = 0;
    *other.ptr = {};
  }
};
} // namespace test_mutating_pointer

namespace test_mutating_indirect_member {
struct S {
  int x;
};

class D {
  S s;
  D(D &other) {
    other.s = {};
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: mutating copied object
    other.s.x = 0;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: mutating copied object
  }
};
} // namespace test_mutating_indirect_member

namespace test_mutating_other_object {
class E {
  E();
  E(E &other) {
    E tmp;
    // no-warning: mutating an object that is not the source is allowed
    tmp = {};
  }
};
} // namespace test_mutating_other_object

namespace test_mutating_member_function {
class F {
  int a;

public:
  void bad_func() { a = 12; }
  void fine_func() const;
  void fine_func_2(int x) { x = 5; }
  void questionable_func();

  F(F &other) : a(other.a) {
    this->bad_func(); // no-warning: mutating this is allowed

    other.bad_func();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: call mutates copied object

    other.fine_func();
    other.fine_func_2(42);
    other.questionable_func();
  }
};
} // namespace test_mutating_member_function

namespace test_mutating_function_on_nested_object {
struct S {
  int x;
  void mutate(int y) {
    x = y;
  }
};

class G {
  S s;
  G(G &other) {
    s.mutate(0); // no-warning: mutating this is allowed

    other.s.mutate(0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: call mutates copied object
  }
};
} // namespace test_mutating_function_on_nested_object
