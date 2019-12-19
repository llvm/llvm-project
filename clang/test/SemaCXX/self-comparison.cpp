// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++2a

int foo(int x) {
  return x == x; // expected-warning {{self-comparison always evaluates to true}}
}

struct X {
  bool operator==(const X &x) const;
};

struct A {
  int x;
  X x2;
  int a[3];
  int b[3];
  bool f() { return x == x; } // expected-warning {{self-comparison always evaluates to true}}
  bool g() { return x2 == x2; } // no-warning
  bool h() { return a == b; } // expected-warning {{array comparison always evaluates to false}} expected-warning {{deprecated}}
  bool i() {
    int c[3];
    return a == c; // expected-warning {{array comparison always evaluates to false}} expected-warning {{deprecated}}
  }
};

namespace NA { extern "C" int x[3]; }
namespace NB { extern "C" int x[3]; }
bool k = NA::x == NB::x; // expected-warning {{self-comparison always evaluates to true}} expected-warning {{deprecated}}

template<typename T> struct Y { static inline int n; };
bool f() {
  return
    Y<int>::n == Y<int>::n || // expected-warning {{self-comparison always evaluates to true}}
    Y<void>::n == Y<int>::n;
}
template<typename T, typename U>
bool g() {
  // FIXME: Ideally we'd produce a self-comparison warning on the first of these.
  return
    Y<T>::n == Y<T>::n ||
    Y<T>::n == Y<U>::n;
}
template bool g<int, int>(); // should not produce any warnings

namespace member_tests {
struct B {
  int field;
  static int static_field;
  int test(B b) {
    return field == field;  // expected-warning {{self-comparison always evaluates to true}}
    return static_field == static_field;  // expected-warning {{self-comparison always evaluates to true}}
    return static_field == b.static_field;  // expected-warning {{self-comparison always evaluates to true}}
    return B::static_field == this->static_field;  // expected-warning {{self-comparison always evaluates to true}}
    return this == this;  // expected-warning {{self-comparison always evaluates to true}}

    return field == b.field;
    return this->field == b.field;
  }
};

enum {
  I0,
  I1,
  I2,
};

struct S {
  int field;
  static int static_field;
  int array[4];
};

struct T {
  int field;
  static int static_field;
  int array[4];
  S s;
};

int struct_test(S s1, S s2, S *s3, T t) {
  return s1.field == s1.field;  // expected-warning {{self-comparison always evaluates to true}}
  return s2.field == s2.field;  // expected-warning {{self-comparison always evaluates to true}}
  return s1.static_field == s2.static_field;  // expected-warning {{self-comparison always evaluates to true}}
  return S::static_field == s1.static_field;  // expected-warning {{self-comparison always evaluates to true}}
  return s1.array == s1.array;  // expected-warning {{self-comparison always evaluates to true}} expected-warning {{deprecated}}
  return t.s.static_field == S::static_field;  // expected-warning {{self-comparison always evaluates to true}}
  return s3->field == s3->field;  // expected-warning {{self-comparison always evaluates to true}}
  return s3->static_field == S::static_field;  // expected-warning {{self-comparison always evaluates to true}}
  return s1.array[0] == s1.array[0];  // expected-warning {{self-comparison always evaluates to true}}
  return s1.array[0] == s1.array[0ull];  // expected-warning {{self-comparison always evaluates to true}}
  return s1.array[I1] == s1.array[I1];  // expected-warning {{self-comparison always evaluates to true}}
  return s1.array[s2.array[0]] == s1.array[s2.array[0]];  // expected-warning {{self-comparison always evaluates to true}}
  return s3->array[t.field] == s3->array[t.field];  // expected-warning {{self-comparison always evaluates to true}}

  // Try all operators
  return t.field == t.field;  // expected-warning {{self-comparison always evaluates to true}}
  return t.field <= t.field;  // expected-warning {{self-comparison always evaluates to true}}
  return t.field >= t.field;  // expected-warning {{self-comparison always evaluates to true}}

  return t.field != t.field;  // expected-warning {{self-comparison always evaluates to false}}
  return t.field < t.field;  // expected-warning {{self-comparison always evaluates to false}}
  return t.field > t.field;  // expected-warning {{self-comparison always evaluates to false}}

  // no warning
  return s1.field == s2.field;
  return s2.array == s1.array; // FIXME: This always evaluates to false. expected-warning {{deprecated}}
  return s2.array[0] == s1.array[0];
  return s1.array[I1] == s1.array[I2];

  return s1.static_field == t.static_field;
};

struct U {
  bool operator!=(const U&);
};

bool operator==(const U&, const U&);

// May want to warn on this in the future.
int user_defined(U u) {
  return u == u;
  return u != u;
}

} // namespace member_tests
