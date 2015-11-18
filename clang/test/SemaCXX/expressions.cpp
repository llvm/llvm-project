// RUN: %clang_cc1 -fsyntax-only -verify -Wno-constant-conversion %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wno-constant-conversion -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wno-constant-conversion -std=c++11 %s

void choice(int);
int choice(bool);

void test() {
  // Result of ! must be type bool.
  int i = choice(!1);
}

// rdar://8018252
void f0() {
  extern void f0_1(int*);
  register int x;
#if __cplusplus >= 201103L // C++11 or later
  // expected-warning@-2 {{'register' storage class specifier is deprecated}}
#endif
  f0_1(&x);
}

namespace test1 {
  template <class T> void bar(T &x) { T::fail(); }
  template <class T> void bar(volatile T &x) {}

  void test_ints() {
    volatile int x;
    bar(x = 5);
    bar(x += 5);
  }

  enum E { E_zero };
  void test_enums() {
    volatile E x;
    bar(x = E_zero);
    bar(x += E_zero); // expected-error {{incompatible type}}
  }
}

int test2(int x) {
  return x && 4; // expected-warning {{use of logical '&&' with constant operand}} \
                   // expected-note {{use '&' for a bitwise operation}} \
                   // expected-note {{remove constant to silence this warning}}

  return x && sizeof(int) == 4;  // no warning, RHS is logical op.
  return x && true;
  return x && false;
  return x || true;
  return x || false;

  return x && (unsigned)0; // expected-warning {{use of logical '&&' with constant operand}} \
                   // expected-note {{use '&' for a bitwise operation}} \
                   // expected-note {{remove constant to silence this warning}}

  return x || (unsigned)1; // expected-warning {{use of logical '||' with constant operand}} \
                   // expected-note {{use '|' for a bitwise operation}}

  return x || 0; // expected-warning {{use of logical '||' with constant operand}} \
                   // expected-note {{use '|' for a bitwise operation}}
  return x || 1; // expected-warning {{use of logical '||' with constant operand}} \
                   // expected-note {{use '|' for a bitwise operation}}
  return x || -1; // expected-warning {{use of logical '||' with constant operand}} \
                   // expected-note {{use '|' for a bitwise operation}}
  return x || 5; // expected-warning {{use of logical '||' with constant operand}} \
                   // expected-note {{use '|' for a bitwise operation}}
  return x && 0; // expected-warning {{use of logical '&&' with constant operand}} \
                   // expected-note {{use '&' for a bitwise operation}} \
                   // expected-note {{remove constant to silence this warning}}
  return x && 1; // expected-warning {{use of logical '&&' with constant operand}} \
                   // expected-note {{use '&' for a bitwise operation}} \
                   // expected-note {{remove constant to silence this warning}}
  return x && -1; // expected-warning {{use of logical '&&' with constant operand}} \
                   // expected-note {{use '&' for a bitwise operation}} \
                   // expected-note {{remove constant to silence this warning}}
  return x && 5; // expected-warning {{use of logical '&&' with constant operand}} \
                   // expected-note {{use '&' for a bitwise operation}} \
                   // expected-note {{remove constant to silence this warning}}
  return x || (0); // expected-warning {{use of logical '||' with constant operand}} \
                   // expected-note {{use '|' for a bitwise operation}}
  return x || (1); // expected-warning {{use of logical '||' with constant operand}} \
                   // expected-note {{use '|' for a bitwise operation}}
  return x || (-1); // expected-warning {{use of logical '||' with constant operand}} \
                   // expected-note {{use '|' for a bitwise operation}}
  return x || (5); // expected-warning {{use of logical '||' with constant operand}} \
                   // expected-note {{use '|' for a bitwise operation}}
  return x && (0); // expected-warning {{use of logical '&&' with constant operand}} \
                   // expected-note {{use '&' for a bitwise operation}} \
                   // expected-note {{remove constant to silence this warning}}
  return x && (1); // expected-warning {{use of logical '&&' with constant operand}} \
                   // expected-note {{use '&' for a bitwise operation}} \
                   // expected-note {{remove constant to silence this warning}}
  return x && (-1); // expected-warning {{use of logical '&&' with constant operand}} \
                   // expected-note {{use '&' for a bitwise operation}} \
                   // expected-note {{remove constant to silence this warning}}
  return x && (5); // expected-warning {{use of logical '&&' with constant operand}} \
                   // expected-note {{use '&' for a bitwise operation}} \
                   // expected-note {{remove constant to silence this warning}}
}

template<unsigned int A, unsigned int B> struct S
{
  enum {
    e1 = A && B,
    e2 = A && 7      // expected-warning {{use of logical '&&' with constant operand}} \
                     // expected-note {{use '&' for a bitwise operation}} \
                     // expected-note {{remove constant to silence this warning}}
  };

  int foo() {
    int x = A && B;
    int y = B && 3;  // expected-warning {{use of logical '&&' with constant operand}} \
                     // expected-note {{use '&' for a bitwise operation}} \
                     // expected-note {{remove constant to silence this warning}}

    return x + y;
  }
};

void test3() {
  S<5, 8> s1;
  S<2, 7> s2;
  (void)s1.foo();
  (void)s2.foo();
}

namespace pr16992 {
  typedef int T;
  unsigned getsz() {
    return (sizeof T());
  }
}

void test4() {
  #define X 0
  #define Y 1
  bool r1 = X || Y;

  #define Y2 2
  bool r2 = X || Y2; // expected-warning {{use of logical '||' with constant operand}} \
                     // expected-note {{use '|' for a bitwise operation}}
}
