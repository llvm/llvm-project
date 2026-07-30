// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

template <class F1> int foo1(F1 X1);

template <int A1> struct A {
  template <class F2> friend int foo1(F2 X2) {
    return A1;
  }
};

template struct A<1>;
int main() { 
  foo1(1.0);
}

template <class F1> int foo2(F1 X1);

template <int A1> struct B {
  template <class F2> friend int foo2(F2 X2) {
    return A1;
  }
};

template struct B<1>;
template int foo2<float>(float X1);
