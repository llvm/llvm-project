// RUN: %clang_cc1 -fsyntax-only -verify %s

template <class T> int f(T) {
  struct A {
    static int B;
    // expected-error@-1 {{static data member 'staticField' not allowed in local struct 'MyClass'}}
  };
  int A::B;
  int A::B = 1;
  return 0;
}

int x = f(0);