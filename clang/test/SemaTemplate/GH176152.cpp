// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

template <class T> int f(T) {
  struct MyClass {
    static int staticField;
    // expected-error@-1 {{static data member 'staticField' not allowed in local struct 'MyClass'}}
  };
  int MyClass::staticField = 42;
  return 0;
}

int x = f(0);
