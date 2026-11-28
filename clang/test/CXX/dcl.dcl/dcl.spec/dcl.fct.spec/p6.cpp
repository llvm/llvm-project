// RUN: %clang_cc1 -verify %s
// RUN: %clang_cc1 -verify -std=c++98 %s
// RUN: %clang_cc1 -verify -std=c++11 %s

class A {
public:
  explicit A();

  explicit operator int();

  explicit void f0(); // expected-error {{'explicit' can only be applied to a constructor or conversion function}}

  operator bool();
};

explicit A::A() { } // expected-error {{'explicit' can only be specified inside the class definition}}
explicit A::operator bool() { return false; }
// expected-error@-1 {{'explicit' can only be specified inside the class definition}}

class B {
  friend explicit A::A(); // expected-error {{'explicit' is invalid in friend declarations}}
};
