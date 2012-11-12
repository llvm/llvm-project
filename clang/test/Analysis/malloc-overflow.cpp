// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.security.MallocOverflow -verify %s
// expected-no-diagnostics

class A {
public:
  A& operator<<(const A &a);
};

void f() {
  A a = A(), b = A();
  a << b;
}
