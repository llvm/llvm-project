// RUN: %clang_cc1 -std=c++2a -verify %s -Wdeprecated

// This test does two things.
// Deleting the copy constructor ensures that an [=, this] capture doesn't copy the object.
// Accessing a member variable from the lambda ensures that the capture actually works.
class A {
  A(const A &) = delete;
  int i;

  void func() {
    auto L = [=, this]() -> int { return i; };
    L();
  }
};

struct B {
  int i;
  void f() {
    (void) [=] { // expected-note {{add an explicit capture of 'this'}}
      return i; // expected-warning {{implicit capture of 'this' with a capture default of '=' is deprecated}}
    };
  }
};
