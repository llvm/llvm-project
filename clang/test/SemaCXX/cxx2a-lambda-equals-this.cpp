// RUN: %clang_cc1 -std=c++2a -verify %s
// expected-no-diagnostics

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
