// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

class A {
  class AInner {
  };

  void a_member();
  friend void A::a_member(); // ok in c++11, ill-formed in c++98
  friend void a_member(); // ok in both, refers to non-member
  friend class A::AInner; // ok in c++11, extension in c++98
  friend class AInner; // ok in both, refers to non-member
};
