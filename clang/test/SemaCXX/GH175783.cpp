// RUN: %clang_cc1 -fspell-checking-limit=0 -verify %s

namespace GH175783 {
  class B {
  public:
    virtual void foo(); // #foo
  };
  void (*p)() = &GH175783::foo;
  // expected-error@-1 {{no member named 'foo' in namespace 'GH175783'; did you mean 'B::foo'?}}
  // expected-error@-2 {{cannot initialize a variable of type 'void (*)()' with an rvalue of type 'void (B::*)()'}}
  // expected-note@#foo {{'B::foo' declared here}}
} // namespace GH175783
