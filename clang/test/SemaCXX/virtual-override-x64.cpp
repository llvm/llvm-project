// RUN: %clang_cc1 -triple=x86_64-pc-unknown -fsyntax-only -verify %s

// Non-x86 targets ignore the calling conventions by default (but will warn
// when one is encountered), so we want to make sure the virtual overrides
// continue to work.
namespace PR14339 {
  class A {
  public:
    virtual void __attribute__((thiscall)) f();	// expected-warning {{'thiscall' calling convention is not supported for this target}}
  };

  class B : public A {
  public:
    void __attribute__((cdecl)) f();
  };

  class C : public A {
  public:
    void __attribute__((thiscall)) f();  // expected-warning {{'thiscall' calling convention is not supported for this target}}
  };

  class D : public A {
  public:
    void f();
  };

  class E {
  public:
    virtual void __attribute__((stdcall)) g();  // expected-warning {{'stdcall' calling convention is not supported for this target}}
  };

  class F : public E {
  public:
    void g();
  };
}
