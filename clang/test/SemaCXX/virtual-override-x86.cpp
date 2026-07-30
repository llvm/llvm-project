// RUN: %clang_cc1 -triple=i686-pc-win32 -fsyntax-only -verify %s -std=c++11

namespace PR14339 {
  class A {
  public:
    virtual void __attribute__((thiscall)) f();	// expected-note{{overridden virtual function is here}}
  };

  class B : public A {
  public:
    void __attribute__((cdecl)) f();  // expected-error{{virtual function 'f' has different calling convention attributes ('void () __attribute__((cdecl))') than the function it overrides (which has calling convention 'void () __attribute__((thiscall))'}}
  };

  class C : public A {
  public:
    void __attribute__((thiscall)) f();  // This override is correct
  };

  class D : public A {
  public:
    void f();  // This override is correct because thiscall is the default calling convention for class members
  };

  class E {
  public:
    virtual void __attribute__((stdcall)) g();  // expected-note{{overridden virtual function is here}}
  };

  class F : public E {
  public:
    void g();  // expected-error{{virtual function 'g' has different calling convention attributes ('void () __attribute__((thiscall))') than the function it overrides (which has calling convention 'void () __attribute__((stdcall))'}}
  };
}
