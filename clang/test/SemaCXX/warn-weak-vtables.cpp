// RUN: %clang_cc1 %s -fsyntax-only -verify -triple %itanium_abi_triple -Wweak-vtables
//
// Check that this warning is disabled on MS ABI targets which don't have key
// functions.
// RUN: %clang_cc1 %s -fsyntax-only -triple %ms_abi_triple -Werror -Wweak-vtables
//
// -Wweak-template-vtables is deprecated but we still parse it.
// RUN: %clang_cc1 %s -fsyntax-only -Werror -Wweak-template-vtables

struct A { // expected-warning {{'A' has no out-of-line virtual method definitions; its vtable will be emitted in every translation unit}}
  virtual void f() { } 
};

template<typename T> struct B {
  virtual void f() { } 
};

namespace {
  struct C { 
    virtual void f() { }
  };
}

void f() {
  struct A {
    virtual void f() { }
  };

  A a;
}

// Use the vtables
void uses_abc() {
  A a;
  B<int> b;
  C c;
}

class Parent {
public:
  Parent() {}
  virtual ~Parent();
  virtual void * getFoo() const = 0;    
};
  
class Derived : public Parent {
public:
  Derived();
  void * getFoo() const;
};

class VeryDerived : public Derived { // expected-warning{{'VeryDerived' has no out-of-line virtual method definitions; its vtable will be emitted in every translation unit}}
public:
  void * getFoo() const { return 0; }
};

Parent::~Parent() {}

void uses_derived() {
  Derived d;
  VeryDerived vd;
}

template<typename T> struct TemplVirt {
  virtual void f();
};

template class TemplVirt<float>;

template<> struct TemplVirt<bool> {
  virtual void f();
};

template<> struct TemplVirt<long> { // expected-warning{{'TemplVirt<long>' has no out-of-line virtual method definitions; its vtable will be emitted in every translation unit}}
  virtual void f() {}
};

void uses_templ() {
  TemplVirt<float> f;
  TemplVirt<bool> b;
  TemplVirt<long> l;
}
