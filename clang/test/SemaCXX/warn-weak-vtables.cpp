// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 %t/warn-weak-vtables.cpp -fsyntax-only -verify -triple %itanium_abi_triple -Wweak-vtables
//
// Check that this warning is disabled on MS ABI targets which don't have key
// functions.
// RUN: %clang_cc1 %t/warn-weak-vtables.cpp -fsyntax-only -triple %ms_abi_triple -Werror -Wweak-vtables
//
// -Wweak-template-vtables is deprecated but we still parse it.
// RUN: %clang_cc1 %t/warn-weak-vtables.cpp -fsyntax-only -Werror -Wweak-template-vtables
//
// Test that -Wweak-vtables is not emitted for classes in named module units.
// RUN: %clang_cc1 -std=c++20 -verify -triple %itanium_abi_triple -Wweak-vtables -emit-module-interface %t/module-weak-vtable.cpp -o %t/m.pcm

//--- warn-weak-vtables.cpp
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

namespace GH195110 {
// Check that no warning is emitted on a template instantiation.
template <class> struct basic_streambuf {
  __attribute__((__exclude_from_explicit_instantiation__)) virtual void
  overflow();

  virtual ~basic_streambuf() {}
};
extern template class basic_streambuf<char>;

basic_streambuf<char> b;
}

//--- module-weak-vtable.cpp
// expected-no-diagnostics
export module m;

struct s {
    virtual void f() {}
};

export struct t {
    virtual void g() {}
};
