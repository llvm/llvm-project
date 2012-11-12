// RUN: %clang_cc1 %s -fsyntax-only -verify -Wweak-vtables -Wweak-template-vtables

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

  A *a;
  a->f();
}

// Use the vtables
void uses(A &a, B<int> &b, C &c) {
  a.f();
  b.f();
  c.f();
}

// <rdar://problem/9979458>
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

void uses(Parent &p, Derived &d, VeryDerived &vd) {
  p.getFoo();
  d.getFoo();
  vd.getFoo();
}

template<typename T> struct TemplVirt {
  virtual void f();
};

template class TemplVirt<float>; // expected-warning{{explicit template instantiation 'TemplVirt<float>' will emit a vtable in every translation unit}}

template<> struct TemplVirt<bool> {
  virtual void f();
};

template<> struct TemplVirt<long> { // expected-warning{{'TemplVirt<long>' has no out-of-line virtual method definitions; its vtable will be emitted in every translation unit}}
  virtual void f() {}
};

void uses(TemplVirt<float>& f, TemplVirt<bool>& b, TemplVirt<long>& l) {
  f.f();
  b.f();
  l.f();
}
