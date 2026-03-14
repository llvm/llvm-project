// RUN: %clang_cc1 -verify -std=c++11 %s
// expected-no-diagnostics
template <typename T> struct OwnPtr {
  T *p;
  ~OwnPtr() {
    static_assert(sizeof(T) > 0, "incomplete T");
    delete p;
  }
};

namespace use_vtable_for_vcall {
struct Incomplete;
struct A {
  virtual ~A() {}
  virtual void m() {}
};
struct B : A {
  B();
  virtual void m() { }
  virtual void m2() { static_cast<A *>(this)->m(); }
  OwnPtr<Incomplete> m_sqlError;
};

void f() {
  // Since B's constructor is declared out of line, nothing in this file
  // references a vtable, so the destructor doesn't get built.
  A *b = new B();
  b->m();
  delete b;
}
}

namespace dont_mark_qualified_vcall {
struct Incomplete;
struct A {
  virtual ~A() {}
  virtual void m() {}
};
struct B : A {
  B();
  // Previously we would mark B's vtable referenced to devirtualize this call to
  // A::m, even though it's not a virtual call.
  virtual void m() { A::m(); }
  OwnPtr<Incomplete> m_sqlError;
};

B *f() {
  return new B();
}
}
