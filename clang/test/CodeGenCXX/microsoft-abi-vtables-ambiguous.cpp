// RUN: %clang_cc1 %s -emit-llvm-only -triple=i386-pc-win32 -verify -DTEST1
// RUN: %clang_cc1 %s -emit-llvm-only -triple=i386-pc-win32 -verify -DTEST2

#ifdef TEST1
struct A {
  virtual A *foo();  // in vftable slot #0.
  virtual A *bar();  // in vftable slot #1.
};

struct B : virtual A {
  // appended to the A subobject's vftable in slot #2.
  virtual B *foo(); // expected-note{{covariant thunk required by 'foo'}}
};

struct C : virtual A {
  // appended to the A subobject's vftable in slot #2.
  virtual C *bar(); // expected-note{{covariant thunk required by 'bar'}}
};

struct D : B, C { D(); }; // expected-error{{ambiguous vftable component}}
D::D() {}
#endif

#ifdef TEST2
struct A {
  virtual A *foo(); // in vftable slot #0
};

struct B : virtual A {
  // appended to the A subobject's vftable in slot #1.
  virtual B *foo(); // expected-note{{covariant thunk required by 'foo'}}
};

struct C : virtual A {
  // appended to the A subobject's vftable in slot #1.
  virtual C *foo(); // expected-note{{covariant thunk required by 'foo'}}
};

struct D : B, C { // expected-error{{ambiguous vftable component}}
  virtual D *foo();
  D();
};
D::D() {}
#endif
