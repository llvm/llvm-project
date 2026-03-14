// RUN: %clang_cc1 -std=c++11 %s -verify
// expected-no-diagnostics

// C++98 [class.copy]p10 / C++11 [class.copy]p18.

// The implicitly-declared copy assignment operator for a class X will have the form
//   X& X::operator=(const X&)
// if [every direct subobject] has a copy assignment operator whose first parameter is
// of type 'const volatile[opt] T &' or 'T'. Otherwise, it will have the form
//   X &X::operator=(X&)

struct ConstCopy {
  ConstCopy &operator=(const ConstCopy &);
};

struct NonConstCopy {
  NonConstCopy &operator=(NonConstCopy &);
};

struct DeletedConstCopy {
  DeletedConstCopy &operator=(const DeletedConstCopy &) = delete;
};

struct DeletedNonConstCopy {
  DeletedNonConstCopy &operator=(DeletedNonConstCopy &) = delete;
};

struct ImplicitlyDeletedConstCopy {
  ImplicitlyDeletedConstCopy &operator=(ImplicitlyDeletedConstCopy &&);
};

struct ByValueCopy {
  ByValueCopy &operator=(ByValueCopy);
};

struct AmbiguousConstCopy {
  AmbiguousConstCopy &operator=(const AmbiguousConstCopy&);
  AmbiguousConstCopy &operator=(AmbiguousConstCopy);
};


struct A : ConstCopy {};
struct B : NonConstCopy { ConstCopy a; };
struct C : ConstCopy { NonConstCopy a; };
struct D : DeletedConstCopy {};
struct E : DeletedNonConstCopy {};
struct F { ImplicitlyDeletedConstCopy a; };
struct G : virtual B {};
struct H : ByValueCopy {};
struct I : AmbiguousConstCopy {};

struct Test {
  friend A &A::operator=(const A &);
  friend B &B::operator=(B &);
  friend C &C::operator=(C &);
  friend D &D::operator=(const D &);
  friend E &E::operator=(E &);
  friend F &F::operator=(const F &);
  friend G &G::operator=(G &);
  friend H &H::operator=(const H &);
  friend I &I::operator=(const I &);
};
