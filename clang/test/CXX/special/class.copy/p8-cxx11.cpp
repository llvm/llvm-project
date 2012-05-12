// RUN: %clang_cc1 -std=c++11 %s -verify

// C++98 [class.copy]p5 / C++11 [class.copy]p8.

// The implicitly-declared copy constructor for a class X will have the form
//   X::X(const X&)
// if [every direct subobject] has a copy constructor whose first parameter is
// of type 'const volatile[opt] T &'. Otherwise, it will have the form
//   X::X(X&)

struct ConstCopy {
  ConstCopy(const ConstCopy &);
};

struct NonConstCopy {
  NonConstCopy(NonConstCopy &);
};

struct DeletedConstCopy {
  DeletedConstCopy(const DeletedConstCopy &) = delete;
};

struct DeletedNonConstCopy {
  DeletedNonConstCopy(DeletedNonConstCopy &) = delete;
};

struct ImplicitlyDeletedConstCopy {
  ImplicitlyDeletedConstCopy(ImplicitlyDeletedConstCopy &&);
};


struct A : ConstCopy {};
struct B : NonConstCopy { ConstCopy a; };
struct C : ConstCopy { NonConstCopy a; };
struct D : DeletedConstCopy {};
struct E : DeletedNonConstCopy {};
struct F { ImplicitlyDeletedConstCopy a; };
struct G : virtual B {};

struct Test {
  friend A::A(const A &);
  friend B::B(B &);
  friend C::C(C &);
  friend D::D(const D &);
  friend E::E(E &);
  friend F::F(const F &);
  friend G::G(G &);
};
