// RUN: %clang_cc1 -std=c++2a -verify %s

struct A {};
struct B { bool operator==(B) const; };
struct C { int operator==(C) const; };
struct D {
  // expected-note@+2 {{candidate function not viable: 'this' argument has type 'const}}
  // expected-note@+1 {{candidate function (with reversed parameter order) not viable: 1st argument ('const}}
  bool operator==(D);
};
struct E {
  E(const E &) = delete; // expected-note {{deleted}}
  int operator==(E) const; // expected-note {{passing}}
};
struct F { void operator==(F) const; };
struct G { bool operator==(G) const = delete; }; // expected-note {{deleted here}}

struct H1 {
  bool operator==(const H1 &) const = default;
  bool operator<(const H1 &) const = default; // expected-warning {{implicitly deleted}} expected-note{{replace 'default'}}
  // expected-note@-1 {{because there is no viable three-way comparison function for 'H1'}}
  void (*x)();
};
struct H2 {
  bool operator==(const H2 &) const = default;
  bool operator<(const H2 &) const = default; // expected-warning {{implicitly deleted}} expected-note{{replace 'default'}}
  // expected-note@-1 {{because there is no viable three-way comparison function for 'H2'}}
  void (H2::*x)();
};
struct H3 {
  bool operator==(const H3 &) const = default;
  bool operator<(const H3 &) const = default; // expected-warning {{implicitly deleted}} expected-note{{replace 'default'}}
  // expected-note@-1 {{because there is no viable three-way comparison function for 'H3'}}
  int H3::*x;
};

template<typename T> struct X {
  X();
  bool operator==(const X&) const = default; // #x expected-note 4{{deleted here}}
  T t; // expected-note 3{{because there is no viable 'operator==' for member 't'}}
       // expected-note@-1 {{because it would invoke a deleted comparison function for member 't'}}
};

struct Mutable {
  bool operator==(const Mutable&) const = default;
  mutable D d;
};

void test() {
  void(X<A>() == X<A>()); // expected-error {{cannot be compared because its 'operator==' is implicitly deleted}}
  void(X<B>() == X<B>());
  void(X<C>() == X<C>());
  void(X<D>() == X<D>()); // expected-error {{cannot be compared because its 'operator==' is implicitly deleted}}
  void(Mutable() == Mutable());

  // FIXME: We would benefit from a note identifying the member of 'X' we were comparing here and below.
  // expected-error@#x {{call to deleted constructor of 'E'}}
  void(X<E>() == X<E>()); // expected-note {{in defaulted equality comparison operator for 'X<E>' first required here}}

  // FIXME: We would benefit from a note pointing at the selected 'operator==' here.
  // expected-error@#x {{value of type 'void' is not contextually convertible to 'bool'}}
  void(X<F>() == X<F>()); // expected-note {{in defaulted equality comparison operator for 'X<F>' first required here}}

  void(X<G>() == X<G>()); // expected-error {{cannot be compared because its 'operator==' is implicitly deleted}}

  void(X<A[3]>() == X<A[3]>()); // expected-error {{cannot be compared because its 'operator==' is implicitly deleted}}
  void(X<B[3]>() == X<B[3]>());
}

namespace Access {
  class A {
    bool operator==(const A &) const; // expected-note 2{{implicitly declared private here}}
  };
  struct B : A { // expected-note 2{{because it would invoke a private 'operator==' to compare base class 'A'}}
    bool operator==(const B &) const = default; // expected-warning {{deleted}} expected-note{{replace 'default'}}
    friend bool operator==(const B &, const B &) = default; // expected-warning {{deleted}} expected-note{{replace 'default'}}
  };

  class C {
  protected:
    bool operator==(const C &) const; // expected-note 2{{declared protected here}}
  };
  struct D : C {
    bool operator==(const D &) const = default;
    friend bool operator==(const D &, const D&) = default;
  };
  struct E {
    C c; // expected-note 2{{because it would invoke a protected 'operator==' member of 'Access::C' to compare member 'c'}}
    bool operator==(const E &) const = default; // expected-warning {{deleted}} expected-note{{replace 'default'}}
    friend bool operator==(const E &, const E &) = default; // expected-warning {{deleted}} expected-note{{replace 'default'}}
  };

  struct F : C {
    using C::operator==;
  };
  struct G : F {
    bool operator==(const G&) const = default;
    friend bool operator==(const G&, const G&) = default;
  };

  struct H : C {
  private:
    using C::operator==; // expected-note 2{{declared private here}}
  };
  struct I : H { // expected-note 2{{private 'operator==' to compare base class 'H'}}
    bool operator==(const I&) const = default; // expected-warning {{deleted}} expected-note{{replace 'default'}}
    friend bool operator==(const I&, const I&) = default; // expected-warning {{deleted}} expected-note{{replace 'default'}}
  };

  class J {
    bool operator==(const J&) const;
    friend class K;
  };
  class K {
    J j;
    bool operator==(const K&) const = default;
    friend bool operator==(const K&, const K&) = default;
  };

  struct X {
    bool operator==(const X&) const; // expected-note {{ambiguity is between a regular call to this operator and a call with the argument order reversed}}
  };
  struct Y : private X { // expected-note {{private}}
    using X::operator==;
  };
  struct Z : Y {
    // Note: this function is not deleted. The selected operator== is
    // accessible. But the derived-to-base conversion involves an inaccessible
    // base class, which we don't check for until we define the function.
    bool operator==(const Z&) const = default; // expected-error {{cannot cast 'const Y' to its private base class 'const X'}} expected-warning {{ambiguous}}
  };
  bool z = Z() == Z(); // expected-note {{first required here}}
}
