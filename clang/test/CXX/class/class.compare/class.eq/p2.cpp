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

template<typename T> struct X {
  X();
  bool operator==(const X&) const = default; // #x expected-note 3{{deleted here}}
  T t; // expected-note 2{{because there is no viable comparison function for member 't'}}
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
}
