// RUN: %clang_cc1 -std=c++2a -verify %s

struct A {};
struct B { bool operator==(B) const; };
struct C { int operator==(C) const; };
struct D {
  // expected-note@+2 {{candidate function not viable: 'this' argument has type 'const}}
  // expected-note@+1 {{candidate function (with reversed parameter order) not viable: 1st argument ('const}}
  bool operator==(D);
};
struct E { E(const E&) = delete; int operator==(E) const; };
struct F { void operator==(F) const; };
struct G { bool operator==(G) const = delete; }; // expected-note {{deleted here}}

template<typename T> struct X {
  X();
  bool operator==(const X&) const = default; // expected-note 3{{deleted here}}
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

  // FIXME: Not deleted, but once we start synthesizing comparison function definitions, we should reject this.
  void(X<E>() == X<E>());
  // FIXME: Similarly, not deleted under P2002R0, but synthesized body is ill-formed.
  void(X<F>() == X<F>());

  void(X<G>() == X<G>()); // expected-error {{cannot be compared because its 'operator==' is implicitly deleted}}
}
