// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef void F() const;

void f() const; // expected-error {{non-member function cannot have 'const' qualifier}}
F g; // expected-error {{non-member function of type 'F' (aka 'void () const') cannot have 'const' qualifier}}

struct X {
  void f() const;
  friend void g() const; // expected-error {{non-member function cannot have 'const' qualifier}}
  static void h() const; // expected-error {{static member function cannot have 'const' qualifier}}
  F i; // ok
  friend F j; // expected-error {{non-member function of type 'F' (aka 'void () const') cannot have 'const' qualifier}}
  static F k; // expected-error {{static member function of type 'F' (aka 'void () const') cannot have 'const' qualifier}}
};

struct Y {
  friend void X::f() const;
  friend void ::f() const; // expected-error {{non-member function cannot have 'const' qualifier}}
};

template<typename T> struct S {
  typedef T F;
  typedef T *P; // expected-error {{pointer to function type 'void () const' cannot have 'const' qualifier}}
  typedef T &R; // expected-error {{reference to function type 'void () const' cannot have 'const' qualifier}}
};
S<F> s; // expected-note {{in instantiation of}}

// FIXME: This is ill-formed.
template<typename T> struct U {
  void f(T);
};
U<F> u;
