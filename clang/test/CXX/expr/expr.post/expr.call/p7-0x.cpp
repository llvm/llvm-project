// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct X1 {
  X1();
};

struct X2 {
  X2();
  ~X2();
};

struct X3 {
  X3(const X3&) = default;
};

struct X4 {
  X4(const X4&) = default;
  X4(X4&);
};

void vararg(...);

void g();

void f(X1 x1, X2 x2, X3 x3, X4 x4) {
  vararg(x1); // OK
  vararg(x2); // expected-error{{cannot pass object of non-trivial type 'X2' through variadic function; call will abort at runtime}}
  vararg(x3); // OK
  vararg(x4); // expected-error{{cannot pass object of non-trivial type 'X4' through variadic function; call will abort at runtime}}

  vararg(g()); // expected-error{{cannot pass expression of type 'void' to variadic function}}
  vararg({1, 2, 3}); // expected-error{{cannot pass initializer list to variadic function}}
}


namespace PR11131 {
  struct S;

  S &getS();

  int f(...);

  void g() {
    (void)sizeof(f(getS()));
  }
}
