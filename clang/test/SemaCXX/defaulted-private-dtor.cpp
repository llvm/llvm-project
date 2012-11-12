// RUN: %clang_cc1 -verify -std=c++11 %s -fcxx-exceptions

class BadDtor {
  // A private, but nonetheless trivial, destructor.
  ~BadDtor() = default; // expected-note 9{{here}}
  friend class K;
};
void f() {
  BadDtor *p = new BadDtor[3]; // expected-error {{private destructor}}
  delete [] p; // expected-error {{private destructor}}
  const BadDtor &dd2 = BadDtor(); // expected-error {{private destructor}}
  BadDtor dd; // expected-error {{private destructor}}
  throw dd; // expected-error {{private destructor}}
}
struct V {
  V();
  BadDtor bd; // expected-note {{inaccessible destructor}}
};
V v; // expected-error {{deleted function}}
struct W : BadDtor { // expected-note {{inaccessible destructor}}
  W();
};
W w; // expected-error {{deleted function}}
struct X : BadDtor { // expected-error {{private destructor}}
  ~X() {}
};
struct Y {
  BadDtor dd; // expected-error {{private destructor}}
  ~Y() {}
};
struct Z : virtual BadDtor { // expected-error {{private destructor}}
  ~Z() {}
};
BadDtor dd; // expected-error {{private destructor}}

class K : BadDtor {
  void f() {
    BadDtor *p = new BadDtor[3];
    delete [] p;
    const BadDtor &dd2 = BadDtor();
    BadDtor dd;
    throw dd;

    {
      BadDtor x;
      goto dont_call_dtor;
    }
dont_call_dtor:
    ;
  }
  struct Z : virtual BadDtor {
    ~Z() {}
  };
  BadDtor dd;
  ~K();
};
