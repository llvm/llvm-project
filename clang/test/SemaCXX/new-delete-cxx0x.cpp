// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11 -triple=i686-pc-linux-gnu -pedantic

void ugly_news(int *ip) {
  (void)new int[-1]; // expected-error {{array size is negative}}
  (void)new int[2000000000]; // expected-error {{array is too large}}
}

void pr22845a() {
  constexpr int i = -1;
  int *p = new int[i]; // expected-error {{array size is negative}}
}

void pr22845b() {
  constexpr int i = 1;
  int *p = new int[i]{1, 2}; // expected-error {{excess elements in array initializer}}
}

struct S {
  S(int);
  S();
  ~S();
};

struct T { // expected-note 1+{{not viable}}
  T(int); // expected-note 1+{{not viable}}
};

void fn(int n) {
  (void) new int[2] {1, 2};
  (void) new S[2] {1, 2};
  (void) new S[3] {1, 2};
  (void) new S[n] {};
  // C++11 [expr.new]p19:
  //   If the new-expression creates an object or an array of objects of class
  //   type, access and ambiguity control are done for the allocation function,
  //   the deallocation function (12.5), and the constructor (12.1).
  //
  // Note that this happens even if the array bound is constant and the
  // initializer initializes every array element.
  //
  // It's not clear that this is the intended interpretation, however -- we
  // obviously don't want to check for a default constructor for 'new S(0)'.
  // Instead, we only check for a default constructor in the case of an array
  // new with a non-constant bound or insufficient initializers.
  (void) new T[2] {1, 2}; // ok
  (void) new T[3] {1, 2}; // expected-error {{no matching constructor}} expected-note {{in implicit initialization of array element 2}}
  (void) new T[n] {1, 2}; // expected-error {{no matching constructor}} expected-note {{in implicit initialization of trailing array elements in runtime-sized array new}}
  (void) new T[n] {}; // expected-error {{no matching constructor}} expected-note {{in implicit initialization of trailing array elements in runtime-sized array new}}
}

struct U {
  T t; // expected-note 3{{in implicit initialization of field 't'}}
  S s;
};
void g(int n) {
  // Aggregate initialization, brace-elision, and array new combine to create
  // this monstrosity.
  (void) new U[2] {1, 2}; // expected-error {{no matching constructor}} expected-note {{in implicit initialization of array element 1}}
  (void) new U[2] {1, 2, 3}; // ok
  (void) new U[2] {1, 2, 3, 4}; // ok
  (void) new U[2] {1, 2, 3, 4, 5}; // expected-error {{excess elements in array initializer}}

  (void) new U[n] {1, 2}; // expected-error {{no matching constructor}} expected-note {{in implicit initialization of trailing array elements}}
  (void) new U[n] {1, 2, 3}; // expected-error {{no matching constructor}} expected-note {{in implicit initialization of trailing array elements}}
}
