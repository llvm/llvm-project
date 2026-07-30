// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR10283
void min(); //expected-note {{'min' declared here}}
void min(int);

template <typename T> void max(T); //expected-note {{'max' declared here}}

void f() {
  fin(); //expected-error {{use of undeclared identifier 'fin'; did you mean 'min'}}
  fax(0); //expected-error {{use of undeclared identifier 'fax'; did you mean 'max'}}
}

template <typename T> void somefunc(T*, T*); //expected-note {{'somefunc' declared here}}
template <typename T> void somefunc(const T[]); //expected-note {{'somefunc' declared here}}
template <typename T1, typename T2> void somefunc(T1*, T2*); //expected-note {{'somefunc' declared here}}
template <typename T1, typename T2> void somefunc(T1*, const T2[]); //expected-note 2 {{'somefunc' declared here}}

void c() {
  int *i = 0, *j = 0;
  const int x[] = {1, 2, 3};
  long *l = 0;
  somefun(i, j); //expected-error {{use of undeclared identifier 'somefun'; did you mean 'somefunc'?}}
  somefun(x); //expected-error {{use of undeclared identifier 'somefun'; did you mean 'somefunc'?}}
  somefun(i, l); //expected-error {{use of undeclared identifier 'somefun'; did you mean 'somefunc'?}}
  somefun(l, x); //expected-error {{use of undeclared identifier 'somefun'; did you mean 'somefunc'?}}
  somefun(i, x); //expected-error {{use of undeclared identifier 'somefun'; did you mean 'somefunc'?}}
}
