// RUN: %clang_cc1 -std=c++2a -verify %s

namespace N {
  struct Q {};
  template<typename> int f(Q);
  template<int> int f(Q);
  template<typename> int g(Q);
  template<int> int g(Q);

  template<int> int some_long_name(Q); // expected-note {{here}}
}
N::Q q;
int g();

int h();
template<int> int h(...);
int h(int);

// OK, these find the above functions by ADL.
int a = f<int>(q);
int b(f<int>(q));
int c(f<0>(q));
int d = g<int>(q);

int e = h<0>(q); // ok, found by unqualified lookup

void fn() {
  f<0>(q);
  int f;
  f<0>(q); // expected-error {{invalid operands to binary expression}} // expected-error {{chained comparison 'X < Y > Z' does not behave the same as a mathematical expression}}
}

void disambig() {
  // FIXME: It's unclear whether ending the template argument at the > inside the ?: is correct here (see DR579).
  f<true ? 1 > 2 : 3>(q); // expected-error {{expected ':'}} expected-note {{to match}} expected-error {{expected expression}}

  f < 1 + 3 > (q); // ok, function call
}

bool typo(int something) { // expected-note 4{{declared here}}
  // FIXME: We shouldn't suggest the N:: for an ADL call if the candidate can be found by ADL.
  some_logn_name<3>(q); // expected-error {{did you mean 'N::some_long_name'?}}
  somethign < 3 ? h() > 4 : h(0); // expected-error {{did you mean 'something'}}
  // This is parsed as a comparison on the left of a ?: expression.
  somethign < 3 ? h() + 4 : h(0); // expected-error {{did you mean 'something'}}
  // This is parsed as an ADL-only template-id call.
  somethign < 3 ? h() + 4 : h(0) >(0); // expected-error {{undeclared identifier 'somethign'}}
  bool k(somethign < 3); // expected-error {{did you mean 'something'}}
  return somethign < 3; // expected-error {{did you mean 'something'}}
}

// Ensure that treating undeclared identifiers as template names doesn't cause
// problems.
struct W<int> {}; // expected-error {{undeclared template struct 'W'}}
X<int>::Y xy; // expected-error {{no template named 'X'}}
void xf(X<int> x); // expected-error {{no template named 'X'}}
struct A : X<int> { // expected-error {{no template named 'X'}}
  A() : X<int>() {} // expected-error {{no template named 'X'}}
};

// Similarly for treating overload sets of functions as template names.
struct g<int> {}; // expected-error {{'g' refers to a function template}}
g<int>::Y xy; // expected-error {{no template named 'g'}} FIXME lies
void xf(g<int> x); // expected-error {{variable has incomplete type 'void'}} expected-error 1+{{}}
struct B : g<int> { // expected-error {{expected class name}}
  B() : g<int>() {} // expected-error {{expected class member or base class name}}
};

namespace vector_components {
  typedef __attribute__((__ext_vector_type__(2))) float vector_float2;
  bool foo123(vector_float2 &A, vector_float2 &B)
  {
    return A.x < B.x && B.y > A.y;
  }
}
