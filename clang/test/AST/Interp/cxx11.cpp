// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=both,expected -std=c++11 %s
// RUN: %clang_cc1 -verify=both,ref -std=c++11 %s

namespace IntOrEnum {
  const int k = 0;
  const int &p = k;
  template<int n> struct S {};
  S<p> s;
}

const int cval = 2;
template <int> struct C{};
template struct C<cval>;


/// FIXME: This example does not get properly diagnosed in the new interpreter.
extern const int recurse1;
const int recurse2 = recurse1; // both-note {{here}}
const int recurse1 = 1;
int array1[recurse1];
int array2[recurse2]; // both-warning {{variable length arrays in C++}} \
                      // both-note {{initializer of 'recurse2' is not a constant expression}} \
                      // expected-error {{variable length array declaration not allowed at file scope}} \
                      // ref-warning {{variable length array folded to constant array as an extension}}

struct S {
  int m;
};
constexpr S s = { 5 };
constexpr const int *p = &s.m + 1;

constexpr const int *np2 = &(*(int(*)[4])nullptr)[0]; // ok
