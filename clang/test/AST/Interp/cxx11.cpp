// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=both,expected -std=c++11 %s
// RUN: %clang_cc1 -verify=both,ref -std=c++11 %s

// expected-no-diagnostics

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
const int recurse2 = recurse1; // ref-note {{here}}
const int recurse1 = 1;
int array1[recurse1];
int array2[recurse2]; // ref-warning 2{{variable length array}} \
                      // ref-note {{initializer of 'recurse2' is not a constant expression}}
