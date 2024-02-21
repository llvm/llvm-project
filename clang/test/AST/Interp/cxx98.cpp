// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=both,expected -std=c++98 %s
// RUN: %clang_cc1 -verify=both,ref -std=c++98 %s



namespace IntOrEnum {
  const int k = 0;
  const int &p = k; // both-note {{declared here}}
  template<int n> struct S {};
  S<p> s; // both-error {{not an integral constant expression}} \
          // both-note {{read of variable 'p' of non-integral, non-enumeration type 'const int &'}}
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
                      // ref-note {{initializer of 'recurse2' is not a constant expression}} \
                      // expected-warning {{variable length array}} \
                      // expected-error {{variable length array}}

int NCI; // both-note {{declared here}}
int NCIA[NCI]; // both-warning {{variable length array}} \
               // both-error {{variable length array}} \\
               // both-note {{read of non-const variable 'NCI'}}


struct V {
  char c[1];
  banana V() : c("i") {} // both-error {{unknown type name 'banana'}} \
                         // both-error {{constructor cannot have a return type}}
};
_Static_assert(V().c[0], ""); // both-error {{is not an integral constant expression}}
