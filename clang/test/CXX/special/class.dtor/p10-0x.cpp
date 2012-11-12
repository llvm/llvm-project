// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// PR10127/N3031
struct A { ~A(); };
struct B {};
template<typename T>
void b(const T *x, const A *y) {
  x->~decltype(T())();
  x->~decltype(*x)(); // expected-error{{the type of object expression ('const int') does not match the type being destroyed ('decltype(*x)' (aka 'const int &')) in pseudo-destructor expression}} \
                         expected-error{{no member named '~const struct A &' in 'A'}}
  x->~decltype(int())(); // expected-error{{no member named '~int' in 'A'}}

  y->~decltype(*y)(); // expected-error{{destructor type 'decltype(*y)' (aka 'const A &') in object destruction expression does not match the type 'const A' of the object being destroyed}}
  y->~decltype(T())(); // expected-error{{destructor type 'decltype(T())' in object destruction expression does not match the type 'const A' of the object being destroyed}}
  y->~decltype(A())();
}
template void b(const int*, const A*); // expected-note{{in instantiation of function template specialization 'b<int>' requested here}}
template void b(const A*,const A*); // expected-note{{in instantiation of function template specialization 'b<A>' requested here}}
void a(const A *x, int i, int *pi) {
  x->~decltype(A())();
  x->~decltype(*x)(); // expected-error{{destructor type 'decltype(*x)' (aka 'const A &') in object destruction expression does not match the type 'const A' of the object being destroyed}}
  x->~decltype()(); // expected-error{{expected expression}}
  x->~decltype(B())(); // expected-error{{destructor type 'decltype(B())' (aka 'B') in object destruction expression does not match the type 'const A' of the object being destroyed}}
  x->~decltype(x)(); // expected-error{{destructor type 'decltype(x)' (aka 'const A *') in object destruction expression does not match the type 'const A' of the object being destroyed}}
  // this last one could be better, mentioning that the nested-name-specifier could be removed or a type name after the ~
  x->::A::~decltype(*x)(); // expected-error{{expected a class name after '~' to name a destructor}}
  y->~decltype(A())(); // expected-error{{use of undeclared identifier 'y'}}

  typedef int *intp;
  i->~decltype(int())(); // expected-error{{member reference type 'int' is not a pointer; maybe you meant to use '.'?}}
  i.~decltype(int())();
  i->~decltype(intp())(); // expected-error{{member reference type 'int' is not a pointer; maybe you meant to use '.'?}} \
                             expected-error{{the type of object expression ('int') does not match the type being destroyed ('decltype(intp())' (aka 'int *')) in pseudo-destructor expression}}
  i.~decltype(intp())(); // expected-error{{the type of object expression ('int') does not match the type being destroyed ('decltype(intp())' (aka 'int *')) in pseudo-destructor expression}}
  pi->~decltype(int())();
  pi.~decltype(int())(); // expected-error{{the type of object expression ('int *') does not match the type being destroyed ('decltype(int())' (aka 'int')) in pseudo-destructor expression}}
  pi.~decltype(intp())();
  pi->~decltype(intp())(); // expected-error{{the type of object expression ('int') does not match the type being destroyed ('decltype(intp())' (aka 'int *')) in pseudo-destructor expression}}
}
