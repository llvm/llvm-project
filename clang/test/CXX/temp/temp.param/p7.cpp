// RUN: %clang_cc1 -fsyntax-only -verify %s

// A non-type template-parameter shall not be declared to have
// floating point, class, or void type.
struct A;

template<double d> class X; // expected-error{{cannot have type}}
template<double* pd> class Y; //OK 
template<double& rd> class Z; //OK 

template<A a> class X0; // expected-error{{cannot have type}}

typedef void VOID;
template<VOID a> class X01; // expected-error{{cannot have type}}

