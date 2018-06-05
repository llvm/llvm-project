// RUN: %clang_cc1 -x c++ -ffixed-point %s -verify

// Name namgling is not provided for fixed point types in c++

_Accum accum;                           // expected-error{{unknown type name '_Accum'}}
