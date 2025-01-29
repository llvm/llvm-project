// RUN: %clang_cc1 %s -fopenacc -verify

struct NoBoolConversion{};
struct BoolConversion{
  operator bool();
};

template <typename T, typename U>
void BoolExpr() {

  // expected-error@+1{{value of type 'NoBoolConversion' is not contextually convertible to 'bool'}}
#pragma acc parallel if (NoBoolConversion{})
  while(0);

  // expected-error@+2{{no member named 'NotValid' in 'NoBoolConversion'}}
  // expected-note@#INST{{in instantiation of function template specialization}}
#pragma acc parallel if (T::NotValid)
  while(0);

#pragma acc parallel if (BoolConversion{})
  while(0);

  // expected-error@+1{{value of type 'NoBoolConversion' is not contextually convertible to 'bool'}}
#pragma acc parallel if (T{})
  while(0);

#pragma acc parallel if (U{})
  while(0);
}

void Instantiate() {
  BoolExpr<NoBoolConversion, BoolConversion>(); // #INST
}
