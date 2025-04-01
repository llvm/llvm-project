// RUN: %clang_cc1 %s -fopenacc -verify

struct NoBoolConversion{};
struct BoolConversion{
  operator bool();
};

template <typename T, typename U>
void BoolExpr() {

  // expected-error@+1{{value of type 'NoBoolConversion' is not contextually convertible to 'bool'}}
#pragma acc parallel loop if (NoBoolConversion{})
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+2{{no member named 'NotValid' in 'NoBoolConversion'}}
  // expected-note@#INST{{in instantiation of function template specialization}}
#pragma acc serial loop if (T::NotValid)
  for (unsigned i = 0; i < 5; ++i);

#pragma acc kernels loop if (BoolConversion{})
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{value of type 'NoBoolConversion' is not contextually convertible to 'bool'}}
#pragma acc serial loop if (T{})
  for (unsigned i = 0; i < 5; ++i);

#pragma acc parallel loop if (U{})
  for (unsigned i = 0; i < 5; ++i);
}

void Instantiate() {
  BoolExpr<NoBoolConversion, BoolConversion>(); // #INST
}
