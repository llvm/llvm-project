// RUN: %clang_cc1 %s -fopenacc -verify

struct NoBoolConversion{};
struct BoolConversion{
  operator bool();
};

template <typename T, typename U>
void BoolExpr() {
  // expected-error@+1{{value of type 'NoBoolConversion' is not contextually convertible to 'bool'}}
#pragma acc parallel loop self (NoBoolConversion{})
  for (unsigned i = 0; i < 5; ++i);
  // expected-error@+2{{no member named 'NotValid' in 'NoBoolConversion'}}
  // expected-note@#INST{{in instantiation of function template specialization}}
#pragma acc serial loop self (T::NotValid)
  for (unsigned i = 0; i < 5; ++i);

#pragma acc kernels loop self (BoolConversion{})
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{value of type 'NoBoolConversion' is not contextually convertible to 'bool'}}
#pragma acc parallel loop self (T{})
  for (unsigned i = 0; i < 5; ++i);

#pragma acc parallel loop self (U{})
  for (unsigned i = 0; i < 5; ++i);
}

struct HasBool {
  static constexpr bool B = true;
};

template<typename T>
void WarnMaybeNotUsed() {
  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop self if(T::B)
  for (unsigned i = 0; i < 5; ++i);

  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop self(T::B) if(T::B)
  for (unsigned i = 0; i < 5; ++i);

  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc serial loop if(T::B) self
  for (unsigned i = 0; i < 5; ++i);

  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop if(T::B) self(T::B)
  for (unsigned i = 0; i < 5; ++i);

  // We still warn in the cases of dependent failures, since the diagnostic
  // happens immediately rather than during instantiation.

  // expected-error@+4{{no member named 'Invalid' in 'HasBool'}}
  // expected-note@#NOT_USED_INST{{in instantiation of function template specialization 'WarnMaybeNotUsed<HasBool>' requested here}}
  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop self if(T::Invalid)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+3{{no member named 'Invalid' in 'HasBool'}}
  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc serial loop self(T::Invalid) if(T::B)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+3{{no member named 'Invalid' in 'HasBool'}}
  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop self(T::B) if(T::Invalid)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+3{{no member named 'Invalid' in 'HasBool'}}
  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop if(T::Invalid) self
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+3{{no member named 'Invalid' in 'HasBool'}}
  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop if(T::Invalid) self(T::B)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+3{{no member named 'Invalid' in 'HasBool'}}
  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop if(T::B) self(T::Invalid)
  for (unsigned i = 0; i < 5; ++i);
}

void Instantiate() {
  BoolExpr<NoBoolConversion, BoolConversion>(); // #INST
  WarnMaybeNotUsed<HasBool>(); // #NOT_USED_INST
}
