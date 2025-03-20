// RUN: %clang_cc1 %s -fopenacc -verify

struct NoBoolConversion{};
struct BoolConversion{
  operator bool();
};

template <typename T, typename U>
void BoolExpr() {
  // expected-error@+1{{value of type 'NoBoolConversion' is not contextually convertible to 'bool'}}
#pragma acc parallel self (NoBoolConversion{})
  while(0);
  // expected-error@+2{{no member named 'NotValid' in 'NoBoolConversion'}}
  // expected-note@#INST{{in instantiation of function template specialization}}
#pragma acc parallel self (T::NotValid)
  while(0);

#pragma acc parallel self (BoolConversion{})
  while(0);

  // expected-error@+1{{value of type 'NoBoolConversion' is not contextually convertible to 'bool'}}
#pragma acc parallel self (T{})
  while(0);

#pragma acc parallel self (U{})
  while(0);
}

struct HasBool {
  static constexpr bool B = true;
};

template<typename T>
void WarnMaybeNotUsed() {
  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel self if(T::B)
  while(0);

  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel self(T::B) if(T::B)
  while(0);

  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel if(T::B) self
  while(0);

  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel if(T::B) self(T::B)
  while(0);

  // We still warn in the cases of dependent failures, since the diagnostic
  // happens immediately rather than during instantiation.

  // expected-error@+4{{no member named 'Invalid' in 'HasBool'}}
  // expected-note@#NOT_USED_INST{{in instantiation of function template specialization 'WarnMaybeNotUsed<HasBool>' requested here}}
  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel self if(T::Invalid)
  while(0);

  // expected-error@+3{{no member named 'Invalid' in 'HasBool'}}
  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel self(T::Invalid) if(T::B)
  while(0);

  // expected-error@+3{{no member named 'Invalid' in 'HasBool'}}
  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel self(T::B) if(T::Invalid)
  while(0);

  // expected-error@+3{{no member named 'Invalid' in 'HasBool'}}
  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel if(T::Invalid) self
  while(0);

  // expected-error@+3{{no member named 'Invalid' in 'HasBool'}}
  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel if(T::Invalid) self(T::B)
  while(0);

  // expected-error@+3{{no member named 'Invalid' in 'HasBool'}}
  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel if(T::B) self(T::Invalid)
  while(0);
}

void Instantiate() {
  BoolExpr<NoBoolConversion, BoolConversion>(); // #INST
  WarnMaybeNotUsed<HasBool>(); // #NOT_USED_INST
}
