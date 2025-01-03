// RUN: %clang_cc1 %s -fopenacc -verify

struct ExplicitConvertOnly {
  explicit operator int() const; // #EXPL_CONV
} Explicit;

struct AmbiguousConvert{
  operator int(); // #AMBIG_INT
  operator short(); // #AMBIG_SHORT
  operator float();
} Ambiguous;

void Test() {

  // expected-error@+3{{multiple conversions from expression type 'struct AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc parallel wait(Ambiguous)
  while (true);

  // expected-error@+2{{OpenACC integer expression requires explicit conversion from 'struct ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
#pragma acc parallel wait(4, Explicit, 5)
  while (true);

  // expected-error@+3{{multiple conversions from expression type 'struct AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc parallel wait(queues: Ambiguous, 5)
  while (true);

  // expected-error@+2{{OpenACC integer expression requires explicit conversion from 'struct ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
#pragma acc parallel wait(devnum: Explicit: 5)
  while (true);

  // expected-error@+2{{OpenACC integer expression requires explicit conversion from 'struct ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
#pragma acc parallel wait(devnum: Explicit:queues:  5)
  while (true);

  // expected-error@+1{{use of undeclared identifier 'queues'}}
#pragma acc parallel wait(devnum: queues:  5)
  while (true);
}

struct HasInt {
  using IntTy = int;
  using ShortTy = short;
  static constexpr int value = 1;
  static constexpr AmbiguousConvert ACValue;
  static constexpr ExplicitConvertOnly EXValue;

  operator char();
};

template<typename T>
void TestInst() {

#pragma acc parallel wait(T{})
  while (true);

#pragma acc parallel wait(devnum:typename T::ShortTy{}:queues:typename T::IntTy{})
  while (true);

  // expected-error@+4{{multiple conversions from expression type 'const AmbiguousConvert' to an integral type}}
  // expected-note@#INST{{in instantiation of function template specialization}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc parallel wait(devnum:T::value :queues:T::ACValue)
  while (true);

  // expected-error@+5{{OpenACC integer expression requires explicit conversion from 'const ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
  // expected-error@+3{{multiple conversions from expression type 'const AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc parallel wait(devnum:T::EXValue :queues:T::ACValue)
  while (true);

  // expected-error@+5{{OpenACC integer expression requires explicit conversion from 'const ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
  // expected-error@+3{{multiple conversions from expression type 'const AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc parallel wait(T::EXValue, T::ACValue)
  while (true);

  // expected-error@+5{{OpenACC integer expression requires explicit conversion from 'const ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
  // expected-error@+3{{multiple conversions from expression type 'const AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc parallel wait(queues: T::EXValue, T::ACValue)
  while (true);

  // expected-error@+1{{no member named 'Invalid' in 'HasInt'}}
#pragma acc parallel wait(queues: T::Invalid, T::Invalid2)
  while (true);
}

void Inst() {
  TestInst<HasInt>(); // #INST
}
