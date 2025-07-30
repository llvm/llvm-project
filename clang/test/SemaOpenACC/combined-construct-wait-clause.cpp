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
#pragma acc parallel loop wait(Ambiguous)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC integer expression requires explicit conversion from 'struct ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
#pragma acc parallel loop wait(4, Explicit, 5)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+3{{multiple conversions from expression type 'struct AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc parallel loop wait(queues: Ambiguous, 5)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC integer expression requires explicit conversion from 'struct ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
#pragma acc parallel loop wait(devnum: Explicit: 5)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC integer expression requires explicit conversion from 'struct ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
#pragma acc parallel loop wait(devnum: Explicit:queues:  5)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{use of undeclared identifier 'queues'}}
#pragma acc parallel loop wait(devnum: queues:  5)
  for (unsigned i = 0; i < 5; ++i);
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

#pragma acc parallel loop wait(T{})
  for (unsigned i = 0; i < 5; ++i);

#pragma acc parallel loop wait(devnum:typename T::ShortTy{}:queues:typename T::IntTy{})
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+4{{multiple conversions from expression type 'const AmbiguousConvert' to an integral type}}
  // expected-note@#INST{{in instantiation of function template specialization}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc parallel loop wait(devnum:T::value :queues:T::ACValue)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+5{{OpenACC integer expression requires explicit conversion from 'const ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
  // expected-error@+3{{multiple conversions from expression type 'const AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc parallel loop wait(devnum:T::EXValue :queues:T::ACValue)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+5{{OpenACC integer expression requires explicit conversion from 'const ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
  // expected-error@+3{{multiple conversions from expression type 'const AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc parallel loop wait(T::EXValue, T::ACValue)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+5{{OpenACC integer expression requires explicit conversion from 'const ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
  // expected-error@+3{{multiple conversions from expression type 'const AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc parallel loop wait(queues: T::EXValue, T::ACValue)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{no member named 'Invalid' in 'HasInt'}}
#pragma acc parallel loop wait(queues: T::Invalid, T::Invalid2)
  for (unsigned i = 0; i < 5; ++i);
}

void Inst() {
  TestInst<HasInt>(); // #INST
}
