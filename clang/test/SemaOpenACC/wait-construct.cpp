// RUN: %clang_cc1 %s -fopenacc -verify

struct NotConvertible{} NC;
short getS();
int getI();

struct AmbiguousConvert{
  operator int(); // #AMBIG_INT
  operator short(); // #AMBIG_SHORT
  operator float();
} Ambiguous;

struct ExplicitConvertOnly {
  explicit operator int() const; // #EXPL_CONV
} Explicit;

void uses() {
  int arr[5];
#pragma acc wait(getS(), getI())
#pragma acc wait(devnum:getS(): getI())
#pragma acc wait(devnum:getS(): queues: getI(), getS())
#pragma acc wait(devnum:getS(): getI(), getS())

  // expected-error@+1{{OpenACC directive 'wait' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc wait(devnum:NC : 5)
  // expected-error@+1{{OpenACC directive 'wait' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc wait(devnum:5 : NC)
  // expected-error@+3{{OpenACC directive 'wait' requires expression of integer type ('int[5]' invalid)}}
  // expected-error@+2{{OpenACC directive 'wait' requires expression of integer type ('int[5]' invalid)}}
  // expected-error@+1{{OpenACC directive 'wait' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc wait(devnum:arr : queues: arr, NC, 5)

  // expected-error@+3{{multiple conversions from expression type 'struct AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc wait(Ambiguous)

  // expected-error@+2{{OpenACC integer expression requires explicit conversion from 'struct ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
#pragma acc wait(4, Explicit, 5)

  // expected-error@+1{{use of undeclared identifier 'queues'}}
#pragma acc wait(devnum: queues:  5)

#pragma acc wait async
#pragma acc wait async(getI())
  // expected-error@+1{{OpenACC clause 'async' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc wait async(NC)

#pragma acc wait if(getI() < getS())
  // expected-error@+1{{value of type 'struct NotConvertible' is not contextually convertible to 'bool'}}
#pragma acc wait if(NC)

}

template<typename T>
void TestInst() {
  // expected-error@+4{{multiple conversions from expression type 'const AmbiguousConvert' to an integral type}}
  // expected-note@#INST{{in instantiation of function template specialization}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc wait(devnum:T::value :queues:T::ACValue)

  // expected-error@+5{{OpenACC integer expression requires explicit conversion from 'const ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
  // expected-error@+3{{multiple conversions from expression type 'const AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc wait(devnum:T::EXValue :queues:T::ACValue)

  // expected-error@+1{{no member named 'Invalid' in 'HasInt'}}
#pragma acc wait(queues: T::Invalid, T::Invalid2)

  // expected-error@+3{{multiple conversions from expression type 'const AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc wait async(T::ACValue)

#pragma acc wait if(T::value < T{})
  // expected-error@+1{{value of type 'const ExplicitConvertOnly' is not contextually convertible to 'bool'}}
#pragma acc wait if(T::EXValue)
}

struct HasInt {
  using IntTy = int;
  using ShortTy = short;
  static constexpr int value = 1;
  static constexpr AmbiguousConvert ACValue;
  static constexpr ExplicitConvertOnly EXValue;

  operator char();
};
void Inst() {
  TestInst<HasInt>(); // #INST
}
