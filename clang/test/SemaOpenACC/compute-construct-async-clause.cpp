// RUN: %clang_cc1 %s -fopenacc -verify

struct NotConvertible{} NC;
struct Incomplete *SomeIncomplete; // #INCOMPLETE
enum E{} SomeE;
enum class E2{} SomeE2;

struct CorrectConvert {
  operator int();
} Convert;

struct ExplicitConvertOnly {
  explicit operator int() const; // #EXPL_CONV
} Explicit;

struct AmbiguousConvert{
  operator int(); // #AMBIG_INT
  operator short(); // #AMBIG_SHORT
  operator float();
} Ambiguous;

void Test() {
#pragma acc parallel async
  while(1);
#pragma acc parallel async(1)
  while(1);
#pragma acc kernels async(-51)
  while(1);

  // expected-error@+1{{OpenACC clause 'async' requires expression of integer type ('struct NotConvertible' invalid}}
#pragma acc parallel async(NC)
  while(1);

  // expected-error@+2{{OpenACC integer expression has incomplete class type 'struct Incomplete'}}
  // expected-note@#INCOMPLETE{{forward declaration of 'Incomplete'}}
#pragma acc kernels async(*SomeIncomplete)
  while(1);

#pragma acc parallel async(SomeE)
  while(1);

  // expected-error@+1{{OpenACC clause 'async' requires expression of integer type ('enum E2' invalid}}
#pragma acc kernels async(SomeE2)
  while(1);

#pragma acc parallel async(Convert)
  while(1);

  // expected-error@+2{{OpenACC integer expression requires explicit conversion from 'struct ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
#pragma acc kernels async(Explicit)
  while(1);

  // expected-error@+3{{multiple conversions from expression type 'struct AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc parallel async(Ambiguous)
  while(1);
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

  // expected-error@+1{{no member named 'Invalid' in 'HasInt'}}
#pragma acc parallel async(HasInt::Invalid)
  while (1);

  // expected-error@+2{{no member named 'Invalid' in 'HasInt'}}
  // expected-note@#INST{{in instantiation of function template specialization 'TestInst<HasInt>' requested here}}
#pragma acc kernels async(T::Invalid)
  while (1);

  // expected-error@+3{{multiple conversions from expression type 'const AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc parallel async(HasInt::ACValue)
  while (1);

  // expected-error@+3{{multiple conversions from expression type 'const AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc kernels async(T::ACValue)
  while (1);

  // expected-error@+2{{OpenACC integer expression requires explicit conversion from 'const ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
#pragma acc parallel async(HasInt::EXValue)
  while (1);

  // expected-error@+2{{OpenACC integer expression requires explicit conversion from 'const ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
#pragma acc kernels async(T::EXValue)
  while (1);

#pragma acc parallel async(HasInt::value)
  while (1);

#pragma acc kernels async(T::value)
  while (1);

#pragma acc parallel async(HasInt::IntTy{})
  while (1);

#pragma acc kernels async(typename T::ShortTy{})
  while (1);

#pragma acc parallel async(HasInt::IntTy{})
  while (1);

#pragma acc kernels async(typename T::ShortTy{})
  while (1);

  HasInt HI{};
  T MyT{};

#pragma acc parallel async(HI)
  while (1);

#pragma acc kernels async(MyT)
  while (1);
}

void Inst() {
  TestInst<HasInt>(); // #INST
}
