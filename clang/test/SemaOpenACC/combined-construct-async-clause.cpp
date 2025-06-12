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
#pragma acc parallel loop async
  for (int i = 5; i < 10; ++i);
#pragma acc parallel loop async(1)
  for (int i = 5; i < 10; ++i);
#pragma acc kernels loop async(-51)
  for (int i = 5; i < 10; ++i);
#pragma acc serial loop async(2)
  for (int i = 5; i < 10; ++i);

  // expected-error@+1{{OpenACC clause 'async' requires expression of integer type ('struct NotConvertible' invalid}}
#pragma acc parallel loop async(NC)
  for (int i = 5; i < 10; ++i);

  // expected-error@+2{{OpenACC integer expression has incomplete class type 'struct Incomplete'}}
  // expected-note@#INCOMPLETE{{forward declaration of 'Incomplete'}}
#pragma acc kernels loop async(*SomeIncomplete)
  for (int i = 5; i < 10; ++i);

#pragma acc parallel loop async(SomeE)
  for (int i = 5; i < 10; ++i);

  // expected-error@+1{{OpenACC clause 'async' requires expression of integer type ('enum E2' invalid}}
#pragma acc kernels loop async(SomeE2)
  for (int i = 5; i < 10; ++i);

#pragma acc parallel loop async(Convert)
  for (int i = 5; i < 10; ++i);

  // expected-error@+2{{OpenACC integer expression requires explicit conversion from 'struct ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
#pragma acc kernels loop async(Explicit)
  for (int i = 5; i < 10; ++i);

  // expected-error@+3{{multiple conversions from expression type 'struct AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc parallel loop async(Ambiguous)
  for (int i = 5; i < 10; ++i);
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
#pragma acc parallel loop async(HasInt::Invalid)
  for (int i = 5; i < 10; ++i);

  // expected-error@+2{{no member named 'Invalid' in 'HasInt'}}
  // expected-note@#INST{{in instantiation of function template specialization 'TestInst<HasInt>' requested here}}
#pragma acc kernels loop async(T::Invalid)
  for (int i = 5; i < 10; ++i);

  // expected-error@+3{{multiple conversions from expression type 'const AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc parallel loop async(HasInt::ACValue)
  for (int i = 5; i < 10; ++i);

  // expected-error@+3{{multiple conversions from expression type 'const AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc kernels loop async(T::ACValue)
  for (int i = 5; i < 10; ++i);

  // expected-error@+2{{OpenACC integer expression requires explicit conversion from 'const ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
#pragma acc parallel loop async(HasInt::EXValue)
  for (int i = 5; i < 10; ++i);

  // expected-error@+2{{OpenACC integer expression requires explicit conversion from 'const ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
#pragma acc kernels loop async(T::EXValue)
  for (int i = 5; i < 10; ++i);

#pragma acc parallel loop async(HasInt::value)
  for (int i = 5; i < 10; ++i);

#pragma acc kernels loop async(T::value)
  for (int i = 5; i < 10; ++i);

#pragma acc parallel loop async(HasInt::IntTy{})
  for (int i = 5; i < 10; ++i);

#pragma acc kernels loop async(typename T::ShortTy{})
  for (int i = 5; i < 10; ++i);

#pragma acc parallel loop async(HasInt::IntTy{})
  for (int i = 5; i < 10; ++i);

#pragma acc kernels loop async(typename T::ShortTy{})
  for (int i = 5; i < 10; ++i);

  HasInt HI{};
  T MyT{};

#pragma acc parallel loop async(HI)
  for (int i = 5; i < 10; ++i);

#pragma acc kernels loop async(MyT)
  for (int i = 5; i < 10; ++i);
}

void Inst() {
  TestInst<HasInt>(); // #INST
}
