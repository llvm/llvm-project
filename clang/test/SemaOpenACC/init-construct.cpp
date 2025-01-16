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
#pragma acc init
#pragma acc init if (getI() < getS())
#pragma acc init device_num(getI())
#pragma acc init device_type(SOMETHING) device_num(getI())
#pragma acc init device_type(SOMETHING) if (getI() < getS())
#pragma acc init device_type(SOMETHING) device_num(getI()) if (getI() < getS())

  // expected-error@+1{{value of type 'struct NotConvertible' is not contextually convertible to 'bool'}}
#pragma acc init if (NC)

  // expected-error@+1{{OpenACC clause 'device_num' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc init device_num(NC)
  // expected-error@+3{{multiple conversions from expression type 'struct AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc init device_num(Ambiguous)
  // expected-error@+2{{OpenACC integer expression requires explicit conversion from 'struct ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
#pragma acc init device_num(Explicit)
}

template<typename T>
void TestInst() {
  T t;
#pragma acc init
#pragma acc init if (T::value < T{})
#pragma acc init device_type(SOMETHING) device_num(getI()) if (getI() < getS())
#pragma acc init device_type(SOMETHING) device_type(T) device_num(t) if (t < T::value) device_num(getI()) if (getI() < getS())

  // expected-error@+1{{value of type 'const NotConvertible' is not contextually convertible to 'bool'}}
#pragma acc init if (T::NCValue)

  // expected-error@+1{{OpenACC clause 'device_num' requires expression of integer type ('const NotConvertible' invalid)}}
#pragma acc init device_num(T::NCValue)
  // expected-error@+3{{multiple conversions from expression type 'const AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc init device_num(T::ACValue)
  // expected-error@+2{{OpenACC integer expression requires explicit conversion from 'const ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
#pragma acc init device_num(T::EXValue)
}

struct HasStuff {
  static constexpr AmbiguousConvert ACValue;
  static constexpr ExplicitConvertOnly EXValue;
  static constexpr NotConvertible NCValue;
  static constexpr unsigned value = 5;
  operator char();
};

void Inst() {
  TestInst<HasStuff>(); // expected-note {{in instantiation of function template specialization}}
}
