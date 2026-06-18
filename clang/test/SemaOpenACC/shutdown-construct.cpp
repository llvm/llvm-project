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
#pragma acc shutdown
#pragma acc shutdown if (getI() < getS())
#pragma acc shutdown device_num(getI())
#pragma acc shutdown device_type(host) device_num(getI())
#pragma acc shutdown device_type(nvidia) if (getI() < getS())
#pragma acc shutdown device_type(radeon) device_num(getI()) if (getI() < getS())

  // expected-error@+1{{value of type 'struct NotConvertible' is not contextually convertible to 'bool'}}
#pragma acc shutdown if (NC)

  // expected-error@+1{{OpenACC clause 'device_num' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc shutdown device_num(NC)
  // expected-error@+3{{multiple conversions from expression type 'struct AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc shutdown device_num(Ambiguous)
  // expected-error@+2{{OpenACC integer expression requires explicit conversion from 'struct ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
#pragma acc shutdown device_num(Explicit)

  // expected-error@+1{{OpenACC 'device_type' clause on a 'shutdown' construct only permits one architecture}}
#pragma acc shutdown device_type(nvidia, radeon)

  // expected-error@+1{{OpenACC 'device_type' clause on a 'shutdown' construct only permits one architecture}}
#pragma acc shutdown device_type(nonsense, nvidia, radeon)
}

template<typename T>
void TestInst() {
  T t;
#pragma acc shutdown
#pragma acc shutdown if (T::value < T{})
#pragma acc shutdown device_type(multicore) device_num(getI()) if (getI() < getS())
  // expected-error@+2{{OpenACC 'device_num' clause cannot appear more than once on a 'shutdown' directive}}
  // expected-note@+1{{previous 'device_num' clause is here}}
#pragma acc shutdown device_type(default) device_type(radeon) device_num(t) if (t < T::value) device_num(getI()) 

  // expected-error@+1{{value of type 'const NotConvertible' is not contextually convertible to 'bool'}}
#pragma acc shutdown if (T::NCValue)

  // expected-error@+2{{OpenACC 'if' clause cannot appear more than once on a 'shutdown' directive}}
  // expected-note@+1{{previous 'if' clause is here}}
#pragma acc shutdown if(t < T::value) if (getI() < getS())

  // expected-error@+1{{OpenACC clause 'device_num' requires expression of integer type ('const NotConvertible' invalid)}}
#pragma acc shutdown device_num(T::NCValue)
  // expected-error@+3{{multiple conversions from expression type 'const AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc shutdown device_num(T::ACValue)
  // expected-error@+2{{OpenACC integer expression requires explicit conversion from 'const ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
#pragma acc shutdown device_num(T::EXValue)
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
