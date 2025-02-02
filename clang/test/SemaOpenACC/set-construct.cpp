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
#pragma acc set default_async(getI())
#pragma acc set device_num(getI())
#pragma acc set device_type(getI)
#pragma acc set device_type(getI) if (getI() < getS())

  // expected-error@+1{{value of type 'struct NotConvertible' is not contextually convertible to 'bool'}}
#pragma acc set if (NC) device_type(I)

  // expected-error@+2{{OpenACC 'set' construct must have at least one 'default_async', 'device_num', 'device_type' or 'if' clause}}
  // expected-error@+1{{OpenACC clause 'device_num' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc set device_num(NC)
  // expected-error@+4{{OpenACC 'set' construct must have at least one 'default_async', 'device_num', 'device_type' or 'if' clause}}
  // expected-error@+3{{multiple conversions from expression type 'struct AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc set device_num(Ambiguous)
  // expected-error@+2{{OpenACC integer expression requires explicit conversion from 'struct ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
#pragma acc set device_num(Explicit)

  // expected-error@+2{{OpenACC clause 'default_async' requires expression of integer type ('struct NotConvertible' invalid)}}
  // expected-error@+1{{OpenACC 'set' construct must have at least one 'default_async', 'device_num', 'device_type' or 'if' clause}}
#pragma acc set default_async(NC)
  // expected-error@+4{{multiple conversions from expression type 'struct AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
  // expected-error@+1{{OpenACC 'set' construct must have at least one 'default_async', 'device_num', 'device_type' or 'if' clause}}
#pragma acc set default_async(Ambiguous)
  // expected-error@+2{{OpenACC integer expression requires explicit conversion from 'struct ExplicitConvertOnly' to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
#pragma acc set default_async(Explicit)

  // expected-error@+1{{OpenACC 'set' construct must have at least one 'default_async', 'device_num', 'device_type' or 'if' clause}}
#pragma acc set

#pragma acc set if (true)

  // expected-error@+2{{'default_async' clause cannot appear more than once on a 'set' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc set default_async(getI()) default_async(getI())

  // expected-error@+2{{'device_num' clause cannot appear more than once on a 'set' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc set device_num(getI()) device_num(getI())

  // expected-error@+2{{'device_type' clause cannot appear more than once on a 'set' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc set device_type(I) device_type(I)
  // expected-error@+2{{'if' clause cannot appear more than once on a 'set' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc set device_type(I) if(true) if (true)
}
