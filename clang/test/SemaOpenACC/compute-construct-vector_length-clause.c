// RUN: %clang_cc1 %s -fopenacc -verify

short getS();

void Test() {
#pragma acc parallel vector_length(1)
  while(1);
#pragma acc kernels vector_length(1)
  while(1);

  // expected-error@+2{{OpenACC 'vector_length' clause cannot appear more than once on a 'kernels' directive}}
  // expected-note@+1{{previous 'vector_length' clause is here}}
#pragma acc kernels vector_length(1) vector_length(2)
  while(1);

  // expected-error@+2{{OpenACC 'vector_length' clause cannot appear more than once on a 'parallel' directive}}
  // expected-note@+1{{previous 'vector_length' clause is here}}
#pragma acc parallel vector_length(1) vector_length(2)
  while(1);

  // expected-error@+3{{OpenACC 'vector_length' clause cannot appear more than once in a 'device_type' region on a 'kernels' directive}}
  // expected-note@+2{{previous 'vector_length' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc kernels vector_length(1) device_type(*) vector_length(1) vector_length(2)
  while(1);

  // expected-error@+3{{OpenACC 'vector_length' clause cannot appear more than once in a 'device_type' region on a 'parallel' directive}}
  // expected-note@+2{{previous 'vector_length' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc parallel device_type(*) vector_length(1) vector_length(2)
  while(1);

#pragma acc parallel vector_length(1) device_type(*) vector_length(2)
  while(1);

  // expected-error@+1{{OpenACC 'vector_length' clause is not valid on 'serial' directive}}
#pragma acc serial vector_length(1)
  while(1);

  struct NotConvertible{} NC;
  // expected-error@+1{{OpenACC clause 'vector_length' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc parallel vector_length(NC)
  while(1);

#pragma acc kernels vector_length(getS())
  while(1);

  struct Incomplete *SomeIncomplete;

  // expected-error@+1{{OpenACC clause 'vector_length' requires expression of integer type ('struct Incomplete' invalid)}}
#pragma acc kernels vector_length(*SomeIncomplete)
  while(1);

  enum E{A} SomeE;

#pragma acc kernels vector_length(SomeE)
  while(1);

  // expected-error@+1{{OpenACC 'vector_length' clause is not valid on 'loop' directive}}
#pragma acc loop vector_length(1)
  for(int i = 5; i < 10;++i);
}

void no_dupes_since_last_device_type() {
  // expected-error@+4{{OpenACC 'vector_length' clause applies to 'device_type' 'radeon', which conflicts with previous 'vector_length' clause}}
  // expected-note@+3{{active 'device_type' clause here}}
  // expected-note@+2{{previous 'vector_length' clause is here}}
  // expected-note@+1{{which applies to 'device_type' clause here}}
#pragma acc parallel device_type(nvidia, radeon) vector_length(getS()) device_type(radeon) vector_length(getS())
  ;
  // expected-error@+4{{OpenACC 'vector_length' clause applies to 'device_type' 'nvidia', which conflicts with previous 'vector_length' clause}}
  // expected-note@+3{{active 'device_type' clause here}}
  // expected-note@+2{{previous 'vector_length' clause is here}}
  // expected-note@+1{{which applies to 'device_type' clause here}}
#pragma acc parallel device_type(nvidia) vector_length(getS()) device_type(nvidia, radeon) vector_length(getS())
  ;
}
