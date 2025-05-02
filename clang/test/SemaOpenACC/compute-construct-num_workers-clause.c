// RUN: %clang_cc1 %s -fopenacc -verify

short getS();

void Test() {
#pragma acc parallel num_workers(1)
  while(1);
#pragma acc kernels num_workers(1)
  while(1);

  // expected-error@+2{{OpenACC 'num_workers' clause cannot appear more than once on a 'kernels' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels num_workers(1) num_workers(2)
  while(1);

  // expected-error@+2{{OpenACC 'num_workers' clause cannot appear more than once on a 'parallel' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel num_workers(1) num_workers(2)
  while(1);

  // expected-error@+3{{OpenACC 'num_workers' clause cannot appear more than once in a 'device_type' region on a 'kernels' directive}}
  // expected-note@+2{{previous clause is here}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels num_workers(1) device_type(*) num_workers(1) num_workers(2)
  while(1);

  // expected-error@+3{{OpenACC 'num_workers' clause cannot appear more than once in a 'device_type' region on a 'parallel' directive}}
  // expected-note@+2{{previous clause is here}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel device_type(*) num_workers(1) num_workers(2)
  while(1);

#pragma acc parallel num_workers(1) device_type(*) num_workers(2)
  while(1);

  // expected-error@+1{{OpenACC 'num_workers' clause is not valid on 'serial' directive}}
#pragma acc serial num_workers(1)
  while(1);

  struct NotConvertible{} NC;
  // expected-error@+1{{OpenACC clause 'num_workers' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc parallel num_workers(NC)
  while(1);

#pragma acc kernels num_workers(getS())
  while(1);

  struct Incomplete *SomeIncomplete;

  // expected-error@+1{{OpenACC clause 'num_workers' requires expression of integer type ('struct Incomplete' invalid)}}
#pragma acc kernels num_workers(*SomeIncomplete)
  while(1);

  enum E{A} SomeE;

#pragma acc kernels num_workers(SomeE)
  while(1);

  // expected-error@+1{{OpenACC 'num_workers' clause is not valid on 'loop' directive}}
#pragma acc loop num_workers(1)
  for(int i = 5; i < 10;++i);
}

void no_dupes_since_last_device_type() {
  // expected-error@+4{{OpenACC 'num_workers' clause applies to 'device_type' 'radEon', which conflicts with previous 'num_workers' clause}}
  // expected-note@+3{{active 'device_type' clause here}}
  // expected-note@+2{{previous 'num_workers' clause is here}}
  // expected-note@+1{{which applies to 'device_type' clause here}}
#pragma acc parallel device_type(nvidia, radeon) num_workers(getS()) device_type(radEon) num_workers(getS())
  ;
  // expected-error@+4{{OpenACC 'num_workers' clause applies to 'device_type' 'nvidia', which conflicts with previous 'num_workers' clause}}
  // expected-note@+3{{active 'device_type' clause here}}
  // expected-note@+2{{previous 'num_workers' clause is here}}
  // expected-note@+1{{which applies to 'device_type' clause here}}
#pragma acc parallel device_type(nvidia) num_workers(getS()) device_type(nvidia, radeon) num_workers(getS())
  ;
}
