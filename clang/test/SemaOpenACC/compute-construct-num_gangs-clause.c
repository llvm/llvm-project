// RUN: %clang_cc1 %s -fopenacc -verify

short getS();
void Test() {
#pragma acc kernels num_gangs(1)
  while(1);

  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'serial' directive}}
#pragma acc serial num_gangs(1)
  while(1);

#pragma acc parallel num_gangs(1)
  while(1);

  // expected-error@+2{{OpenACC 'num_gangs' clause cannot appear more than once on a 'kernels' directive}}
  // expected-note@+1{{previous 'num_gangs' clause is here}}
#pragma acc kernels num_gangs(1) num_gangs(2)
  while(1);

  // expected-error@+2{{OpenACC 'num_gangs' clause cannot appear more than once on a 'parallel' directive}}
  // expected-note@+1{{previous 'num_gangs' clause is here}}
#pragma acc parallel num_gangs(1) num_gangs(2)
  while(1);

  // expected-error@+3{{OpenACC 'num_gangs' clause cannot appear more than once in a 'device_type' region on a 'kernels' directive}}
  // expected-note@+2{{previous 'num_gangs' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc kernels num_gangs(1) device_type(*) num_gangs(1) num_gangs(2)
  while(1);

  // expected-error@+3{{OpenACC 'num_gangs' clause cannot appear more than once in a 'device_type' region on a 'parallel' directive}}
  // expected-note@+2{{previous 'num_gangs' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc parallel device_type(*) num_gangs(1) num_gangs(2)
  while(1);

#pragma acc parallel num_gangs(1) device_type(*) num_gangs(2)
  while(1);

  // expected-error@+1{{too many integer expression arguments provided to OpenACC 'num_gangs' clause: 'kernels' directive expects maximum of 1, 2 were provided}}
#pragma acc kernels num_gangs(1, getS())
  while(1);

  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'serial' directive}}
#pragma acc serial num_gangs(1, getS())
  while(1);
#pragma acc parallel num_gangs(1, getS())
  while(1);

  struct NotConvertible{} NC;
  // expected-error@+1{{OpenACC clause 'num_gangs' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc parallel num_gangs(NC)
  while(1);

  // expected-error@+1{{OpenACC clause 'num_gangs' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc parallel num_gangs(1, NC)
  while(1);

  // expected-error@+1{{OpenACC clause 'num_gangs' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc parallel num_gangs(NC, 1)
  while(1);

#pragma acc parallel num_gangs(getS(), 1, getS())
  while(1);

  // expected-error@+1{{too many integer expression arguments provided to OpenACC 'num_gangs' clause: 'parallel' directive expects maximum of 3, 4 were provided}}
#pragma acc parallel num_gangs(getS(), 1, getS(), 1)
  while(1);

  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'loop' directive}}
#pragma acc loop num_gangs(1)
  for(int i = 5; i < 10;++i);
}

void no_dupes_since_last_device_type() {
  // expected-error@+4{{OpenACC 'num_gangs' clause applies to 'device_type' 'radeon', which conflicts with previous 'num_gangs' clause}}
  // expected-note@+3{{active 'device_type' clause here}}
  // expected-note@+2{{previous 'num_gangs' clause is here}}
  // expected-note@+1{{which applies to 'device_type' clause here}}
#pragma acc parallel device_type(nvidia, radeon) num_gangs(getS()) device_type(radeon) num_gangs(getS())
  ;
  // expected-error@+4{{OpenACC 'num_gangs' clause applies to 'device_type' 'nvidia', which conflicts with previous 'num_gangs' clause}}
  // expected-note@+3{{active 'device_type' clause here}}
  // expected-note@+2{{previous 'num_gangs' clause is here}}
  // expected-note@+1{{which applies to 'device_type' clause here}}
#pragma acc parallel device_type(nvidia) num_gangs(getS()) device_type(nvidia, radeon) num_gangs(getS())
  ;
}
