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
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels num_gangs(1) num_gangs(2)
  while(1);

  // expected-error@+2{{OpenACC 'num_gangs' clause cannot appear more than once on a 'parallel' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel num_gangs(1) num_gangs(2)
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
  for(;;);
}
