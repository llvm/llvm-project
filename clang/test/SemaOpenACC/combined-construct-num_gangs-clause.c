// RUN: %clang_cc1 %s -fopenacc -verify

short getS();
float getF();
void Test() {
#pragma acc kernels loop num_gangs(1)
  for(int i = 5; i < 10;++i);

  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'serial loop' directive}}
#pragma acc serial loop num_gangs(1)
  for(int i = 5; i < 10;++i);

#pragma acc parallel loop num_gangs(1)
  for(int i = 5; i < 10;++i);

  // expected-error@+1{{OpenACC clause 'num_gangs' requires expression of integer type}}
#pragma acc parallel loop num_gangs(getF())
  for(int i = 5; i < 10;++i);

  // expected-error@+1{{expected expression}}
#pragma acc kernels loop num_gangs()
  for(int i = 5; i < 10;++i);

  // expected-error@+1{{expected expression}}
#pragma acc parallel loop num_gangs()
  for(int i = 5; i < 10;++i);

  // expected-error@+2{{OpenACC 'num_gangs' clause cannot appear more than once on a 'kernels loop' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop num_gangs(1) num_gangs(2)
  for(int i = 5; i < 10;++i);

  // expected-error@+2{{OpenACC 'num_gangs' clause cannot appear more than once on a 'parallel loop' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop num_gangs(1) num_gangs(2)
  for(int i = 5; i < 10;++i);

  // expected-error@+1{{too many integer expression arguments provided to OpenACC 'num_gangs' clause: 'kernels loop' directive expects maximum of 1, 2 were provided}}
#pragma acc kernels loop num_gangs(1, getS())
  for(int i = 5; i < 10;++i);

  // expected-error@+1{{too many integer expression arguments provided to OpenACC 'num_gangs' clause: 'parallel loop' directive expects maximum of 3, 4 were provided}}
#pragma acc parallel loop num_gangs(getS(), 1, getS(), 1)
  for(int i = 5; i < 10;++i);
}
