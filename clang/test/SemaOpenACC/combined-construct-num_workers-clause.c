// RUN: %clang_cc1 %s -fopenacc -verify

short getS();
float getF();
void Test() {
#pragma acc kernels loop num_workers(1)
  for(int i = 5; i < 10;++i);

  // expected-error@+1{{OpenACC 'num_workers' clause is not valid on 'serial loop' directive}}
#pragma acc serial loop num_workers(1)
  for(int i = 5; i < 10;++i);

#pragma acc parallel loop num_workers(1)
  for(int i = 5; i < 10;++i);

  // expected-error@+2{{OpenACC 'num_workers' clause cannot appear more than once on a 'kernels loop' directive}}
  // expected-note@+1{{previous 'num_workers' clause is here}}
#pragma acc kernels loop num_workers(1) num_workers(2)
  for(int i = 5; i < 10;++i);

  // expected-error@+2{{OpenACC 'num_workers' clause cannot appear more than once on a 'parallel loop' directive}}
  // expected-note@+1{{previous 'num_workers' clause is here}}
#pragma acc parallel loop num_workers(1) num_workers(2)
  for(int i = 5; i < 10;++i);

  // expected-error@+3{{OpenACC 'num_workers' clause cannot appear more than once in a 'device_type' region on a 'kernels loop' directive}}
  // expected-note@+2{{previous 'num_workers' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc kernels loop num_workers(1) device_type(*) num_workers(1) num_workers(2)
  for(int i = 5; i < 10;++i);

  // expected-error@+3{{OpenACC 'num_workers' clause cannot appear more than once in a 'device_type' region on a 'parallel loop' directive}}
  // expected-note@+2{{previous 'num_workers' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc parallel loop device_type(*) num_workers(1) num_workers(2)
  for(int i = 5; i < 10;++i);

#pragma acc parallel loop num_workers(1) device_type(*) num_workers(2)
  for(int i = 5; i < 10;++i);

  // expected-error@+1{{OpenACC clause 'num_workers' requires expression of integer type}}
#pragma acc parallel loop num_workers(getF())
  for(int i = 5; i < 10;++i);

  // expected-error@+1{{expected expression}}
#pragma acc kernels loop num_workers()
  for(int i = 5; i < 10;++i);

  // expected-error@+1{{expected expression}}
#pragma acc parallel loop num_workers()
  for(int i = 5; i < 10;++i);

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc kernels loop num_workers(1, 2)
  for(int i = 5; i < 10;++i);

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel loop num_workers(1, 2)
  for(int i = 5; i < 10;++i);
}
