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
