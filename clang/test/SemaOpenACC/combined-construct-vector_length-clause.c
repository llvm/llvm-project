// RUN: %clang_cc1 %s -fopenacc -verify

short getS();
float getF();
void Test() {
#pragma acc kernels loop vector_length(1)
  for(int i = 5; i < 10;++i);

  // expected-error@+1{{OpenACC 'vector_length' clause is not valid on 'serial loop' directive}}
#pragma acc serial loop vector_length(1)
  for(int i = 5; i < 10;++i);

#pragma acc parallel loop vector_length(1)
  for(int i = 5; i < 10;++i);

  // expected-error@+1{{OpenACC clause 'vector_length' requires expression of integer type}}
#pragma acc parallel loop vector_length(getF())
  for(int i = 5; i < 10;++i);

  // expected-error@+1{{expected expression}}
#pragma acc kernels loop vector_length()
  for(int i = 5; i < 10;++i);

  // expected-error@+1{{expected expression}}
#pragma acc parallel loop vector_length()
  for(int i = 5; i < 10;++i);

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc kernels loop vector_length(1, 2)
  for(int i = 5; i < 10;++i);

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel loop vector_length(1, 2)
  for(int i = 5; i < 10;++i);
}
