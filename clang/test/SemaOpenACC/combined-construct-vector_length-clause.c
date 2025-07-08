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

  // expected-error@+2{{OpenACC 'vector_length' clause cannot appear more than once on a 'kernels loop' directive}}
  // expected-note@+1{{previous 'vector_length' clause is here}}
#pragma acc kernels loop vector_length(1) vector_length(2)
  for(int i = 5; i < 10;++i);

  // expected-error@+2{{OpenACC 'vector_length' clause cannot appear more than once on a 'parallel loop' directive}}
  // expected-note@+1{{previous 'vector_length' clause is here}}
#pragma acc parallel loop vector_length(1) vector_length(2)
  for(int i = 5; i < 10;++i);

  // expected-error@+3{{OpenACC 'vector_length' clause cannot appear more than once in a 'device_type' region on a 'kernels loop' directive}}
  // expected-note@+2{{previous 'vector_length' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc kernels loop vector_length(1) device_type(*) vector_length(1) vector_length(2)
  for(int i = 5; i < 10;++i);

  // expected-error@+3{{OpenACC 'vector_length' clause cannot appear more than once in a 'device_type' region on a 'parallel loop' directive}}
  // expected-note@+2{{previous 'vector_length' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc parallel loop device_type(*) vector_length(1) vector_length(2)
  for(int i = 5; i < 10;++i);

#pragma acc parallel loop vector_length(1) device_type(*) vector_length(2)
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
