// RUN: %clang_cc1 %s -fopenacc -verify

short getS();

void Test() {
#pragma acc parallel vector_length(1)
  while(1);
#pragma acc kernels vector_length(1)
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
