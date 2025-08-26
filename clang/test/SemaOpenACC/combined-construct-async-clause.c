// RUN: %clang_cc1 %s -fopenacc -verify

short getS();

void Test() {
#pragma acc parallel loop async
  for (int i = 5; i < 10; ++i);
#pragma acc parallel loop async(1)
  for (int i = 5; i < 10; ++i);
#pragma acc kernels loop async(1)
  for (int i = 5; i < 10; ++i);
#pragma acc kernels loop async(-51)
  for (int i = 5; i < 10; ++i);

#pragma acc serial loop async(1)
  for (int i = 5; i < 10; ++i);

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial loop async(1, 2)
  for (int i = 5; i < 10; ++i);

  struct NotConvertible{} NC;
  // expected-error@+1{{OpenACC clause 'async' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc parallel loop async(NC)
  for (int i = 5; i < 10; ++i);

#pragma acc kernels loop async(getS())
  for (int i = 5; i < 10; ++i);

  struct Incomplete *SomeIncomplete;

  // expected-error@+1{{OpenACC clause 'async' requires expression of integer type ('struct Incomplete' invalid)}}
#pragma acc kernels loop async(*SomeIncomplete)
  for (int i = 5; i < 10; ++i);

  enum E{A} SomeE;

#pragma acc kernels loop async(SomeE)
  for (int i = 5; i < 10; ++i);

  // expected-error@+1{{OpenACC 'async' clause is not valid on 'loop' directive}}
#pragma acc loop async(1)
  for(int i = 5; i < 10;++i);
}
