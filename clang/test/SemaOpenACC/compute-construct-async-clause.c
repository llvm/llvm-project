// RUN: %clang_cc1 %s -fopenacc -verify

short getS();

void Test() {
#pragma acc parallel async
  while(1);
#pragma acc parallel async(1)
  while(1);
#pragma acc kernels async(1)
  while(1);
#pragma acc kernels async(-51)
  while(1);

#pragma acc serial async(1)
  while(1);

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial async(1, 2)
  while(1);

  struct NotConvertible{} NC;
  // expected-error@+1{{OpenACC clause 'async' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc parallel async(NC)
  while(1);

#pragma acc kernels async(getS())
  while(1);

  struct Incomplete *SomeIncomplete;

  // expected-error@+1{{OpenACC clause 'async' requires expression of integer type ('struct Incomplete' invalid)}}
#pragma acc kernels async(*SomeIncomplete)
  while(1);

  enum E{A} SomeE;

#pragma acc kernels async(SomeE)
  while(1);

  // expected-error@+1{{OpenACC 'async' clause is not valid on 'loop' directive}}
#pragma acc loop async(1)
  for(;;);
}
