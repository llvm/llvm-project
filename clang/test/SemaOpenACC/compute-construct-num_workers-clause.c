// RUN: %clang_cc1 %s -fopenacc -verify

short getS();

void Test() {
#pragma acc parallel num_workers(1)
  while(1);
#pragma acc kernels num_workers(1)
  while(1);

  // expected-error@+1{{OpenACC 'num_workers' clause is not valid on 'serial' directive}}
#pragma acc serial num_workers(1)
  while(1);

  struct NotConvertible{} NC;
  // expected-error@+1{{OpenACC clause 'num_workers' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc parallel num_workers(NC)
  while(1);

#pragma acc kernels num_workers(getS())
  while(1);

  struct Incomplete *SomeIncomplete;

  // expected-error@+1{{OpenACC clause 'num_workers' requires expression of integer type ('struct Incomplete' invalid)}}
#pragma acc kernels num_workers(*SomeIncomplete)
  while(1);

  enum E{A} SomeE;

#pragma acc kernels num_workers(SomeE)
  while(1);

  // expected-error@+1{{OpenACC 'num_workers' clause is not valid on 'loop' directive}}
#pragma acc loop num_workers(1)
  for(;;);
}
