// RUN: %clang_cc1 %s -fopenacc -verify

void BoolExpr(int *I, float *F) {

  typedef struct {} SomeStruct;
  int Array[5];

  struct C{};
  // expected-error@+1{{expected expression}}
#pragma acc parallel loop if (struct C f())
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{unexpected type name 'SomeStruct': expected expression}}
#pragma acc serial loop if (SomeStruct)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{unexpected type name 'SomeStruct': expected expression}}
#pragma acc serial loop if (SomeStruct())
  for (unsigned i = 0; i < 5; ++i);

  SomeStruct S;
  // expected-error@+1{{statement requires expression of scalar type ('SomeStruct' invalid)}}
#pragma acc serial loop if (S)
  for (unsigned i = 0; i < 5; ++i);

  // expected-warning@+1{{address of array 'Array' will always evaluate to 'true'}}
#pragma acc kernels loop if (Array)
  for (unsigned i = 0; i < 5; ++i);

  // expected-warning@+4{{incompatible pointer types assigning to 'int *' from 'float *'}}
  // expected-warning@+3{{using the result of an assignment as a condition without parentheses}}
  // expected-note@+2{{place parentheses around the assignment to silence this warning}}
  // expected-note@+1{{use '==' to turn this assignment into an equality comparison}}
#pragma acc kernels loop if (I = F)
  for (unsigned i = 0; i < 5; ++i);

#pragma acc parallel loop if (I)
  for (unsigned i = 0; i < 5; ++i);

#pragma acc serial loop if (F)
  for (unsigned i = 0; i < 5; ++i);

#pragma acc kernels loop if (*I < *F)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC 'data' construct must have at least one 'copy', 'copyin', 'copyout', 'create', 'no_create', 'present', 'deviceptr', 'attach' or 'default' clause}}
#pragma acc data if (*I < *F)
  for (unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop if (*I < *F)
  for(int i = 0; i < 5; ++i);
#pragma acc serial loop if (*I < *F)
  for(int i = 0; i < 5; ++i);
#pragma acc kernels loop if (*I < *F)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC 'if' clause is not valid on 'loop' directive}}
#pragma acc loop if(I)
  for(int i = 5; i < 10;++i);
}
