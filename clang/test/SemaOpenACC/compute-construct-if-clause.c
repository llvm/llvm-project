// RUN: %clang_cc1 %s -fopenacc -verify

void BoolExpr(int *I, float *F) {

  typedef struct {} SomeStruct;
  int Array[5];

  struct C{};
  // expected-error@+1{{expected expression}}
#pragma acc parallel if (struct C f())
  while(0);

  // expected-error@+1{{unexpected type name 'SomeStruct': expected expression}}
#pragma acc serial if (SomeStruct)
  while(0);

  // expected-error@+1{{unexpected type name 'SomeStruct': expected expression}}
#pragma acc serial if (SomeStruct())
  while(0);

  SomeStruct S;
  // expected-error@+1{{statement requires expression of scalar type ('SomeStruct' invalid)}}
#pragma acc serial if (S)
  while(0);

  // expected-warning@+1{{address of array 'Array' will always evaluate to 'true'}}
#pragma acc kernels if (Array)
  while(0);

  // expected-warning@+4{{incompatible pointer types assigning to 'int *' from 'float *'}}
  // expected-warning@+3{{using the result of an assignment as a condition without parentheses}}
  // expected-note@+2{{place parentheses around the assignment to silence this warning}}
  // expected-note@+1{{use '==' to turn this assignment into an equality comparison}}
#pragma acc kernels if (I = F)
  while(0);

#pragma acc parallel if (I)
  while(0);

#pragma acc serial if (F)
  while(0);

#pragma acc kernels if (*I < *F)
  while(0);

  // expected-warning@+2{{OpenACC construct 'data' not yet implemented}}
  // expected-warning@+1{{OpenACC clause 'if' not yet implemented}}
#pragma acc data if (*I < *F)
  while(0);
  // expected-warning@+2{{OpenACC construct 'parallel loop' not yet implemented}}
  // expected-warning@+1{{OpenACC clause 'if' not yet implemented}}
#pragma acc parallel loop if (*I < *F)
  while(0);
  // expected-warning@+2{{OpenACC construct 'serial loop' not yet implemented}}
  // expected-warning@+1{{OpenACC clause 'if' not yet implemented}}
#pragma acc serial loop if (*I < *F)
  while(0);
  // expected-warning@+2{{OpenACC construct 'kernels loop' not yet implemented}}
  // expected-warning@+1{{OpenACC clause 'if' not yet implemented}}
#pragma acc kernels loop if (*I < *F)
  while(0);

  // expected-error@+1{{OpenACC 'if' clause is not valid on 'loop' directive}}
#pragma acc loop if(I)
  for(;;);
}
