// RUN: %clang_cc1 %s -fopenacc -verify

void BoolExpr(int *I, float *F) {
  typedef struct {} SomeStruct;
  struct C{};
  // expected-error@+1{{expected expression}}
#pragma acc parallel loop self (struct C f())
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{unexpected type name 'SomeStruct': expected expression}}
#pragma acc serial loop self (SomeStruct)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{unexpected type name 'SomeStruct': expected expression}}
#pragma acc kernels loop self (SomeStruct())
  for (unsigned i = 0; i < 5; ++i);

  SomeStruct S;
  // expected-error@+1{{statement requires expression of scalar type ('SomeStruct' invalid)}}
#pragma acc parallel loop self (S)
  for (unsigned i = 0; i < 5; ++i);

#pragma acc parallel loop self (I)
  for (unsigned i = 0; i < 5; ++i);

#pragma acc serial loop self (F)
  for (unsigned i = 0; i < 5; ++i);

#pragma acc kernels loop self (*I < *F)
  for (unsigned i = 0; i < 5; ++i);
}

void WarnMaybeNotUsed(int val1, int val2) {

  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop self if(val1)
  for (unsigned i = 0; i < 5; ++i);

  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc serial loop self(val1) if(val1)
  for (unsigned i = 0; i < 5; ++i);

  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop if(val1) self
  for (unsigned i = 0; i < 5; ++i);

  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop if(val1) self(val2)
  for (unsigned i = 0; i < 5; ++i);

  // The below don't warn because one side or the other has an error, thus is
  // not added to the AST.

  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc serial loop self if(invalid)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc kernels loop self(invalid) if(val1)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+2{{expected expression}}
  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc parallel loop self() if(invalid)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc serial loop if(invalid) self
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc kernels loop if(val2) self(invalid)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc parallel loop if(invalid) self(val1)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC 'self' clause is not valid on 'loop' directive}}
#pragma acc loop self
  for(int i = 5; i < 10;++i);
}
