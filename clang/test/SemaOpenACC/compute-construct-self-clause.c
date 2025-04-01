// RUN: %clang_cc1 %s -fopenacc -verify

void BoolExpr(int *I, float *F) {
  typedef struct {} SomeStruct;
  struct C{};
  // expected-error@+1{{expected expression}}
#pragma acc parallel self (struct C f())
  while(0);

  // expected-error@+1{{unexpected type name 'SomeStruct': expected expression}}
#pragma acc serial self (SomeStruct)
  while(0);

  // expected-error@+1{{unexpected type name 'SomeStruct': expected expression}}
#pragma acc serial self (SomeStruct())
  while(0);

  SomeStruct S;
  // expected-error@+1{{statement requires expression of scalar type ('SomeStruct' invalid)}}
#pragma acc serial self (S)
  while(0);

#pragma acc parallel self (I)
  while(0);

#pragma acc serial self (F)
  while(0);

#pragma acc kernels self (*I < *F)
  while(0);
}

void WarnMaybeNotUsed(int val1, int val2) {

  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel self if(val1)
  while(0);

  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel self(val1) if(val1)
  while(0);

  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel if(val1) self
  while(0);

  // expected-warning@+2{{OpenACC construct 'self' has no effect when an 'if' clause evaluates to true}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel if(val1) self(val2)
  while(0);

  // The below don't warn because one side or the other has an error, thus is
  // not added to the AST.

  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc parallel self if(invalid)
  while(0);

  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc parallel self(invalid) if(val1)
  while(0);

  // expected-error@+2{{expected expression}}
  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc parallel self() if(invalid)
  while(0);

  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc parallel if(invalid) self
  while(0);

  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc parallel if(val2) self(invalid)
  while(0);

  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc parallel if(invalid) self(val1)
  while(0);

  // expected-error@+1{{OpenACC 'self' clause is not valid on 'loop' directive}}
#pragma acc loop self
  for(int i = 5; i < 10;++i);
}
