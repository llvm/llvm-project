// RUN: %clang_cc1 %s -fopenacc -verify

struct S {
  int IntMem;
  int *PtrMem;
};

void uses() {
  int LocalInt;
  int *LocalPtr;
  int Array[5];
  int *PtrArray[5];
  struct S s;

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'int'}}
#pragma acc parallel loop attach(LocalInt)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop attach(&LocalInt)
  for (unsigned i = 0; i < 5; ++i);

#pragma acc serial loop attach(LocalPtr)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'int[5]'}}
#pragma acc kernels loop attach(Array)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'int'}}
#pragma acc parallel loop attach(Array[0])
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc parallel loop attach(Array[0:1])
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'int *[5]'}}
#pragma acc parallel loop attach(PtrArray)
  for (unsigned i = 0; i < 5; ++i);

#pragma acc parallel loop attach(PtrArray[0])
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc parallel loop attach(PtrArray[0:1])
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'struct S'}}
#pragma acc parallel loop attach(s)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'int'}}
#pragma acc parallel loop attach(s.IntMem)
  for (unsigned i = 0; i < 5; ++i);

#pragma acc parallel loop attach(s.PtrMem)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC 'attach' clause is not valid on 'loop' directive}}
#pragma acc loop attach(LocalInt)
  for(int i = 5; i < 10;++i);
}
