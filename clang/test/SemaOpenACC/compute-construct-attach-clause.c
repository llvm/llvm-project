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
#pragma acc parallel attach(LocalInt)
  while (1);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel attach(&LocalInt)
  while (1);

#pragma acc serial attach(LocalPtr)
  while (1);

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'int[5]'}}
#pragma acc kernels attach(Array)
  while (1);

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'int'}}
#pragma acc parallel attach(Array[0])
  while (1);

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc parallel attach(Array[0:1])
  while (1);

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'int *[5]'}}
#pragma acc parallel attach(PtrArray)
  while (1);

#pragma acc parallel attach(PtrArray[0])
  while (1);

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc parallel attach(PtrArray[0:1])
  while (1);

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'struct S'}}
#pragma acc parallel attach(s)
  while (1);

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'int'}}
#pragma acc parallel attach(s.IntMem)
  while (1);

#pragma acc parallel attach(s.PtrMem)
  while (1);

  // expected-error@+1{{OpenACC 'attach' clause is not valid on 'loop' directive}}
#pragma acc loop attach(LocalInt)
  for(;;);
}
