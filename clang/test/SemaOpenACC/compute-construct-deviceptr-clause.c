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

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int'}}
#pragma acc parallel deviceptr(LocalInt)
  while (1);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel deviceptr(&LocalInt)
  while (1);

#pragma acc serial deviceptr(LocalPtr)
  while (1);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int[5]'}}
#pragma acc kernels deviceptr(Array)
  while (1);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int'}}
#pragma acc parallel deviceptr(Array[0])
  while (1);

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc parallel deviceptr(Array[0:1])
  while (1);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int *[5]'}}
#pragma acc parallel deviceptr(PtrArray)
  while (1);

#pragma acc parallel deviceptr(PtrArray[0])
  while (1);

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc parallel deviceptr(PtrArray[0:1])
  while (1);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'struct S'}}
#pragma acc parallel deviceptr(s)
  while (1);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int'}}
#pragma acc parallel deviceptr(s.IntMem)
  while (1);

#pragma acc parallel deviceptr(s.PtrMem)
  while (1);

  // expected-error@+1{{OpenACC 'deviceptr' clause is not valid on 'loop' directive}}
#pragma acc loop deviceptr(LocalInt)
  for(;;);
}
