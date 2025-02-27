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
#pragma acc parallel loop deviceptr(LocalInt)
  for (int i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop deviceptr(&LocalInt)
  for (int i = 0; i < 5; ++i);

#pragma acc serial loop deviceptr(LocalPtr)
  for (int i = 0; i < 5; ++i);

#pragma acc kernels loop deviceptr(LocalPtr)
  for (int i = 0; i < 5; ++i);


  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int[5]'}}
#pragma acc kernels loop deviceptr(Array)
  for (int i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int'}}
#pragma acc parallel loop deviceptr(Array[0])
  for (int i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc parallel loop deviceptr(Array[0:1])
  for (int i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int *[5]'}}
#pragma acc parallel loop deviceptr(PtrArray)
  for (int i = 0; i < 5; ++i);

#pragma acc parallel loop deviceptr(PtrArray[0])
  for (int i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc parallel loop deviceptr(PtrArray[0:1])
  for (int i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'struct S'}}
#pragma acc parallel loop deviceptr(s)
  for (int i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int'}}
#pragma acc parallel loop deviceptr(s.IntMem)
  for (int i = 0; i < 5; ++i);

#pragma acc parallel loop deviceptr(s.PtrMem)
  for (int i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC 'deviceptr' clause is not valid on 'loop' directive}}
#pragma acc loop deviceptr(LocalInt)
  for(int i = 5; i < 10;++i);
}
