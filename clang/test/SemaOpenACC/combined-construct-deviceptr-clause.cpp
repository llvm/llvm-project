// RUN: %clang_cc1 %s -fopenacc -verify

struct S {
  int IntMem;
  int *PtrMem;
  operator int*();
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

#pragma acc parallel loop deviceptr(LocalPtr)
  for (int i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int[5]'}}
#pragma acc parallel loop deviceptr(Array)
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
}

template<typename T, typename TPtr, typename TStruct, auto &R1>
void Templ() {
  T SomeInt;
  TPtr SomePtr;
  T SomeIntArray[5];
  TPtr SomeIntPtrArray[5];
  TStruct SomeStruct;

  // expected-error@+2{{expected pointer in 'deviceptr' clause, type is 'int'}}
  // expected-note@#INST{{in instantiation of function template specialization}}
#pragma acc parallel loop deviceptr(SomeInt)
  for (int i = 0; i < 5; ++i);

#pragma acc parallel loop deviceptr(SomePtr)
  for (int i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int[5]'}}
#pragma acc parallel loop deviceptr(SomeIntArray)
  for (int i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int'}}
#pragma acc parallel loop deviceptr(SomeIntArray[0])
  for (int i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc parallel loop deviceptr(SomeIntArray[0:1])
  for (int i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int *[5]'}}
#pragma acc parallel loop deviceptr(SomeIntPtrArray)
  for (int i = 0; i < 5; ++i);

#pragma acc parallel loop deviceptr(SomeIntPtrArray[0])
  for (int i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc parallel loop deviceptr(SomeIntPtrArray[0:1])
  for (int i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'S'}}
#pragma acc parallel loop deviceptr(SomeStruct)
  for (int i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int'}}
#pragma acc parallel loop deviceptr(SomeStruct.IntMem)
  for (int i = 0; i < 5; ++i);

#pragma acc parallel loop deviceptr(SomeStruct.PtrMem)
  for (int i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int'}}
#pragma acc parallel loop deviceptr(R1)
  for (int i = 0; i < 5; ++i);
}

void inst() {
  static constexpr int CEVar = 1;
  Templ<int, int*, S, CEVar>(); // #INST
}
