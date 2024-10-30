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
#pragma acc parallel deviceptr(LocalInt)
  while (true);

#pragma acc parallel deviceptr(LocalPtr)
  while (true);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int[5]'}}
#pragma acc parallel deviceptr(Array)
  while (true);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int'}}
#pragma acc parallel deviceptr(Array[0])
  while (true);

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc parallel deviceptr(Array[0:1])
  while (true);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int *[5]'}}
#pragma acc parallel deviceptr(PtrArray)
  while (true);

#pragma acc parallel deviceptr(PtrArray[0])
  while (true);

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc parallel deviceptr(PtrArray[0:1])
  while (true);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'struct S'}}
#pragma acc parallel deviceptr(s)
  while (true);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int'}}
#pragma acc parallel deviceptr(s.IntMem)
  while (true);

#pragma acc parallel deviceptr(s.PtrMem)
  while (true);
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
#pragma acc parallel deviceptr(SomeInt)
  while (true);

#pragma acc parallel deviceptr(SomePtr)
  while (true);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int[5]'}}
#pragma acc parallel deviceptr(SomeIntArray)
  while (true);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int'}}
#pragma acc parallel deviceptr(SomeIntArray[0])
  while (true);

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc parallel deviceptr(SomeIntArray[0:1])
  while (true);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int *[5]'}}
#pragma acc parallel deviceptr(SomeIntPtrArray)
  while (true);

#pragma acc parallel deviceptr(SomeIntPtrArray[0])
  while (true);

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc parallel deviceptr(SomeIntPtrArray[0:1])
  while (true);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'S'}}
#pragma acc parallel deviceptr(SomeStruct)
  while (true);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int'}}
#pragma acc parallel deviceptr(SomeStruct.IntMem)
  while (true);

#pragma acc parallel deviceptr(SomeStruct.PtrMem)
  while (true);

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int'}}
#pragma acc parallel deviceptr(R1)
  while (true);
}

void inst() {
  static constexpr int CEVar = 1;
  Templ<int, int*, S, CEVar>(); // #INST
}
