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

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'int'}}
#pragma acc parallel loop attach(LocalInt)
  for (unsigned i = 0; i < 5; ++i);

#pragma acc parallel loop attach(LocalPtr)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'int[5]'}}
#pragma acc parallel loop attach(Array)
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
}

template<typename T, typename TPtr, typename TStruct, auto &R1>
void Templ() {
  T SomeInt;
  TPtr SomePtr;
  T SomeIntArray[5];
  TPtr SomeIntPtrArray[5];
  TStruct SomeStruct;

  // expected-error@+2{{expected pointer in 'attach' clause, type is 'int'}}
  // expected-note@#INST{{in instantiation of function template specialization}}
#pragma acc parallel loop attach(SomeInt)
  for (unsigned i = 0; i < 5; ++i);

#pragma acc parallel loop attach(SomePtr)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'int[5]'}}
#pragma acc parallel loop attach(SomeIntArray)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'int'}}
#pragma acc parallel loop attach(SomeIntArray[0])
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc parallel loop attach(SomeIntArray[0:1])
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'int *[5]'}}
#pragma acc parallel loop attach(SomeIntPtrArray)
  for (unsigned i = 0; i < 5; ++i);

#pragma acc parallel loop attach(SomeIntPtrArray[0])
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc parallel loop attach(SomeIntPtrArray[0:1])
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'S'}}
#pragma acc parallel loop attach(SomeStruct)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'int'}}
#pragma acc parallel loop attach(SomeStruct.IntMem)
  for (unsigned i = 0; i < 5; ++i);

#pragma acc parallel loop attach(SomeStruct.PtrMem)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'int'}}
#pragma acc parallel loop attach(R1)
  for (unsigned i = 0; i < 5; ++i);
}

void inst() {
  static constexpr int CEVar = 1;
  Templ<int, int*, S, CEVar>(); // #INST
}
