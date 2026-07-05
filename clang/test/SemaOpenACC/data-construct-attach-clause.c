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
#pragma acc data default(none) attach(LocalInt)
  ;

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data default(none) attach(&LocalInt)
  ;


  // expected-error@+1{{expected pointer in 'attach' clause, type is 'int[5]'}}
#pragma acc enter data copyin(LocalInt) attach(Array)

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'int'}}
#pragma acc data default(none) attach(Array[0])
  ;

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc data default(none) attach(Array[0:1])
  ;

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'int *[5]'}}
#pragma acc data default(none) attach(PtrArray)
  ;

#pragma acc data default(none) attach(PtrArray[0])
  ;

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc data default(none) attach(PtrArray[0:1])
  ;

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'struct S'}}
#pragma acc data default(none) attach(s)
  ;

  // expected-error@+1{{expected pointer in 'attach' clause, type is 'int'}}
#pragma acc data default(none) attach(s.IntMem)
  ;

#pragma acc data default(none) attach(s.PtrMem)
  ;

  // expected-error@+1{{OpenACC 'attach' clause is not valid on 'exit data' directive}}
#pragma acc exit data copyout(LocalInt) attach(PtrArray[0])
  // expected-error@+1{{OpenACC 'attach' clause is not valid on 'host_data' directive}}
#pragma acc host_data use_device(LocalInt) attach(PtrArray[0])
  ;
}
