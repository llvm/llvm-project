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
#pragma acc data default(none) deviceptr(LocalInt)
  ;

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data default(none) deviceptr(&LocalInt)
  ;


  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int[5]'}}
#pragma acc data default(none) deviceptr(Array)
  ;

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int'}}
#pragma acc data default(none) deviceptr(Array[0])
  ;

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc data default(none) deviceptr(Array[0:1])
  ;

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int *[5]'}}
#pragma acc data default(none) deviceptr(PtrArray)
  ;

#pragma acc data default(none) deviceptr(PtrArray[0])
  ;

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc data default(none) deviceptr(PtrArray[0:1])
  ;

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'struct S'}}
#pragma acc data default(none) deviceptr(s)
  ;

  // expected-error@+1{{expected pointer in 'deviceptr' clause, type is 'int'}}
#pragma acc data default(none) deviceptr(s.IntMem)
  ;

#pragma acc data default(none) deviceptr(s.PtrMem)
  ;

  // expected-error@+1{{OpenACC 'deviceptr' clause is not valid on 'enter data' directive}}
#pragma acc enter data copyin(LocalInt) deviceptr(LocalInt)
  // expected-error@+1{{OpenACC 'deviceptr' clause is not valid on 'exit data' directive}}
#pragma acc exit data copyout(LocalInt) deviceptr(LocalInt)
  // expected-error@+1{{OpenACC 'deviceptr' clause is not valid on 'host_data' directive}}
#pragma acc host_data use_device(LocalInt) deviceptr(LocalInt)
  ;
}
