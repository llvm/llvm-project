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

  // expected-error@+1{{expected pointer in 'detach' clause, type is 'int'}}
#pragma acc exit data copyout(LocalInt) detach(LocalInt)
  ;

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc exit data copyout(LocalInt) detach(&LocalInt)
  ;


  // expected-error@+1{{expected pointer in 'detach' clause, type is 'int[5]'}}
#pragma acc exit data copyout(LocalInt) detach(Array)

  // expected-error@+1{{expected pointer in 'detach' clause, type is 'int'}}
#pragma acc exit data copyout(LocalInt) detach(Array[0])
  ;

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc exit data copyout(LocalInt) detach(Array[0:1])
  ;

  // expected-error@+1{{expected pointer in 'detach' clause, type is 'int *[5]'}}
#pragma acc exit data copyout(LocalInt) detach(PtrArray)
  ;

#pragma acc exit data copyout(LocalInt) detach(PtrArray[0])
  ;

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc exit data copyout(LocalInt) detach(PtrArray[0:1])
  ;

  // expected-error@+1{{expected pointer in 'detach' clause, type is 'struct S'}}
#pragma acc exit data copyout(LocalInt) detach(s)
  ;

  // expected-error@+1{{expected pointer in 'detach' clause, type is 'int'}}
#pragma acc exit data copyout(LocalInt) detach(s.IntMem)
  ;

#pragma acc exit data copyout(LocalInt) detach(s.PtrMem)
  ;

  // expected-error@+1{{OpenACC 'detach' clause is not valid on 'data' directive}}
#pragma acc data copyin(LocalInt) detach(PtrArray[0])
  ;
  // expected-error@+1{{OpenACC 'detach' clause is not valid on 'enter data' directive}}
#pragma acc enter data copyin(LocalInt) detach(PtrArray[0])
  // expected-error@+1{{OpenACC 'detach' clause is not valid on 'host_data' directive}}
#pragma acc host_data use_device(LocalInt) detach(PtrArray[0])
  ;
}
