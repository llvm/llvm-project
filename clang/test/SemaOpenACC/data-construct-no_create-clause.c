// RUN: %clang_cc1 %s -fopenacc -verify

typedef struct IsComplete {
  struct S { int A; } CompositeMember;
  int ScalarMember;
  float ArrayMember[5];
  void *PointerMember;
} Complete;
void uses(int IntParam, short *PointerParam, float ArrayParam[5], Complete CompositeParam) {
  int LocalInt;
  short *LocalPointer;
  float LocalArray[5];
  Complete LocalComposite;
  // Check Appertainment:
#pragma acc data no_create(LocalInt)
  ;

  // Valid cases:
#pragma acc data no_create(LocalInt, LocalPointer, LocalArray)
  ;
#pragma acc data no_create(LocalArray[2:1])
  ;

#pragma acc data no_create(LocalComposite.ScalarMember, LocalComposite.ScalarMember)
  ;

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data no_create(1 + IntParam)
  ;

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data no_create(+IntParam)
  ;

  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc data no_create(PointerParam[2:])
  ;

  // expected-error@+1{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
#pragma acc data no_create(ArrayParam[2:5])
  ;

  // expected-error@+2{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data no_create((float*)ArrayParam[2:5])
  ;
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data no_create((float)ArrayParam[2])
  ;

  // expected-error@+2{{OpenACC 'exit data' construct must have at least one 'copyout', 'delete' or 'detach' clause}}
  // expected-error@+1{{OpenACC 'no_create' clause is not valid on 'exit data' directive}}
#pragma acc exit data no_create(LocalInt)
  // expected-error@+2{{OpenACC 'enter data' construct must have at least one 'copyin', 'create' or 'attach' clause}}
  // expected-error@+1{{OpenACC 'no_create' clause is not valid on 'enter data' directive}}
#pragma acc enter data no_create(LocalInt)
  // expected-error@+2{{OpenACC 'host_data' construct must have at least one 'use_device' clause}}
  // expected-error@+1{{OpenACC 'no_create' clause is not valid on 'host_data' directive}}
#pragma acc host_data no_create(LocalInt)
  ;
}
