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
#pragma acc exit data delete(LocalInt)

  // Valid cases:
#pragma acc exit data delete(LocalInt, LocalPointer, LocalArray)
#pragma acc exit data delete(LocalArray[2:1])
#pragma acc exit data delete(LocalComposite.ScalarMember, LocalComposite.ScalarMember)

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc exit data delete(1 + IntParam)

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc exit data delete(+IntParam)

  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc exit data delete(PointerParam[2:])

  // expected-error@+1{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
#pragma acc exit data delete(ArrayParam[2:5])

  // expected-error@+2{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc exit data delete((float*)ArrayParam[2:5])
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc exit data delete((float)ArrayParam[2])

  // expected-error@+2{{OpenACC 'data' construct must have at least one 'copy', 'copyin', 'copyout', 'create', 'no_create', 'present', 'deviceptr', 'attach' or 'default' clause}}
  // expected-error@+1{{OpenACC 'delete' clause is not valid on 'data' directive}}
#pragma acc data delete(LocalInt)
  ;
  // expected-error@+2{{OpenACC 'enter data' construct must have at least one 'copyin', 'create' or 'attach' clause}}
  // expected-error@+1{{OpenACC 'delete' clause is not valid on 'enter data' directive}}
#pragma acc enter data delete(LocalInt)
  // expected-error@+2{{OpenACC 'host_data' construct must have at least one 'use_device' clause}}
  // expected-error@+1{{OpenACC 'delete' clause is not valid on 'host_data' directive}}
#pragma acc host_data delete(LocalInt)
  ;
}
