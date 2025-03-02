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
#pragma acc data create(LocalInt)
  ;
#pragma acc enter data create(LocalInt)

  // expected-warning@+1{{OpenACC clause name 'pcreate' is a deprecated clause name and is now an alias for 'create'}}
#pragma acc data pcreate(LocalInt)
  ;

  // expected-warning@+1{{OpenACC clause name 'present_or_create' is a deprecated clause name and is now an alias for 'create'}}
#pragma acc data present_or_create(LocalInt)
  ;

  // Valid cases:
#pragma acc data create(LocalInt, LocalPointer, LocalArray)
  ;
#pragma acc data create(LocalArray[2:1])
  ;
#pragma acc data create(zero:LocalArray[2:1])
  ;

#pragma acc data create(LocalComposite.ScalarMember, LocalComposite.ScalarMember)
  ;

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data create(1 + IntParam)
  ;

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data create(+IntParam)
  ;

  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc data create(PointerParam[2:])
  ;

  // expected-error@+1{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
#pragma acc data create(ArrayParam[2:5])
  ;

  // expected-error@+2{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data create((float*)ArrayParam[2:5])
  ;
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data create((float)ArrayParam[2])
  ;
  // expected-error@+2{{invalid tag 'invalid' on 'create' clause}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data create(invalid:(float)ArrayParam[2])
  ;

  // expected-error@+2{{OpenACC 'exit data' construct must have at least one 'copyout', 'delete' or 'detach' clause}}
  // expected-error@+1{{OpenACC 'create' clause is not valid on 'exit data' directive}}
#pragma acc exit data create(LocalInt)
  // expected-error@+2{{OpenACC 'host_data' construct must have at least one 'use_device' clause}}
  // expected-error@+1{{OpenACC 'pcreate' clause is not valid on 'host_data' directive}}
#pragma acc host_data pcreate(LocalInt)
  ;
}
