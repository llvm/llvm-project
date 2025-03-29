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
#pragma acc data copyin(LocalInt)
  ;
#pragma acc enter data copyin(LocalInt)

  // expected-warning@+1{{OpenACC clause name 'pcopyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc data pcopyin(LocalInt)
  ;

  // expected-warning@+1{{OpenACC clause name 'present_or_copyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc data present_or_copyin(LocalInt)
  ;

  // Valid cases:
#pragma acc data copyin(LocalInt, LocalPointer, LocalArray)
  ;
#pragma acc data copyin(LocalArray[2:1])
  ;
#pragma acc data copyin(readonly:LocalArray[2:1])
  ;

#pragma acc data copyin(LocalComposite.ScalarMember, LocalComposite.ScalarMember)
  ;

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data copyin(1 + IntParam)
  ;

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data copyin(+IntParam)
  ;

  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc data copyin(PointerParam[2:])
  ;

  // expected-error@+1{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
#pragma acc data copyin(ArrayParam[2:5])
  ;

  // expected-error@+2{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data copyin((float*)ArrayParam[2:5])
  ;
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data copyin((float)ArrayParam[2])
  ;
  // expected-error@+2{{invalid tag 'invalid' on 'copyin' clause}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data copyin(invalid:(float)ArrayParam[2])
  ;

  // expected-error@+2{{OpenACC 'exit data' construct must have at least one 'copyout', 'delete' or 'detach' clause}}
  // expected-error@+1{{OpenACC 'copyin' clause is not valid on 'exit data' directive}}
#pragma acc exit data copyin(LocalInt)
  // expected-error@+2{{OpenACC 'host_data' construct must have at least one 'use_device' clause}}
  // expected-error@+1{{OpenACC 'pcopyin' clause is not valid on 'host_data' directive}}
#pragma acc host_data pcopyin(LocalInt)
  ;
}
