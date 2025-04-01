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
#pragma acc data copyout(LocalInt)
  ;
#pragma acc exit data copyout(LocalInt)

  // expected-warning@+1{{OpenACC clause name 'pcopyout' is a deprecated clause name and is now an alias for 'copyout'}}
#pragma acc data pcopyout(LocalInt)
  ;

  // expected-warning@+1{{OpenACC clause name 'present_or_copyout' is a deprecated clause name and is now an alias for 'copyout'}}
#pragma acc data present_or_copyout(LocalInt)
  ;

  // Valid cases:
#pragma acc data copyout(LocalInt, LocalPointer, LocalArray)
  ;
#pragma acc data copyout(LocalArray[2:1])
  ;
#pragma acc data copyout(zero:LocalArray[2:1])
  ;

#pragma acc data copyout(LocalComposite.ScalarMember, LocalComposite.ScalarMember)
  ;

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data copyout(1 + IntParam)
  ;

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data copyout(+IntParam)
  ;

  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc data copyout(PointerParam[2:])
  ;

  // expected-error@+1{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
#pragma acc data copyout(ArrayParam[2:5])
  ;

  // expected-error@+2{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data copyout((float*)ArrayParam[2:5])
  ;
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data copyout((float)ArrayParam[2])
  ;
  // expected-error@+2{{invalid tag 'invalid' on 'copyout' clause}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data copyout(invalid:(float)ArrayParam[2])
  ;

  // expected-error@+2{{OpenACC 'enter data' construct must have at least one 'copyin', 'create' or 'attach' clause}}
  // expected-error@+1{{OpenACC 'copyout' clause is not valid on 'enter data' directive}}
#pragma acc enter data copyout(LocalInt)
  // expected-error@+2{{OpenACC 'host_data' construct must have at least one 'use_device' clause}}
  // expected-error@+1{{OpenACC 'pcopyout' clause is not valid on 'host_data' directive}}
#pragma acc host_data pcopyout(LocalInt)
  ;
}
