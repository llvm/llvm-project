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
#pragma acc parallel copyout(LocalInt)
  while(1);
#pragma acc serial copyout(LocalInt)
  while(1);
#pragma acc kernels copyout(LocalInt)
  while(1);

  // expected-warning@+1{{OpenACC clause name 'pcopyout' is a deprecated clause name and is now an alias for 'copyout'}}
#pragma acc parallel pcopyout(LocalInt)
  while(1);

  // expected-warning@+1{{OpenACC clause name 'present_or_copyout' is a deprecated clause name and is now an alias for 'copyout'}}
#pragma acc parallel present_or_copyout(LocalInt)
  while(1);

  // Valid cases:
#pragma acc parallel copyout(LocalInt, LocalPointer, LocalArray)
  while(1);
#pragma acc parallel copyout(LocalArray[2:1])
  while(1);
#pragma acc parallel copyout(zero:LocalArray[2:1])
  while(1);

#pragma acc parallel copyout(LocalComposite.ScalarMember, LocalComposite.ScalarMember)
  while(1);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel copyout(1 + IntParam)
  while(1);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel copyout(+IntParam)
  while(1);

  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc parallel copyout(PointerParam[2:])
  while(1);

  // expected-error@+1{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
#pragma acc parallel copyout(ArrayParam[2:5])
  while(1);

  // expected-error@+2{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel copyout((float*)ArrayParam[2:5])
  while(1);
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel copyout((float)ArrayParam[2])
  while(1);
  // expected-error@+2{{invalid tag 'invalid' on 'copyout' clause}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel copyout(invalid:(float)ArrayParam[2])
  while(1);

  // expected-error@+1{{OpenACC 'copyout' clause is not valid on 'loop' directive}}
#pragma acc loop copyout(LocalInt)
  for(int i = 0; i < 6;++i);
  // expected-error@+1{{OpenACC 'pcopyout' clause is not valid on 'loop' directive}}
#pragma acc loop pcopyout(LocalInt)
  for(int i = 0; i < 6;++i);
  // expected-error@+1{{OpenACC 'present_or_copyout' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_copyout(LocalInt)
  for(int i = 0; i < 6;++i);
}
