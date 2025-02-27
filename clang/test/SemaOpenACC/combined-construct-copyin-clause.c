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
#pragma acc parallel loop copyin(LocalInt)
  for(int i = 0; i < 5; ++i);
#pragma acc serial loop copyin(LocalInt)
  for(int i = 0; i < 5; ++i);
#pragma acc kernels loop copyin(LocalInt)
  for(int i = 0; i < 5; ++i);

  // expected-warning@+1{{OpenACC clause name 'pcopyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc parallel loop pcopyin(LocalInt)
  for(int i = 0; i < 5; ++i);

  // expected-warning@+1{{OpenACC clause name 'present_or_copyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc parallel loop present_or_copyin(LocalInt)
  for(int i = 0; i < 5; ++i);

  // Valid cases:
#pragma acc parallel loop copyin(LocalInt, LocalPointer, LocalArray)
  for(int i = 0; i < 5; ++i);
#pragma acc parallel loop copyin(LocalArray[2:1])
  for(int i = 0; i < 5; ++i);
#pragma acc parallel loop copyin(readonly:LocalArray[2:1])
  for(int i = 0; i < 5; ++i);

#pragma acc parallel loop copyin(LocalComposite.ScalarMember, LocalComposite.ScalarMember)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop copyin(1 + IntParam)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop copyin(+IntParam)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc parallel loop copyin(PointerParam[2:])
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
#pragma acc parallel loop copyin(ArrayParam[2:5])
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop copyin((float*)ArrayParam[2:5])
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop copyin((float)ArrayParam[2])
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{invalid tag 'invalid' on 'copyin' clause}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop copyin(invalid:(float)ArrayParam[2])
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC 'copyin' clause is not valid on 'loop' directive}}
#pragma acc loop copyin(LocalInt)
  for(int i = 5; i < 10;++i);
  // expected-error@+1{{OpenACC 'pcopyin' clause is not valid on 'loop' directive}}
#pragma acc loop pcopyin(LocalInt)
  for(int i = 5; i < 10;++i);
  // expected-error@+1{{OpenACC 'present_or_copyin' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_copyin(LocalInt)
  for(int i = 5; i < 10;++i);
}
