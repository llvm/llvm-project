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
#pragma acc parallel copyin(LocalInt)
  while(1);
#pragma acc serial copyin(LocalInt)
  while(1);
#pragma acc kernels copyin(LocalInt)
  while(1);

  // expected-warning@+1{{OpenACC clause name 'pcopyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc parallel pcopyin(LocalInt)
  while(1);

  // expected-warning@+1{{OpenACC clause name 'present_or_copyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc parallel present_or_copyin(LocalInt)
  while(1);

  // Valid cases:
#pragma acc parallel copyin(LocalInt, LocalPointer, LocalArray)
  while(1);
#pragma acc parallel copyin(LocalArray[2:1])
  while(1);
#pragma acc parallel copyin(readonly:LocalArray[2:1])
  while(1);

#pragma acc parallel copyin(LocalComposite.ScalarMember, LocalComposite.ScalarMember)
  while(1);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel copyin(1 + IntParam)
  while(1);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel copyin(+IntParam)
  while(1);

  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc parallel copyin(PointerParam[2:])
  while(1);

  // expected-error@+1{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
#pragma acc parallel copyin(ArrayParam[2:5])
  while(1);

  // expected-error@+2{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel copyin((float*)ArrayParam[2:5])
  while(1);
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel copyin((float)ArrayParam[2])
  while(1);
  // expected-error@+2{{invalid tag 'invalid' on 'copyin' clause}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel copyin(invalid:(float)ArrayParam[2])
  while(1);

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
