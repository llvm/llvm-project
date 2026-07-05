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
#pragma acc parallel loop no_create(LocalInt)
  for (unsigned i = 0; i < 5; ++i);
#pragma acc serial loop no_create(LocalInt)
  for (unsigned i = 0; i < 5; ++i);
#pragma acc kernels loop no_create(LocalInt)
  for (unsigned i = 0; i < 5; ++i);

  // Valid cases:
#pragma acc parallel loop no_create(LocalInt, LocalPointer, LocalArray)
  for (unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop no_create(LocalArray[2:1])
  for (unsigned i = 0; i < 5; ++i);

#pragma acc parallel loop no_create(LocalComposite.ScalarMember, LocalComposite.ScalarMember)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop no_create(1 + IntParam)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop no_create(+IntParam)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc parallel loop no_create(PointerParam[2:])
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
#pragma acc parallel loop no_create(ArrayParam[2:5])
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop no_create((float*)ArrayParam[2:5])
  for (unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop no_create((float)ArrayParam[2])
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC 'no_create' clause is not valid on 'loop' directive}}
#pragma acc loop no_create(LocalInt)
  for(int i = 5; i < 10;++i);
}
