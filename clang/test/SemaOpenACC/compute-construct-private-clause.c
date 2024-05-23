// RUN: %clang_cc1 %s -fopenacc -verify

struct Incomplete;
enum SomeE{ A };
typedef struct IsComplete {
  struct S { int A; } CompositeMember;
  int ScalarMember;
  float ArrayMember[5];
  enum SomeE EnumMember;
  void *PointerMember;
} Complete;

int GlobalInt;
float GlobalArray[5];
short *GlobalPointer;
Complete GlobalComposite;

void uses(int IntParam, short *PointerParam, float ArrayParam[5], Complete CompositeParam) {
  int LocalInt;
  short *LocalPointer;
  float LocalArray[5];
  Complete LocalComposite;

  // Check Appertainment:
#pragma acc parallel private(LocalInt)
  while(1);
#pragma acc serial private(LocalInt)
  while(1);
  // expected-error@+1{{OpenACC 'private' clause is not valid on 'kernels' directive}}
#pragma acc kernels private(LocalInt)
  while(1);

  // Valid cases:
#pragma acc parallel private(LocalInt, LocalPointer, LocalArray)
  while(1);
#pragma acc parallel private(LocalArray)
  while(1);
#pragma acc parallel private(LocalArray[:])
  while(1);
#pragma acc parallel private(LocalArray[:5])
  while(1);
#pragma acc parallel private(LocalArray[2:])
  while(1);
#pragma acc parallel private(LocalArray[2:1])
  while(1);
#pragma acc parallel private(LocalArray[2])
  while(1);
#pragma acc parallel private(LocalComposite)
  while(1);
#pragma acc parallel private(LocalComposite.EnumMember)
  while(1);
#pragma acc parallel private(LocalComposite.ScalarMember)
  while(1);
#pragma acc parallel private(LocalComposite.ArrayMember)
  while(1);
#pragma acc parallel private(LocalComposite.ArrayMember[5])
  while(1);
#pragma acc parallel private(LocalComposite.PointerMember)
  while(1);
#pragma acc parallel private(GlobalInt, GlobalArray, GlobalPointer, GlobalComposite)
  while(1);
#pragma acc parallel private(GlobalArray[2], GlobalPointer[2], GlobalComposite.CompositeMember.A)
  while(1);
#pragma acc parallel private(LocalComposite, GlobalComposite)
  while(1);
#pragma acc parallel private(IntParam, PointerParam, ArrayParam, CompositeParam)
  while(1);
#pragma acc parallel private(PointerParam[IntParam], ArrayParam[IntParam], CompositeParam.CompositeMember.A)
  while(1);

#pragma acc parallel private(LocalArray) private(LocalArray[2])
  while(1);

#pragma acc parallel private(LocalArray, LocalArray[2])
  while(1);

#pragma acc parallel private(LocalComposite, LocalComposite.ScalarMember)
  while(1);

#pragma acc parallel private(LocalComposite.CompositeMember.A, LocalComposite.ScalarMember)
  while(1);

#pragma acc parallel private(LocalComposite.CompositeMember.A) private(LocalComposite.ScalarMember)
  while(1);

  Complete LocalComposite2;
#pragma acc parallel private(LocalComposite2.ScalarMember, LocalComposite2.ScalarMember)
  while(1);

  // Invalid cases, arbitrary expressions.
  struct Incomplete *I;
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel private(*I)
  while(1);
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel private(GlobalInt + IntParam)
  while(1);
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel private(+GlobalInt)
  while(1);

  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc parallel private(PointerParam[:])
  while(1);
#pragma acc parallel private(PointerParam[:5])
  while(1);
#pragma acc parallel private(PointerParam[:IntParam])
  while(1);
  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc parallel private(PointerParam[2:])
  while(1);
#pragma acc parallel private(PointerParam[2:5])
  while(1);
#pragma acc parallel private(PointerParam[2])
  while(1);
#pragma acc parallel private(ArrayParam[:])
  while(1);
#pragma acc parallel private(ArrayParam[:5])
  while(1);
#pragma acc parallel private(ArrayParam[:IntParam])
  while(1);
#pragma acc parallel private(ArrayParam[2:])
  while(1);
  // expected-error@+1{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
#pragma acc parallel private(ArrayParam[2:5])
  while(1);
#pragma acc parallel private(ArrayParam[2])
  while(1);

  // expected-error@+2{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel private((float*)ArrayParam[2:5])
  while(1);
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel private((float)ArrayParam[2])
  while(1);
}
