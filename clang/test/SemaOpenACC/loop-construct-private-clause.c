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
#pragma acc loop private(LocalInt)
  for(int i = 0; i < 5; ++i);

  // Valid cases:
#pragma acc loop private(LocalInt, LocalPointer, LocalArray)
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(LocalArray)
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(LocalArray[:])
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(LocalArray[:5])
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(LocalArray[2:])
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(LocalArray[2:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(LocalArray[2])
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(LocalComposite)
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(LocalComposite.EnumMember)
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(LocalComposite.ScalarMember)
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(LocalComposite.ArrayMember)
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(LocalComposite.ArrayMember[5])
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(LocalComposite.PointerMember)
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(GlobalInt, GlobalArray, GlobalPointer, GlobalComposite)
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(GlobalArray[2], GlobalPointer[2], GlobalComposite.CompositeMember.A)
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(LocalComposite, GlobalComposite)
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(IntParam, PointerParam, ArrayParam, CompositeParam)
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(PointerParam[IntParam], ArrayParam[IntParam], CompositeParam.CompositeMember.A)
  for(int i = 0; i < 5; ++i);

#pragma acc loop private(LocalArray) private(LocalArray[2])
  for(int i = 0; i < 5; ++i);

#pragma acc loop private(LocalArray, LocalArray[2])
  for(int i = 0; i < 5; ++i);

#pragma acc loop private(LocalComposite, LocalComposite.ScalarMember)
  for(int i = 0; i < 5; ++i);

#pragma acc loop private(LocalComposite.CompositeMember.A, LocalComposite.ScalarMember)
  for(int i = 0; i < 5; ++i);

#pragma acc loop private(LocalComposite.CompositeMember.A) private(LocalComposite.ScalarMember)
  for(int i = 0; i < 5; ++i);

  Complete LocalComposite2;
#pragma acc loop private(LocalComposite2.ScalarMember, LocalComposite2.ScalarMember)
  for(int i = 0; i < 5; ++i);

  // Invalid cases, arbitrary expressions.
  struct Incomplete *I;
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc loop private(*I)
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc loop private(GlobalInt + IntParam)
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc loop private(+GlobalInt)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc loop private(PointerParam[:])
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(PointerParam[:5])
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(PointerParam[:IntParam])
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc loop private(PointerParam[2:])
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(PointerParam[2:5])
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(PointerParam[2])
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(ArrayParam[:])
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(ArrayParam[:5])
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(ArrayParam[:IntParam])
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(ArrayParam[2:])
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
#pragma acc loop private(ArrayParam[2:5])
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(ArrayParam[2])
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc loop private((float*)ArrayParam[2:5])
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc loop private((float)ArrayParam[2])
  for(int i = 0; i < 5; ++i);
}
