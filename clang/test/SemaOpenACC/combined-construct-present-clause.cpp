// RUN: %clang_cc1 %s -fopenacc -verify

enum SomeE{};
typedef struct IsComplete {
  struct S { int A; } CompositeMember;
  int ScalarMember;
  float ArrayMember[5];
  SomeE EnumMember;
  char *PointerMember;
} Complete;

void uses(int IntParam, char *PointerParam, float ArrayParam[5], Complete CompositeParam, int &IntParamRef) {
  int LocalInt;
  char *LocalPointer;
  float LocalArray[5];
  // Check Appertainment:
#pragma acc parallel loop present(LocalInt)
  for(unsigned I = 0; I < 5; ++I);
#pragma acc serial loop present(LocalInt)
  for(unsigned I = 0; I < 5; ++I);
#pragma acc kernels loop present(LocalInt)
  for(unsigned I = 0; I < 5; ++I);

  // Valid cases:
#pragma acc parallel loop present(LocalInt, LocalPointer, LocalArray)
  for(unsigned I = 0; I < 5; ++I);
#pragma acc parallel loop present(LocalArray[2:1])
  for(unsigned I = 0; I < 5; ++I);

  Complete LocalComposite2;
#pragma acc parallel loop present(LocalComposite2.ScalarMember, LocalComposite2.ScalarMember)
  for(unsigned I = 0; I < 5; ++I);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop present(1 + IntParam)
  for(unsigned I = 0; I < 5; ++I);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop present(+IntParam)
  for(unsigned I = 0; I < 5; ++I);

  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc parallel loop present(PointerParam[2:])
  for(unsigned I = 0; I < 5; ++I);

  // expected-error@+1{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
#pragma acc parallel loop present(ArrayParam[2:5])
  for(unsigned I = 0; I < 5; ++I);

  // expected-error@+2{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop present((float*)ArrayParam[2:5])
  for(unsigned I = 0; I < 5; ++I);
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop present((float)ArrayParam[2])
  for(unsigned I = 0; I < 5; ++I);
}

template<typename T, unsigned Int, typename V>
void TemplUses(T t, T (&arrayT)[Int], V TemplComp) {
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop present(+t)
  for(unsigned I = 0; I < 5; ++I);

  // NTTP's are only valid if it is a reference to something.
  // expected-error@+2{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
  // expected-note@#TEMPL_USES_INST{{in instantiation of}}
#pragma acc parallel loop present(Int)
  for(unsigned I = 0; I < 5; ++I);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop present(t, Int)
  for(unsigned I = 0; I < 5; ++I);

#pragma acc parallel loop present(arrayT)
  for(unsigned I = 0; I < 5; ++I);

#pragma acc parallel loop present(TemplComp)
  for(unsigned I = 0; I < 5; ++I);

#pragma acc parallel loop present(TemplComp.PointerMember[5])
  for(unsigned I = 0; I < 5; ++I);
 int *Pointer;
#pragma acc parallel loop present(Pointer[:Int])
  for(unsigned I = 0; I < 5; ++I);
#pragma acc parallel loop present(Pointer[:t])
  for(unsigned I = 0; I < 5; ++I);
  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc parallel loop present(Pointer[1:])
  for(unsigned I = 0; I < 5; ++I);
}

template<unsigned Int, auto &NTTP_REF>
void NTTP() {
  // NTTP's are only valid if it is a reference to something.
  // expected-error@+2{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
  // expected-note@#NTTP_INST{{in instantiation of}}
#pragma acc parallel loop present(Int)
  for(unsigned I = 0; I < 5; ++I);

#pragma acc parallel loop present(NTTP_REF)
  for(unsigned I = 0; I < 5; ++I);
}

void Inst() {
  static constexpr int NTTP_REFed = 1;
  int i;
  int Arr[5];
  Complete C;
  TemplUses(i, Arr, C); // #TEMPL_USES_INST
  NTTP<5, NTTP_REFed>(); // #NTTP_INST
}
