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
#pragma acc parallel copy(LocalInt)
  while(1);
#pragma acc serial copy(LocalInt)
  while(1);
#pragma acc kernels copy(LocalInt)
  while(1);

  // Valid cases:
#pragma acc parallel copy(LocalInt, LocalPointer, LocalArray)
  while(1);
#pragma acc parallel copy(LocalArray[2:1])
  while(1);

  Complete LocalComposite2;
#pragma acc parallel copy(LocalComposite2.ScalarMember, LocalComposite2.ScalarMember)
  while(1);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel copy(1 + IntParam)
  while(1);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel copy(+IntParam)
  while(1);

  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc parallel copy(PointerParam[2:])
  while(1);

  // expected-error@+1{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
#pragma acc parallel copy(ArrayParam[2:5])
  while(1);

  // expected-error@+2{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel copy((float*)ArrayParam[2:5])
  while(1);
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel copy((float)ArrayParam[2])
  while(1);
}

template<typename T, unsigned I, typename V>
void TemplUses(T t, T (&arrayT)[I], V TemplComp) {
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel copy(+t)
  while(true);

  // NTTP's are only valid if it is a reference to something.
  // expected-error@+2{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
  // expected-note@#TEMPL_USES_INST{{in instantiation of}}
#pragma acc parallel copy(I)
  while(true);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel copy(t, I)
  while(true);

#pragma acc parallel copy(arrayT)
  while(true);

#pragma acc parallel copy(TemplComp)
  while(true);

#pragma acc parallel copy(TemplComp.PointerMember[5])
  while(true);
 int *Pointer;
#pragma acc parallel copy(Pointer[:I])
  while(true);
#pragma acc parallel copy(Pointer[:t])
  while(true);
  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc parallel copy(Pointer[1:])
  while(true);
}

template<unsigned I, auto &NTTP_REF>
void NTTP() {
  // NTTP's are only valid if it is a reference to something.
  // expected-error@+2{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
  // expected-note@#NTTP_INST{{in instantiation of}}
#pragma acc parallel copy(I)
  while(true);

#pragma acc parallel copy(NTTP_REF)
  while(true);
}

void Inst() {
  static constexpr int NTTP_REFed = 1;
  int i;
  int Arr[5];
  Complete C;
  TemplUses(i, Arr, C); // #TEMPL_USES_INST
  NTTP<5, NTTP_REFed>(); // #NTTP_INST
}
