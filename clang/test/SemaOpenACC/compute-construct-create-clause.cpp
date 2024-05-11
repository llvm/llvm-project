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
#pragma acc parallel create(LocalInt)
  while(1);
#pragma acc serial create(LocalInt)
  while(1);
#pragma acc kernels create(LocalInt)
  while(1);

  // Valid cases:
#pragma acc parallel create(LocalInt, LocalPointer, LocalArray)
  while(1);
#pragma acc parallel create(LocalArray[2:1])
  while(1);

  Complete LocalComposite2;
#pragma acc parallel create(LocalComposite2.ScalarMember, LocalComposite2.ScalarMember)
  while(1);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, or composite variable member}}
#pragma acc parallel create(1 + IntParam)
  while(1);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, or composite variable member}}
#pragma acc parallel create(+IntParam)
  while(1);

  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc parallel create(PointerParam[2:])
  while(1);

  // expected-error@+1{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
#pragma acc parallel create(ArrayParam[2:5])
  while(1);

  // expected-error@+2{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, or composite variable member}}
#pragma acc parallel create((float*)ArrayParam[2:5])
  while(1);
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, or composite variable member}}
#pragma acc parallel create((float)ArrayParam[2])
  while(1);
}

template<typename T, unsigned I, typename V>
void TemplUses(T t, T (&arrayT)[I], V TemplComp) {
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, or composite variable member}}
#pragma acc parallel create(+t)
  while(true);

  // NTTP's are only valid if it is a reference to something.
  // expected-error@+2{{OpenACC variable is not a valid variable name, sub-array, array element, or composite variable member}}
  // expected-note@#TEMPL_USES_INST{{in instantiation of}}
#pragma acc parallel create(I)
  while(true);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, or composite variable member}}
#pragma acc parallel create(t, I)
  while(true);

#pragma acc parallel create(arrayT)
  while(true);

#pragma acc parallel create(TemplComp)
  while(true);

#pragma acc parallel create(TemplComp.PointerMember[5])
  while(true);
 int *Pointer;
#pragma acc parallel create(Pointer[:I])
  while(true);
#pragma acc parallel create(Pointer[:t])
  while(true);
  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc parallel create(Pointer[1:])
  while(true);
}

template<unsigned I, auto &NTTP_REF>
void NTTP() {
  // NTTP's are only valid if it is a reference to something.
  // expected-error@+2{{OpenACC variable is not a valid variable name, sub-array, array element, or composite variable member}}
  // expected-note@#NTTP_INST{{in instantiation of}}
#pragma acc parallel create(I)
  while(true);

#pragma acc parallel create(NTTP_REF)
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
