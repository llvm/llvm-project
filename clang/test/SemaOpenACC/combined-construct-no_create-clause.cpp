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

  Complete LocalComposite2;
#pragma acc parallel loop no_create(LocalComposite2.ScalarMember, LocalComposite2.ScalarMember)
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
}

template<typename T, unsigned I, typename V>
void TemplUses(T t, T (&arrayT)[I], V TemplComp) {
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop no_create(+t)
  for (unsigned i = 0; i < 5; ++i);

  // NTTP's are only valid if it is a reference to something.
  // expected-error@+2{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
  // expected-note@#TEMPL_USES_INST{{in instantiation of}}
#pragma acc parallel loop no_create(I)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop no_create(t, I)
  for (unsigned i = 0; i < 5; ++i);

#pragma acc parallel loop no_create(arrayT)
  for (unsigned i = 0; i < 5; ++i);

#pragma acc parallel loop no_create(TemplComp)
  for (unsigned i = 0; i < 5; ++i);

#pragma acc parallel loop no_create(TemplComp.PointerMember[5])
  for (unsigned i = 0; i < 5; ++i);
 int *Pointer;
#pragma acc parallel loop no_create(Pointer[:I])
  for (unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop no_create(Pointer[:t])
  for (unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc parallel loop no_create(Pointer[1:])
  for (unsigned i = 0; i < 5; ++i);
}

template<unsigned I, auto &NTTP_REF>
void NTTP() {
  // NTTP's are only valid if it is a reference to something.
  // expected-error@+2{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
  // expected-note@#NTTP_INST{{in instantiation of}}
#pragma acc parallel loop no_create(I)
  for (unsigned i = 0; i < 5; ++i);

#pragma acc parallel loop no_create(NTTP_REF)
  for (unsigned i = 0; i < 5; ++i);
}

void Inst() {
  static constexpr int NTTP_REFed = 1;
  int i;
  int Arr[5];
  Complete C;
  TemplUses(i, Arr, C); // #TEMPL_USES_INST
  NTTP<5, NTTP_REFed>(); // #NTTP_INST
}
