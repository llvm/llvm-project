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
#pragma acc parallel loop firstprivate(LocalInt)
  for (int i = 5; i < 10; ++i);
#pragma acc serial loop firstprivate(LocalInt)
  for (int i = 5; i < 10; ++i);
  // expected-error@+1{{OpenACC 'firstprivate' clause is not valid on 'kernels loop' directive}}
#pragma acc kernels loop firstprivate(LocalInt)
  for (int i = 5; i < 10; ++i);

  // Valid cases:
#pragma acc parallel loop firstprivate(LocalInt, LocalPointer, LocalArray)
  for (int i = 5; i < 10; ++i);
#pragma acc serial loop firstprivate(LocalArray[2:1])
  for (int i = 5; i < 10; ++i);

  Complete LocalComposite2;
#pragma acc parallel loop firstprivate(LocalComposite2.ScalarMember, LocalComposite2.ScalarMember)
  for (int i = 5; i < 10; ++i);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop firstprivate(1 + IntParam)
  for (int i = 5; i < 10; ++i);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc serial loop firstprivate(+IntParam)
  for (int i = 5; i < 10; ++i);

  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc parallel loop firstprivate(PointerParam[2:])
  for (int i = 5; i < 10; ++i);

  // expected-error@+1{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
#pragma acc parallel loop firstprivate(ArrayParam[2:5])
  for (int i = 5; i < 10; ++i);

  // expected-error@+2{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc serial loop firstprivate((float*)ArrayParam[2:5])
  for (int i = 5; i < 10; ++i);
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop firstprivate((float)ArrayParam[2])
  for (int i = 5; i < 10; ++i);
}

template<typename T, unsigned I, typename V>
void TemplUses(T t, T (&arrayT)[I], V TemplComp) {
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel loop firstprivate(+t)
  for (int i = 5; i < 10; ++i);

  // NTTP's are only valid if it is a reference to something.
  // expected-error@+2{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
  // expected-note@#TEMPL_USES_INST{{in instantiation of}}
#pragma acc parallel loop firstprivate(I)
  for (int i = 5; i < 10; ++i);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc serial loop firstprivate(t, I)
  for (int i = 5; i < 10; ++i);

#pragma acc parallel loop firstprivate(arrayT)
  for (int i = 5; i < 10; ++i);

#pragma acc parallel loop firstprivate(TemplComp)
  for (int i = 5; i < 10; ++i);

#pragma acc parallel loop firstprivate(TemplComp.PointerMember[5])
  for (int i = 5; i < 10; ++i);
 int *Pointer;
#pragma acc serial loop firstprivate(Pointer[:I])
  for (int i = 5; i < 10; ++i);
#pragma acc parallel loop firstprivate(Pointer[:t])
  for (int i = 5; i < 10; ++i);
  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc parallel loop firstprivate(Pointer[1:])
  for (int i = 5; i < 10; ++i);
}

template<unsigned I, auto &NTTP_REF>
void NTTP() {
  // NTTP's are only valid if it is a reference to something.
  // expected-error@+2{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
  // expected-note@#NTTP_INST{{in instantiation of}}
#pragma acc parallel loop firstprivate(I)
  for (int i = 5; i < 10; ++i);

#pragma acc serial loop firstprivate(NTTP_REF)
  for (int i = 5; i < 10; ++i);
}

void Inst() {
  static constexpr int NTTP_REFed = 1;
  int i;
  int Arr[5];
  Complete C;
  TemplUses(i, Arr, C); // #TEMPL_USES_INST
  NTTP<5, NTTP_REFed>(); // #NTTP_INST
}
