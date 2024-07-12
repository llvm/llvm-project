// RUN: %clang_cc1 %s -fopenacc -verify

struct Incomplete;
enum SomeE{};
typedef struct IsComplete {
  struct S { int A; } CompositeMember;
  int ScalarMember;
  float ArrayMember[5];
  SomeE EnumMember;
  char *PointerMember;
} Complete;

int GlobalInt;
float GlobalArray[5];
char *GlobalPointer;
Complete GlobalComposite;

void uses(int IntParam, char *PointerParam, float ArrayParam[5], Complete CompositeParam, int &IntParamRef) {
  int LocalInt;
  char *LocalPointer;
  float LocalArray[5];
  Complete LocalComposite;

  // Check Appertainment:

#pragma acc parallel private(LocalInt)
  while(true);
#pragma acc serial private(LocalInt)
  while(true);
  // expected-error@+1{{OpenACC 'private' clause is not valid on 'kernels' directive}}
#pragma acc kernels private(LocalInt)
  while(true);

  // Valid cases:
#pragma acc parallel private(LocalInt, LocalPointer, LocalArray)
  while(true);
#pragma acc parallel private(LocalArray)
  while(true);
#pragma acc parallel private(LocalArray[2])
  while(true);
#pragma acc parallel private(LocalComposite)
  while(true);
#pragma acc parallel private(LocalComposite.EnumMember)
  while(true);
#pragma acc parallel private(LocalComposite.ScalarMember)
  while(true);
#pragma acc parallel private(LocalComposite.ArrayMember)
  while(true);
#pragma acc parallel private(LocalComposite.ArrayMember[5])
  while(true);
#pragma acc parallel private(LocalComposite.PointerMember)
  while(true);
#pragma acc parallel private(GlobalInt, GlobalArray, GlobalPointer, GlobalComposite)
  while(true);
#pragma acc parallel private(GlobalArray[2], GlobalPointer[2], GlobalComposite.CompositeMember.A)
  while(true);
#pragma acc parallel private(LocalComposite, GlobalComposite)
  while(true);
#pragma acc parallel private(IntParam, PointerParam, ArrayParam, CompositeParam) private(IntParamRef)
  while(true);
#pragma acc parallel private(PointerParam[IntParam], ArrayParam[IntParam], CompositeParam.CompositeMember.A)
  while(true);


  // Invalid cases, arbitrary expressions.
  Incomplete *I;
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel private(*I)
  while(true);
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel private(GlobalInt + IntParam)
  while(true);
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel private(+GlobalInt)
  while(true);
}

template<typename T, unsigned I, typename V>
void TemplUses(T t, T (&arrayT)[I], V TemplComp) {
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel private(+t)
  while(true);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel private(+I)
  while(true);

  // NTTP's are only valid if it is a reference to something.
  // expected-error@+2{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
  // expected-note@#TEMPL_USES_INST{{in instantiation of}}
#pragma acc parallel private(I)
  while(true);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc parallel private(t, I)
  while(true);

#pragma acc parallel private(arrayT)
  while(true);

#pragma acc parallel private(TemplComp)
  while(true);

#pragma acc parallel private(TemplComp.PointerMember[5])
  while(true);

#pragma acc parallel private(TemplComp.PointerMember[5]) private(TemplComp)
  while(true);

 int *Pointer;
#pragma acc parallel private(Pointer[:I])
  while(true);
#pragma acc parallel private(Pointer[:t])
  while(true);
  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc parallel private(Pointer[1:])
  while(true);
}

template<unsigned I, auto &NTTP_REF>
void NTTP() {
  // NTTP's are only valid if it is a reference to something.
  // expected-error@+2{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
  // expected-note@#NTTP_INST{{in instantiation of}}
#pragma acc parallel private(I)
  while(true);

#pragma acc parallel private(NTTP_REF)
  while(true);
}

struct S {
  int ThisMember;
  int ThisMemberArray[5];

  void foo();
};

void S::foo() {
#pragma acc parallel private(ThisMember, this->ThisMemberArray[1])
  while(true);

#pragma acc parallel private(ThisMemberArray[1:2])
  while(true);

#pragma acc parallel private(this)
  while(true);

#pragma acc parallel private(ThisMember, this->ThisMember)
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
