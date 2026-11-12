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

#pragma acc loop private(LocalInt)
  for(int i = 0; i < 5; ++i);

  // Valid cases:
#pragma acc loop private(LocalInt, LocalPointer, LocalArray)
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(LocalArray)
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
#pragma acc loop private(IntParam, PointerParam, ArrayParam, CompositeParam) private(IntParamRef)
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(PointerParam[IntParam], ArrayParam[IntParam], CompositeParam.CompositeMember.A)
  for(int i = 0; i < 5; ++i);


  // Invalid cases, arbitrary expressions.
  Incomplete *I;
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc loop private(*I)
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc loop private(GlobalInt + IntParam)
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc loop private(+GlobalInt)
  for(int i = 0; i < 5; ++i);
}

template<typename T, unsigned I, typename V>
void TemplUses(T t, T (&arrayT)[I], V TemplComp) {
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc loop private(+t)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc loop private(+I)
  for(int i = 0; i < 5; ++i);

  // NTTP's are only valid if it is a reference to something.
  // expected-error@+2{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
  // expected-note@#TEMPL_USES_INST{{in instantiation of}}
#pragma acc loop private(I)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc loop private(t, I)
  for(int i = 0; i < 5; ++i);

#pragma acc loop private(arrayT)
  for(int i = 0; i < 5; ++i);

#pragma acc loop private(TemplComp)
  for(int i = 0; i < 5; ++i);

#pragma acc loop private(TemplComp.PointerMember[5])
  for(int i = 0; i < 5; ++i);

#pragma acc loop private(TemplComp.PointerMember[5]) private(TemplComp)
  for(int i = 0; i < 5; ++i);

 int *Pointer;
#pragma acc loop private(Pointer[:I])
  for(int i = 0; i < 5; ++i);
#pragma acc loop private(Pointer[:t])
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc loop private(Pointer[1:])
  for(int i = 0; i < 5; ++i);
}

template<unsigned I, auto &NTTP_REF>
void NTTP() {
  // NTTP's are only valid if it is a reference to something.
  // expected-error@+2{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
  // expected-note@#NTTP_INST{{in instantiation of}}
#pragma acc loop private(I)
  for(int i = 0; i < 5; ++i);

#pragma acc loop private(NTTP_REF)
  for(int i = 0; i < 5; ++i);
}

struct S {
  int ThisMember;
  int ThisMemberArray[5];

  void foo();
};

void S::foo() {
#pragma acc loop private(ThisMember, this->ThisMemberArray[1])
  for(int i = 0; i < 5; ++i);

#pragma acc loop private(ThisMemberArray[1:2])
  for(int i = 0; i < 5; ++i);

#pragma acc loop private(this)
  for(int i = 0; i < 5; ++i);

#pragma acc loop private(ThisMember, this->ThisMember)
  for(int i = 0; i < 5; ++i);
}

void Inst() {
  static constexpr int NTTP_REFed = 1;
  int i;
  int Arr[5];
  Complete C;
  TemplUses(i, Arr, C); // #TEMPL_USES_INST
  NTTP<5, NTTP_REFed>(); // #NTTP_INST
}
