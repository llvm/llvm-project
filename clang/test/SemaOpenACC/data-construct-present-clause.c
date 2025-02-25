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
#pragma acc data default(none) present(LocalInt)
  ;

  // Valid cases:
#pragma acc data default(none) present(LocalInt, LocalPointer, LocalArray)
  ;
#pragma acc data default(none) present(LocalArray[2:1])
  ;

#pragma acc data default(none) present(LocalComposite.ScalarMember, LocalComposite.ScalarMember)
  ;

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data default(none) present(1 + IntParam)
  ;

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data default(none) present(+IntParam)
  ;

  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc data default(none) present(PointerParam[2:])
  ;

  // expected-error@+1{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
#pragma acc data default(none) present(ArrayParam[2:5])
  ;

  // expected-error@+2{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data default(none) present((float*)ArrayParam[2:5])
  ;
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, member of a composite variable, or composite variable member}}
#pragma acc data default(none) present((float)ArrayParam[2])
  ;

  // expected-error@+1{{OpenACC 'present' clause is not valid on 'enter data' directive}}
#pragma acc enter data copyin(LocalInt) present(LocalInt)
  // expected-error@+1{{OpenACC 'present' clause is not valid on 'exit data' directive}}
#pragma acc exit data copyout(LocalInt) present(LocalInt)
  // expected-error@+1{{OpenACC 'present' clause is not valid on 'host_data' directive}}
#pragma acc host_data use_device(LocalInt) present(LocalInt)
  ;
}
