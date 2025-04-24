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
#pragma acc host_data use_device(LocalInt)

  // Valid cases:
#pragma acc host_data use_device(LocalInt, LocalPointer, LocalArray)
  ;
  ;
  // expected-error@+2{{OpenACC variable in 'use_device' clause is not a valid variable name or array name}}
  // expected-error@+1{{OpenACC variable in 'use_device' clause is not a valid variable name or array name}}
#pragma acc host_data use_device(LocalComposite.ScalarMember, LocalComposite.ScalarMember)
  ;

  // expected-error@+1{{OpenACC variable in 'use_device' clause is not a valid variable name or array name}}
#pragma acc host_data use_device(LocalArray[2:1])

  // expected-error@+1{{OpenACC variable in 'use_device' clause is not a valid variable name or array name}}
#pragma acc host_data use_device(1 + IntParam)
  ;

  // expected-error@+1{{OpenACC variable in 'use_device' clause is not a valid variable name or array name}}
#pragma acc host_data use_device(+IntParam)
  ;

  // expected-error@+2{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
  // expected-error@+1{{OpenACC variable in 'use_device' clause is not a valid variable name or array name}}
#pragma acc host_data use_device(PointerParam[2:])
  ;

  // expected-error@+2{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
  // expected-error@+1{{OpenACC variable in 'use_device' clause is not a valid variable name or array name}}
#pragma acc host_data use_device(ArrayParam[2:5])
  ;

  // expected-error@+2{{OpenACC sub-array specified range [2:5] would be out of the range of the subscripted array size of 5}}
  // expected-error@+1{{OpenACC variable in 'use_device' clause is not a valid variable name or array name}}
#pragma acc host_data use_device((float*)ArrayParam[2:5])
  ;
  // expected-error@+1{{OpenACC variable in 'use_device' clause is not a valid variable name or array name}}
#pragma acc host_data use_device((float)ArrayParam[2])
  ;

  // expected-error@+2{{OpenACC 'data' construct must have at least one 'copy', 'copyin', 'copyout', 'create', 'no_create', 'present', 'deviceptr', 'attach' or 'default' clause}}
  // expected-error@+1{{OpenACC 'use_device' clause is not valid on 'data' directive}}
#pragma acc data use_device(LocalInt)
  ;
  // expected-error@+2{{OpenACC 'enter data' construct must have at least one 'copyin', 'create' or 'attach' clause}}
  // expected-error@+1{{OpenACC 'use_device' clause is not valid on 'enter data' directive}}
#pragma acc enter data use_device(LocalInt)
  // expected-error@+2{{OpenACC 'exit data' construct must have at least one 'copyout', 'delete' or 'detach' clause}}
  // expected-error@+1{{OpenACC 'use_device' clause is not valid on 'exit data' directive}}
#pragma acc exit data use_device(LocalInt)
}
