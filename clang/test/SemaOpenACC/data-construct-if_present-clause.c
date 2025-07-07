// RUN: %clang_cc1 %s -fopenacc -verify

void Test() {
  int I;

  // expected-error@+1{{OpenACC 'if_present' clause is not valid on 'data' directive}}
#pragma acc data copyin(I) if_present
  ;
  // expected-error@+1{{OpenACC 'if_present' clause is not valid on 'enter data' directive}}
#pragma acc enter data copyin(I) if_present
  ;

  // expected-error@+1{{OpenACC 'if_present' clause is not valid on 'exit data' directive}}
#pragma acc exit data copyout(I) if_present
  ;
#pragma acc host_data use_device(I) if_present
  ;
}
