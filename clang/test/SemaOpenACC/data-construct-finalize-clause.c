// RUN: %clang_cc1 %s -fopenacc -verify

void Test() {
  int I;

  // expected-error@+1{{OpenACC 'finalize' clause is not valid on 'data' directive}}
#pragma acc data copyin(I) finalize
  ;
  // expected-error@+1{{OpenACC 'finalize' clause is not valid on 'enter data' directive}}
#pragma acc enter data copyin(I) finalize
  ;

  // finalize is valid only on exit data, otherwise has no other rules.
#pragma acc exit data copyout(I) finalize
  ;
  // expected-error@+1{{OpenACC 'finalize' clause is not valid on 'host_data' directive}}
#pragma acc host_data use_device(I) finalize
  ;
}
