// RUN: %clang_cc1 %s -verify -fopenacc -fcxx-exceptions


void ReturnTest() {
#pragma acc parallel
  {
    (void)[]() { return; };
  }

#pragma acc parallel
  {
    try {}
    catch(...){
      return; // expected-error{{invalid return out of OpenACC Compute Construct}}
    }
  }
}
