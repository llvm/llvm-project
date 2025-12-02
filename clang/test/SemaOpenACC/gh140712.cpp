// RUN: %clang_cc1 %s -fopenacc -verify

void foo() {
  switch (int x = 0) {
    case 0:
#pragma acc parallel
      break; // expected-error{{invalid branch out of OpenACC Compute/Combined Construct}}
  }
}
