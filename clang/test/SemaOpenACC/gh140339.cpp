// RUN: %clang_cc1 %s -fopenacc -verify

void foo() {
#pragma acc parallel loop
  for (int i = 0; i < 5; ++i) {
#pragma acc loop vector(1)
    for(int j = 0; j < 5; ++j);
  }

#pragma acc serial loop
  for (int i = 0; i < 5; ++i) {
    // expected-error@+1{{'length' argument on 'vector' clause is not permitted on a 'loop' construct associated with a 'serial loop' compute construct}}
#pragma acc loop vector(1)
    for(int j = 0; j < 5; ++j);
  }

#pragma acc kernels loop
  for (int i = 0; i < 5; ++i) {
#pragma acc loop vector(1)
    for(int j = 0; j < 5; ++j);
  }
}
