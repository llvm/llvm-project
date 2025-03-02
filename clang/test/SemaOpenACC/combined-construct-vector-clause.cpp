// RUN: %clang_cc1 %s -fopenacc -verify

template<typename T, unsigned I>
void TemplUses(T t) {

#pragma acc parallel loop vector
  for(int j = 0; j < 5; ++j);

#pragma acc parallel loop vector(I)
  for(int j = 0; j < 5; ++j);

#pragma acc parallel loop vector(length:I)
  for(int j = 0; j < 5; ++j);

#pragma acc serial loop vector
  for(int j = 0; j < 5; ++j);

  // expected-error@+1{{'length' argument on 'vector' clause is not permitted on a 'serial loop' construct}}
#pragma acc serial loop vector(I)
  for(int j = 0; j < 5; ++j);

  // expected-error@+1{{'length' argument on 'vector' clause is not permitted on a 'serial loop' construct}}
#pragma acc serial loop vector(length:I)
  for(int j = 0; j < 5; ++j);

#pragma acc kernels loop vector
  for(int j = 0; j < 5; ++j);

#pragma acc kernels loop vector(I)
  for(int j = 0; j < 5; ++j);

#pragma acc kernels loop vector(length:I)
  for(int j = 0; j < 5; ++j);

#pragma acc kernels loop vector vector_length(t)
  for(int j = 0; j < 5; ++j);

#pragma acc kernels loop vector_length(t) vector
  for(int j = 0; j < 5; ++j);

  // expected-error@+2{{'vector_length' clause not allowed on a 'kernels loop' construct that has a 'vector' clause with an argument}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop vector(I) vector_length(t)
  for(int j = 0; j < 5; ++j);

  // expected-error@+2{{'length' argument to 'vector' clause not allowed on a 'kernels loop' construct that has a 'vector_length' clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop vector_length(t) vector(I)
  for(int j = 0; j < 5; ++j);

#pragma acc parallel loop vector
  for(int j = 0; j < 5; ++j) {
    // expected-error@+4{{loop with a 'vector' clause may not exist in the region of a 'vector' clause}}
    // expected-error@+3{{loop with a 'worker' clause may not exist in the region of a 'vector' clause}}
    // expected-error@+2{{loop with a 'gang' clause may not exist in the region of a 'vector' clause}}
    // expected-note@-5 3{{previous clause is here}}
#pragma acc loop vector worker, gang
    for(int j = 0; j < 5; ++j);
  }
#pragma acc parallel loop vector
  for(int j = 0; j < 5; ++j) {
#pragma acc serial loop vector worker, gang
    for(int j = 0; j < 5; ++j);
  }

#pragma acc loop vector
  for(int j = 0; j < 5; ++j) {
#pragma acc serial loop vector worker, gang
    for(int j = 0; j < 5; ++j);
  }

#pragma acc kernels vector_length(t)
  for(int j = 0; j < 5; ++j) {
    // expected-error@+1{{'length' argument on 'vector' clause is not permitted on a 'serial loop' construct}}
#pragma acc serial loop vector(I)
    for(int j = 0; j < 5; ++j);
  }

#pragma acc kernels vector_length(t)
  for(int j = 0; j < 5; ++j) {
#pragma acc parallel loop vector(I)
    for(int j = 0; j < 5; ++j);
  }
}

void uses() {
  TemplUses<int, 5>(5);

  unsigned I;
  int t;

#pragma acc parallel loop vector
  for(int j = 0; j < 5; ++j);

#pragma acc parallel loop vector(I)
  for(int j = 0; j < 5; ++j);

#pragma acc parallel loop vector(length:I)
  for(int j = 0; j < 5; ++j);

#pragma acc serial loop vector
  for(int j = 0; j < 5; ++j);

    // expected-error@+1{{'length' argument on 'vector' clause is not permitted on a 'serial loop' construct}}
#pragma acc serial loop vector(I)
  for(int j = 0; j < 5; ++j);

    // expected-error@+1{{'length' argument on 'vector' clause is not permitted on a 'serial loop' construct}}
#pragma acc serial loop vector(length:I)
  for(int j = 0; j < 5; ++j);

#pragma acc kernels loop vector
  for(int j = 0; j < 5; ++j);

#pragma acc kernels loop vector(I)
  for(int j = 0; j < 5; ++j);

#pragma acc kernels loop vector(length:I)
  for(int j = 0; j < 5; ++j);

#pragma acc kernels loop vector vector_length(t)
  for(int j = 0; j < 5; ++j);

#pragma acc kernels loop vector_length(t) vector
  for(int j = 0; j < 5; ++j);

  // expected-error@+2{{'vector_length' clause not allowed on a 'kernels loop' construct that has a 'vector' clause with an argument}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop vector(I) vector_length(t)
  for(int j = 0; j < 5; ++j);

    // expected-error@+2{{'length' argument to 'vector' clause not allowed on a 'kernels loop' construct that has a 'vector_length' clause}}
    // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop vector_length(t) vector(I)
  for(int j = 0; j < 5; ++j);

#pragma acc parallel loop vector
  for(int j = 0; j < 5; ++j) {
    // expected-error@+4{{loop with a 'vector' clause may not exist in the region of a 'vector' clause}}
    // expected-error@+3{{loop with a 'worker' clause may not exist in the region of a 'vector' clause}}
    // expected-error@+2{{loop with a 'gang' clause may not exist in the region of a 'vector' clause}}
    // expected-note@-5 3{{previous clause is here}}
#pragma acc loop vector worker, gang
    for(int j = 0; j < 5; ++j);
  }
#pragma acc parallel loop vector
  for(int j = 0; j < 5; ++j) {
#pragma acc serial loop vector worker, gang
    for(int j = 0; j < 5; ++j);
  }

#pragma acc loop vector
  for(int j = 0; j < 5; ++j) {
#pragma acc serial loop vector worker, gang
    for(int j = 0; j < 5; ++j);
  }

#pragma acc kernels vector_length(t)
  for(int j = 0; j < 5; ++j) {
#pragma acc parallel loop vector(I)
    for(int j = 0; j < 5; ++j);
  }

#pragma acc kernels vector_length(t)
  for(int j = 0; j < 5; ++j) {
    // expected-error@+1{{'length' argument on 'vector' clause is not permitted on a 'serial loop' construct}}
#pragma acc serial loop vector(I)
    for(int j = 0; j < 5; ++j);
  }
}

