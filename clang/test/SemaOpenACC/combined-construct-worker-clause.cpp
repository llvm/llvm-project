// RUN: %clang_cc1 %s -fopenacc -verify

template<unsigned I>
void TemplUses() {

#pragma acc parallel loop worker
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected expression}}
#pragma acc parallel loop worker()
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{'num' argument on 'worker' clause is not permitted on a 'parallel loop' construct}}
#pragma acc parallel loop worker(I)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{'num' argument on 'worker' clause is not permitted on a 'parallel loop' construct}}
#pragma acc parallel loop worker(num:I)
  for(int i = 0; i < 5; ++i);

#pragma acc serial loop worker
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected expression}}
#pragma acc serial loop worker()
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{'num' argument on 'worker' clause is not permitted on a 'serial loop' construct}}
#pragma acc serial loop worker(I)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{'num' argument on 'worker' clause is not permitted on a 'serial loop' construct}}
#pragma acc serial loop worker(num:I)
  for(int i = 0; i < 5; ++i);

#pragma acc kernels loop worker(I)
  for(int i = 0; i < 5; ++i);

#pragma acc kernels loop worker(num:I)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{'num' argument to 'worker' clause not allowed on a 'kernels loop' construct that has a 'num_workers' clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop num_workers(1) worker(num:I)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{'num_workers' clause not allowed on a 'kernels loop' construct that has a 'worker' clause with an argument}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop worker(num:I) num_workers(1)
  for(int i = 0; i < 5; ++i);
}

void NormalUses() {
  TemplUses<4>();

  int I;
#pragma acc parallel loop worker
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected expression}}
#pragma acc parallel loop worker()
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{'num' argument on 'worker' clause is not permitted on a 'parallel loop' construct}}
#pragma acc parallel loop worker(I)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{'num' argument on 'worker' clause is not permitted on a 'parallel loop' construct}}
#pragma acc parallel loop worker(num:I)
  for(int i = 0; i < 5; ++i);

#pragma acc serial loop worker
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected expression}}
#pragma acc serial loop worker()
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{'num' argument on 'worker' clause is not permitted on a 'serial loop' construct}}
#pragma acc serial loop worker(I)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{'num' argument on 'worker' clause is not permitted on a 'serial loop' construct}}
#pragma acc serial loop worker(num:I)
  for(int i = 0; i < 5; ++i);

#pragma acc kernels loop worker(I)
  for(int i = 0; i < 5; ++i);

#pragma acc kernels loop worker(num:I)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{'num' argument to 'worker' clause not allowed on a 'kernels loop' construct that has a 'num_workers' clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop num_workers(1) worker(num:I)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{'num_workers' clause not allowed on a 'kernels loop' construct that has a 'worker' clause with an argument}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop worker(num:I) num_workers(1)
  for(int i = 0; i < 5; ++i);

  // OK, kernels loop is a new compute construct
#pragma acc kernels num_workers(1)
  for(int i = 0; i < 5; ++i) {
#pragma acc kernels loop worker(num:1)
  for(int i = 0; i < 5; ++i);
  }
#pragma acc kernels loop num_workers(1)
  for(int i = 0; i < 5; ++i) {
#pragma acc kernels loop worker(num:1)
  for(int i = 0; i < 5; ++i);
  }

#pragma acc kernels loop num_workers(1)
  for(int i = 0; i < 5; ++i) {
  // expected-error@+2{{'num' argument to 'worker' clause not allowed on a 'loop' construct associated with a 'kernels loop' construct that has a 'num_workers' clause}}
  // expected-note@-3{{previous clause is here}}
#pragma acc loop worker(num:1)
  for(int i = 0; i < 5; ++i);
  }

#pragma acc parallel loop worker
  for(int i = 0; i < 5; ++i) {
    // expected-error@+4{{loop with a 'gang' clause may not exist in the region of a 'worker' clause}}
    // expected-note@-3{{previous clause is here}}
    // expected-error@+2{{loop with a 'worker' clause may not exist in the region of a 'worker' clause}}
    // expected-note@-5{{previous clause is here}}
#pragma acc loop gang, worker, vector
  for(int i = 0; i < 5; ++i);
  }
#pragma acc kernels loop worker
  for(int i = 0; i < 5; ++i) {
    // expected-error@+4{{loop with a 'gang' clause may not exist in the region of a 'worker' clause}}
    // expected-note@-3{{previous clause is here}}
    // expected-error@+2{{loop with a 'worker' clause may not exist in the region of a 'worker' clause}}
    // expected-note@-5{{previous clause is here}}
#pragma acc loop gang, worker, vector
  for(int i = 0; i < 5; ++i);
  }
#pragma acc serial loop worker
  for(int i = 0; i < 5; ++i) {
    // expected-error@+4{{loop with a 'gang' clause may not exist in the region of a 'worker' clause}}
    // expected-note@-3{{previous clause is here}}
    // expected-error@+2{{loop with a 'worker' clause may not exist in the region of a 'worker' clause}}
    // expected-note@-5{{previous clause is here}}
#pragma acc loop gang, worker, vector
  for(int i = 0; i < 5; ++i);
  }
}
