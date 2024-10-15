// RUN: %clang_cc1 %s -fopenacc -verify

template<unsigned I, typename NotInt, typename ConvertsToInt, typename Int>
void TemplUses(NotInt NI, ConvertsToInt CTI, Int IsI) {
  int i;

  // expected-error@+1{{'num' argument on 'worker' clause is not permitted on an orphaned 'loop' construct}}
#pragma acc loop worker(i)
  for(;;);

  // expected-error@+1{{'num' argument on 'worker' clause is not permitted on an orphaned 'loop' construct}}
#pragma acc loop worker(num:IsI)
  for(;;);

#pragma acc kernels
#pragma acc loop worker
  for(;;);

#pragma acc kernels
#pragma acc loop worker(i)
  for(;;);

#pragma acc kernels
#pragma acc loop worker(CTI)
  for(;;);

#pragma acc kernels
#pragma acc loop worker(IsI)
  for(;;);

#pragma acc kernels
#pragma acc loop worker(I)
  for(;;);

#pragma acc kernels
  // expected-error@+1{{OpenACC clause 'worker' requires expression of integer type ('NoConvert' invalid)}}
#pragma acc loop worker(NI)
  for(;;);

#pragma acc kernels
#pragma acc loop worker(num:i)
  for(;;);

  // expected-error@+3{{'num' argument to 'worker' clause not allowed on a 'loop' construct associated with a 'kernels' construct that has a 'num_workers' clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels num_workers(IsI)
#pragma acc loop worker(num:CTI)
  for(;;);
 for(;;);
}

struct NoConvert{};
struct Converts{
  operator int();
};

void uses() {
  TemplUses<3>(NoConvert{}, Converts{}, 5); // expected-note{{in instantiation of function template specialization}}

#pragma acc loop worker
  for(;;);

#pragma acc parallel
#pragma acc loop worker
  for(;;);

  int i;

  // expected-error@+1{{'num' argument on 'worker' clause is not permitted on an orphaned 'loop' construct}}
#pragma acc loop worker(i)
  for(;;);

  // expected-error@+2{{'num' argument on 'worker' clause is not permitted on a 'loop' construct associated with a 'parallel' compute construct}}
#pragma acc parallel
#pragma acc loop worker(i)
  for(;;);

  // expected-error@+1{{'num' argument on 'worker' clause is not permitted on an orphaned 'loop' construct}}
#pragma acc loop worker(num:i)
  for(;;);

  // expected-error@+2{{'num' argument on 'worker' clause is not permitted on a 'loop' construct associated with a 'parallel' compute construct}}
#pragma acc parallel
#pragma acc loop worker(num:i)
  for(;;);

#pragma acc serial
#pragma acc loop worker
  for(;;);

  // expected-error@+2{{'num' argument on 'worker' clause is not permitted on a 'loop' construct associated with a 'serial' compute construct}}
#pragma acc serial
#pragma acc loop worker(i)
  for(;;);

  // expected-error@+2{{'num' argument on 'worker' clause is not permitted on a 'loop' construct associated with a 'serial' compute construct}}
#pragma acc serial
#pragma acc loop worker(num:i)
  for(;;);

#pragma acc kernels
#pragma acc loop worker
  for(;;);

#pragma acc kernels
#pragma acc loop worker(i)
  for(;;);

  Converts Cvts;

#pragma acc kernels
#pragma acc loop worker(Cvts)
  for(;;);

  NoConvert NoCvts;

#pragma acc kernels
  // expected-error@+1{{OpenACC clause 'worker' requires expression of integer type ('NoConvert' invalid)}}
#pragma acc loop worker(NoCvts)
  for(;;);

#pragma acc kernels
#pragma acc loop worker(num:i)
  for(;;);

  // expected-error@+3{{'num' argument to 'worker' clause not allowed on a 'loop' construct associated with a 'kernels' construct that has a 'num_workers' clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels num_workers(i)
#pragma acc loop worker(num:i)
  for(;;);

#pragma acc loop worker
  for(;;) {
    // expected-error@+3{{loop with a 'worker' clause may not exist in the region of a 'worker' clause}}
    // expected-error@+2{{loop with a 'gang' clause may not exist in the region of a 'worker' clause}}
    // expected-note@-4 2{{previous clause is here}}
#pragma acc loop worker, gang
  for(;;) {}
  }

#pragma acc loop worker
  for(;;) {
#pragma acc parallel
#pragma acc loop worker, gang
  for(;;) {}
  }


#pragma acc parallel
#pragma acc loop worker
  for(;;) {
    // expected-error@+3{{loop with a 'worker' clause may not exist in the region of a 'worker' clause}}
    // expected-error@+2{{loop with a 'gang' clause may not exist in the region of a 'worker' clause}}
    // expected-note@-4 2{{previous clause is here}}
#pragma acc loop worker, gang
  for(;;) {}
  }

#pragma acc parallel
#pragma acc loop worker
  for(;;) {
#pragma acc parallel
#pragma acc loop worker, gang
  for(;;) {}
  }

#pragma acc serial
#pragma acc loop worker
  for(;;) {
    // expected-error@+3{{loop with a 'worker' clause may not exist in the region of a 'worker' clause}}
    // expected-error@+2{{loop with a 'gang' clause may not exist in the region of a 'worker' clause}}
    // expected-note@-4 2{{previous clause is here}}
#pragma acc loop worker, gang
  for(;;) {}
  }

#pragma acc serial
#pragma acc loop worker
  for(;;) {
#pragma acc parallel
#pragma acc loop worker, gang
  for(;;) {}
  }

#pragma acc kernels
#pragma acc loop worker
  for(;;) {
    // expected-error@+3{{loop with a 'worker' clause may not exist in the region of a 'worker' clause}}
    // expected-error@+2{{loop with a 'gang' clause may not exist in the region of a 'worker' clause}}
    // expected-note@-4 2{{previous clause is here}}
#pragma acc loop worker, gang
  for(;;) {}
  }

#pragma acc kernels
#pragma acc loop worker
  for(;;) {
#pragma acc parallel
#pragma acc loop worker, gang
  for(;;) {}
  }
}
