// RUN: %clang_cc1 %s -fopenacc -verify

template<typename Int, typename NotInt, typename ConvertsToInt>
void TemplUses(Int I, NotInt NI, ConvertsToInt CTI) {
#pragma acc loop vector(I)
  for(int j = 0; j < 5; ++j);

#pragma acc parallel
#pragma acc loop vector(length: I)
  for(int j = 0; j < 5; ++j);

#pragma acc kernels
#pragma acc loop vector(CTI)
  for(int j = 0; j < 5; ++j);

  // expected-error@+2{{OpenACC clause 'vector' requires expression of integer type ('NoConvert' invalid)}}
#pragma acc kernels
#pragma acc loop vector(length: NI)
  for(int j = 0; j < 5; ++j);

  // expected-error@+2{{'length' argument on 'vector' clause is not permitted on a 'loop' construct associated with a 'serial' compute construct}}
#pragma acc serial
#pragma acc loop vector(length: I)
  for(int j = 0; j < 5; ++j);

  // expected-error@+3{{'length' argument to 'vector' clause not allowed on a 'loop' construct associated with a 'kernels' construct that has a 'vector_length' clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels vector_length(I)
#pragma acc loop vector(length: CTI)
  for(int j = 0; j < 5; ++j);

#pragma acc loop vector
  for(int i = 0; i < 5; ++i) {
    for(int j = 0; j < 5; ++j);
    // expected-error@+2{{loop with a 'vector' clause may not exist in the region of a 'vector' clause}}
    // expected-note@-4{{previous clause is here}}
#pragma acc loop vector
    for(int j = 0; j < 5; ++j);
    for(int j = 0; j < 5; ++j);
  }

#pragma acc loop vector
  for(int i = 0; i < 5; ++i) {
    for(int j = 0; j < 5; ++j);
    // expected-error@+4{{loop with a 'vector' clause may not exist in the region of a 'vector' clause}}
    // expected-error@+3{{loop with a 'worker' clause may not exist in the region of a 'vector' clause}}
    // expected-error@+2{{loop with a 'gang' clause may not exist in the region of a 'vector' clause}}
    // expected-note@-6 3{{previous clause is here}}
#pragma acc loop vector, worker, gang
    for(int j = 0; j < 5; ++j);
    for(int j = 0; j < 5; ++j);
  }

#pragma acc loop vector
  for(int i = 0; i < 5; ++i) {
#pragma acc serial
#pragma acc loop vector
    for(int j = 0; j < 5; ++j);
  }
}

struct NoConvert{};
struct Converts{
  operator int();
};

void uses() {
  TemplUses(5, NoConvert{}, Converts{}); // expected-note{{in instantiation of function template specialization}}

  unsigned i;
  NoConvert NI;
  Converts CTI;

#pragma acc loop vector(i)
  for(int j = 0; j < 5; ++j);

#pragma acc parallel
#pragma acc loop vector(length: i)
  for(int j = 0; j < 5; ++j);

#pragma acc kernels
#pragma acc loop vector(CTI)
  for(int j = 0; j < 5; ++j);

  // expected-error@+2{{OpenACC clause 'vector' requires expression of integer type ('NoConvert' invalid)}}
#pragma acc kernels
#pragma acc loop vector(length: NI)
  for(int j = 0; j < 5; ++j);

  // expected-error@+2{{'length' argument on 'vector' clause is not permitted on a 'loop' construct associated with a 'serial' compute construct}}
#pragma acc serial
#pragma acc loop vector(length: i)
  for(int j = 0; j < 5; ++j);

  // expected-error@+3{{'length' argument to 'vector' clause not allowed on a 'loop' construct associated with a 'kernels' construct that has a 'vector_length' clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels vector_length(i)
#pragma acc loop vector(length: i)
  for(int j = 0; j < 5; ++j);

#pragma acc loop vector
  for(int i = 0; i < 5; ++i) {
    for(int j = 0; j < 5; ++j);
    // expected-error@+2{{loop with a 'vector' clause may not exist in the region of a 'vector' clause}}
    // expected-note@-4{{previous clause is here}}
#pragma acc loop vector
    for(int j = 0; j < 5; ++j);
    for(int j = 0; j < 5; ++j);
  }

#pragma acc loop vector
  for(int i = 0; i < 5; ++i) {
#pragma acc serial
#pragma acc loop vector
    for(int j = 0; j < 5; ++j);
  }

#pragma acc loop vector
  for(int i = 0; i < 5; ++i) {
    for(int j = 0; j < 5; ++j);
    // expected-error@+4{{loop with a 'vector' clause may not exist in the region of a 'vector' clause}}
    // expected-error@+3{{loop with a 'worker' clause may not exist in the region of a 'vector' clause}}
    // expected-error@+2{{loop with a 'gang' clause may not exist in the region of a 'vector' clause}}
    // expected-note@-6 3{{previous clause is here}}
#pragma acc loop vector, worker, gang
    for(int j = 0; j < 5; ++j);
    for(int j = 0; j < 5; ++j);
  }

#pragma acc loop vector
  for(int i = 0; i < 5; ++i) {
#pragma acc serial
#pragma acc loop vector, worker, gang
    for(int j = 0; j < 5; ++j);
  }
}
