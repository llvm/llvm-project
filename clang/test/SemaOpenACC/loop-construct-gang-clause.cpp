// RUN: %clang_cc1 %s -fopenacc -verify

struct S{};
struct Converts{
  operator int();
};

template<typename T, unsigned Zero, unsigned Two, unsigned Four>
void ParallelOrOrphanTempl() {
  T i;
  // expected-error@+1{{'num' argument on 'gang' clause is not permitted on an orphaned 'loop' construct}}
#pragma acc loop gang(i)
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{'num' argument on 'gang' clause is not permitted on an orphaned 'loop' construct}}
#pragma acc loop gang(num:i)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{'num' argument on 'gang' clause is not permitted on a 'loop' construct associated with a 'parallel' compute construct}}
#pragma acc parallel
#pragma acc loop gang(i)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{'num' argument on 'gang' clause is not permitted on a 'loop' construct associated with a 'parallel' compute construct}}
#pragma acc parallel
#pragma acc loop gang(num:i)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{argument to 'gang' clause dimension must be a constant expression}}
#pragma acc loop gang(dim:i)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{argument to 'gang' clause dimension must be a constant expression}}
#pragma acc parallel
#pragma acc loop gang(dim:i)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{argument to 'gang' clause dimension must be 1, 2, or 3: evaluated to 0}}
#pragma acc loop gang(dim:Zero)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{argument to 'gang' clause dimension must be 1, 2, or 3: evaluated to 0}}
#pragma acc parallel
#pragma acc loop gang(dim:Zero)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{argument to 'gang' clause dimension must be 1, 2, or 3: evaluated to 4}}
#pragma acc loop gang(dim:Four)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{argument to 'gang' clause dimension must be 1, 2, or 3: evaluated to 4}}
#pragma acc parallel
#pragma acc loop gang(dim:Four)
  for(int i = 0; i < 5; ++i);

#pragma acc loop gang(static:i) gang(dim:Two)
  for(int i = 0; i < 5; ++i);

#pragma acc parallel
#pragma acc loop gang(dim:Two) gang(static:*)
  for(int i = 0; i < 5; ++i);

#pragma acc parallel
#pragma acc loop gang(dim:Two, static:i)
  for(int i = 0; i < 5; ++i);

  // expected-error@+4{{OpenACC 'gang' clause may have at most one 'static' argument}}
  // expected-note@+3{{previous expression is here}}
  // expected-error@+2{{OpenACC 'gang' clause may have at most one 'dim' argument}}
  // expected-note@+1{{previous expression is here}}
#pragma acc loop gang(static:i, static:i, dim:Two, dim:1)
  for(int i = 0; i < 5; ++i);
}

void ParallelOrOrphan() {
  ParallelOrOrphanTempl<int, 0, 2, 4>(); // expected-note{{in instantiation of function template}}

  int i;
  // expected-error@+1{{'num' argument on 'gang' clause is not permitted on an orphaned 'loop' construct}}
#pragma acc loop gang(i)
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{'num' argument on 'gang' clause is not permitted on an orphaned 'loop' construct}}
#pragma acc loop gang(num:i)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{'num' argument on 'gang' clause is not permitted on a 'loop' construct associated with a 'parallel' compute construct}}
#pragma acc parallel
#pragma acc loop gang(i)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{'num' argument on 'gang' clause is not permitted on a 'loop' construct associated with a 'parallel' compute construct}}
#pragma acc parallel
#pragma acc loop gang(num:i)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{argument to 'gang' clause dimension must be a constant expression}}
#pragma acc loop gang(dim:i)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{argument to 'gang' clause dimension must be a constant expression}}
#pragma acc parallel
#pragma acc loop gang(dim:i)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{argument to 'gang' clause dimension must be 1, 2, or 3: evaluated to 0}}
#pragma acc loop gang(dim:0)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{argument to 'gang' clause dimension must be 1, 2, or 3: evaluated to 0}}
#pragma acc parallel
#pragma acc loop gang(dim:0)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{argument to 'gang' clause dimension must be 1, 2, or 3: evaluated to 4}}
#pragma acc loop gang(dim:4)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{argument to 'gang' clause dimension must be 1, 2, or 3: evaluated to 4}}
#pragma acc parallel
#pragma acc loop gang(dim:4)
  for(int i = 0; i < 5; ++i);

#pragma acc loop gang(static:i) gang(dim:2)
  for(int i = 0; i < 5; ++i);

#pragma acc parallel
#pragma acc loop gang(dim:2) gang(static:i)
  for(int i = 0; i < 5; ++i);

  S s;
  // expected-error@+2{{OpenACC clause 'gang' requires expression of integer type ('S' invalid)}}
#pragma acc parallel
#pragma acc loop gang(dim:2) gang(static:s)
  for(int i = 0; i < 5; ++i);

  Converts C;
#pragma acc parallel
#pragma acc loop gang(dim:2) gang(static:C)
  for(int i = 0; i < 5; ++i);
}

template<typename SomeS, typename SomeC, typename Int>
void StaticIsIntegralTempl() {
  SomeS s;
  // expected-error@+2{{OpenACC clause 'gang' requires expression of integer type ('S' invalid)}}
#pragma acc parallel
#pragma acc loop gang(dim:2) gang(static:s)
  for(int i = 0; i < 5; ++i);

  SomeC C;
#pragma acc parallel
#pragma acc loop gang(dim:2) gang(static:C)
  for(int i = 0; i < 5; ++i);
  Int i;
#pragma acc parallel
#pragma acc loop gang(dim:2) gang(static:i)
  for(int i = 0; i < 5; ++i);

#pragma acc parallel
#pragma acc loop gang(dim:2) gang(static:*)
  for(int i = 0; i < 5; ++i);
}

void StaticIsIntegral() {
  StaticIsIntegralTempl<S, Converts, int>();// expected-note{{in instantiation of function template}}

  S s;
  // expected-error@+2{{OpenACC clause 'gang' requires expression of integer type ('S' invalid)}}
#pragma acc parallel
#pragma acc loop gang(dim:2) gang(static:s)
  for(int i = 0; i < 5; ++i);

  Converts C;
#pragma acc parallel
#pragma acc loop gang(dim:2) gang(static:C)
  for(int i = 0; i < 5; ++i);
}

template<unsigned I>
void SerialTempl() {
  // expected-error@+2{{'num' argument on 'gang' clause is not permitted on a 'loop' construct associated with a 'serial' compute construct}}
#pragma acc serial
#pragma acc loop gang(I)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{'num' argument on 'gang' clause is not permitted on a 'loop' construct associated with a 'serial' compute construct}}
#pragma acc serial
#pragma acc loop gang(num:I)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{'dim' argument on 'gang' clause is not permitted on a 'loop' construct associated with a 'serial' compute construct}}
#pragma acc serial
#pragma acc loop gang(dim:I)
  for(int i = 0; i < 5; ++i);

#pragma acc serial
#pragma acc loop gang(static:I)
  for(int i = 0; i < 5; ++i);
}

void Serial() {
  SerialTempl<2>();

  // expected-error@+2{{'num' argument on 'gang' clause is not permitted on a 'loop' construct associated with a 'serial' compute construct}}
#pragma acc serial
#pragma acc loop gang(1)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{'num' argument on 'gang' clause is not permitted on a 'loop' construct associated with a 'serial' compute construct}}
#pragma acc serial
#pragma acc loop gang(num:1)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{'dim' argument on 'gang' clause is not permitted on a 'loop' construct associated with a 'serial' compute construct}}
#pragma acc serial
#pragma acc loop gang(dim:1)
  for(int i = 0; i < 5; ++i);

#pragma acc serial
#pragma acc loop gang(static:1)
  for(int i = 0; i < 5; ++i);

  int i;

#pragma acc serial
#pragma acc loop gang(static:i)
  for(int i = 0; i < 5; ++i);
}

template<typename T>
void KernelsTempl() {
  T t;
  // expected-error@+2{{'dim' argument on 'gang' clause is not permitted on a 'loop' construct associated with a 'kernels' compute construct}}
#pragma acc kernels
#pragma acc loop gang(dim:t)
  for(int i = 0; i < 5; ++i);

#pragma acc kernels
#pragma acc loop gang(static:t)
  for(int i = 0; i < 5; ++i);

  // expected-error@+3{{'num' argument to 'gang' clause not allowed on a 'loop' construct associated with a 'kernels' construct that has a 'num_gangs' clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels num_gangs(t)
#pragma acc loop gang(t)
  for(int i = 0; i < 5; ++i);

  // expected-error@+3{{'num' argument to 'gang' clause not allowed on a 'loop' construct associated with a 'kernels' construct that has a 'num_gangs' clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels num_gangs(t)
#pragma acc loop gang(num:t)
  for(int i = 0; i < 5; ++i);
}

void Kernels() {
  KernelsTempl<unsigned>();

  // expected-error@+2{{'dim' argument on 'gang' clause is not permitted on a 'loop' construct associated with a 'kernels' compute construct}}
#pragma acc kernels
#pragma acc loop gang(dim:1)
  for(int i = 0; i < 5; ++i);
  unsigned t;
#pragma acc kernels
#pragma acc loop gang(static:t)
  for(int i = 0; i < 5; ++i);

  // expected-error@+3{{'num' argument to 'gang' clause not allowed on a 'loop' construct associated with a 'kernels' construct that has a 'num_gangs' clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels num_gangs(1)
#pragma acc loop gang(1)
  for(int i = 0; i < 5; ++i);

  // expected-error@+3{{'num' argument to 'gang' clause not allowed on a 'loop' construct associated with a 'kernels' construct that has a 'num_gangs' clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels num_gangs(1)
#pragma acc loop gang(num:1)
  for(int i = 0; i < 5; ++i);

#pragma acc kernels
#pragma acc loop gang(num:1)
  for(int j = 0; j < 5; ++j) {
    // expected-error@+2{{loop with a 'gang' clause may not exist in the region of a 'gang' clause on a 'kernels' construct}}
    // expected-note@-3{{previous clause is here}}
#pragma acc loop gang(static:1)
    for(int i = 0; i < 5; ++i);
  }

#pragma acc kernels
#pragma acc loop gang(num:1)
  for(int j = 0; j < 5; ++j) {
#pragma acc loop worker(1) vector(1)
    for(int i = 0; i < 5; ++i);
  }

#pragma acc kernels
#pragma acc loop gang(num:1)
  for(int j = 0; j < 5; ++j) {
    // allowed, intervening compute construct
#pragma acc serial
#pragma acc loop gang(static:1)
    for(int i = 0; i < 5; ++i);
  }

#pragma acc kernels
#pragma acc loop gang(num:1)
  for(int i = 0; i < 5; ++i);

  // OK, on a different 'loop', not in the assoc statement.
#pragma acc loop gang(static:1)
  for(int i = 0; i < 5; ++i);

  // expected-error@+3{{OpenACC 'gang' clause may have at most one unnamed or 'num' argument}}
  // expected-note@+2{{previous expression is here}}
#pragma acc kernels
#pragma acc loop gang(5, num:1)
  for(int i = 0; i < 5; ++i);

  // expected-error@+3{{OpenACC 'gang' clause may have at most one unnamed or 'num' argument}}
  // expected-note@+2{{previous expression is here}}
#pragma acc kernels
#pragma acc loop gang(num:5, 1)
  for(int i = 0; i < 5; ++i);

  // expected-error@+3{{OpenACC 'gang' clause may have at most one unnamed or 'num' argument}}
  // expected-note@+2{{previous expression is here}}
#pragma acc kernels
#pragma acc loop gang(num:5, num:1)
  for(int i = 0; i < 5; ++i);
}

void MaxOneEntry() {
  // expected-error@+3{{OpenACC 'gang' clause may have at most one 'static' argument}}
  // expected-note@+2{{previous expression is here}}
#pragma acc kernels
#pragma acc loop gang(static: 1, static:1)
    for(int i = 0; i < 5; ++i);

#pragma acc kernels
#pragma acc loop gang gang(static:1)
    for(int i = 0; i < 5; ++i);
}


