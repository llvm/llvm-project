// RUN: %clang_cc1 %s -fopenacc -verify

struct S{};
struct Converts{
  operator int();
};

template<typename T, unsigned Zero, unsigned Two, unsigned Four>
void ParallelTempl() {
  T i;

  // expected-error@+1{{'num' argument on 'gang' clause is not permitted on a 'parallel loop' construct}}
#pragma acc parallel loop gang(i)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{'num' argument on 'gang' clause is not permitted on a 'parallel loop' construct}}
#pragma acc parallel loop gang(num:i)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{argument to 'gang' clause dimension must be a constant expression}}
#pragma acc parallel loop gang(dim:i)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{argument to 'gang' clause dimension must be 1, 2, or 3: evaluated to 0}}
#pragma acc parallel loop gang(dim:Zero)
  for(int i = 0; i < 5; ++i);

#pragma acc parallel loop gang(dim:Two)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{argument to 'gang' clause dimension must be 1, 2, or 3: evaluated to 4}}
#pragma acc parallel loop gang(dim:Four)
  for(int i = 0; i < 5; ++i);

#pragma acc parallel loop gang(static:i) gang(dim:Two)
  for(int i = 0; i < 5; ++i);

#pragma acc parallel loop gang(static:i, dim:Two)
  for(int i = 0; i < 5; ++i);

#pragma acc parallel loop gang(dim:Two) gang(static:*)
  for(int i = 0; i < 5; ++i);

#pragma acc parallel loop gang(dim:Two, static:*)
  for(int i = 0; i < 5; ++i);

  // expected-error@+4{{OpenACC 'gang' clause may have at most one 'static' argument}}
  // expected-note@+3{{previous expression is here}}
  // expected-error@+2{{OpenACC 'gang' clause may have at most one 'dim' argument}}
  // expected-note@+1{{previous expression is here}}
#pragma acc parallel loop gang(dim:Two, static:*, dim:1, static:i)
  for(int i = 0; i < 5; ++i);
}

void Parallel() {
  ParallelTempl<int, 0, 2, 4>(); // expected-note{{in instantiation of function template}}
  int i;

  // expected-error@+1{{'num' argument on 'gang' clause is not permitted on a 'parallel loop' construct}}
#pragma acc parallel loop gang(i)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{'num' argument on 'gang' clause is not permitted on a 'parallel loop' construct}}
#pragma acc parallel loop gang(num:i)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{argument to 'gang' clause dimension must be a constant expression}}
#pragma acc parallel loop gang(dim:i)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{argument to 'gang' clause dimension must be 1, 2, or 3: evaluated to 0}}
#pragma acc parallel loop gang(dim:0)
  for(int i = 0; i < 5; ++i);

#pragma acc parallel loop gang(dim:2)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{argument to 'gang' clause dimension must be 1, 2, or 3: evaluated to 4}}
#pragma acc parallel loop gang(dim:4)
  for(int i = 0; i < 5; ++i);

#pragma acc parallel loop gang(static:i) gang(dim:2)
  for(int i = 0; i < 5; ++i);

#pragma acc parallel loop gang(static:i, dim:2)
  for(int i = 0; i < 5; ++i);

#pragma acc parallel loop gang(dim:2) gang(static:*)
  for(int i = 0; i < 5; ++i);

#pragma acc parallel loop gang(dim:2, static:*)
  for(int i = 0; i < 5; ++i);

  // expected-error@+4{{OpenACC 'gang' clause may have at most one 'static' argument}}
  // expected-note@+3{{previous expression is here}}
  // expected-error@+2{{OpenACC 'gang' clause may have at most one 'dim' argument}}
  // expected-note@+1{{previous expression is here}}
#pragma acc parallel loop gang(dim:2, static:*, dim:1, static:i)
  for(int i = 0; i < 5; ++i);
}

template<typename SomeS, typename SomeC, typename Int>
void StaticIsIntegralTempl() {
  SomeS s;
  // expected-error@+1{{OpenACC clause 'gang' requires expression of integer type ('S' invalid)}}
#pragma acc parallel loop gang(dim:2) gang(static:s)
  for(int i = 0; i < 5; ++i);

  SomeC C;
#pragma acc parallel loop gang(dim:2) gang(static:C)
  for(int i = 0; i < 5; ++i);
  Int i;
#pragma acc parallel loop gang(dim:2) gang(static:i)
  for(int i = 0; i < 5; ++i);

#pragma acc parallel loop gang(dim:2) gang(static:*)
  for(int i = 0; i < 5; ++i);
}

void StaticIsIntegral() {
  StaticIsIntegralTempl<S, Converts, int>();// expected-note{{in instantiation of function template}}

  S s;
  // expected-error@+1{{OpenACC clause 'gang' requires expression of integer type ('S' invalid)}}
#pragma acc parallel loop gang(dim:2) gang(static:s)
  for(int i = 0; i < 5; ++i);

  Converts C;
#pragma acc parallel loop gang(dim:2) gang(static:C)
  for(int i = 0; i < 5; ++i);
}

template<unsigned I>
void SerialTempl() {
  // expected-error@+1{{'num' argument on 'gang' clause is not permitted on a 'serial loop'}}
#pragma acc serial loop gang(I)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{'num' argument on 'gang' clause is not permitted on a 'serial loop'}}
#pragma acc serial loop gang(num:I)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{'dim' argument on 'gang' clause is not permitted on a 'serial loop'}}
#pragma acc serial loop gang(dim:I)
  for(int i = 0; i < 5; ++i);

#pragma acc serial loop gang(static:I)
  for(int i = 0; i < 5; ++i);
}

void Serial() {
  SerialTempl<2>();

  // expected-error@+1{{'num' argument on 'gang' clause is not permitted on a 'serial loop'}}
#pragma acc serial loop gang(1)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{'num' argument on 'gang' clause is not permitted on a 'serial loop'}}
#pragma acc serial loop gang(num:1)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{'dim' argument on 'gang' clause is not permitted on a 'serial loop'}}
#pragma acc serial loop gang(dim:1)
  for(int i = 0; i < 5; ++i);

#pragma acc serial loop gang(static:1)
  for(int i = 0; i < 5; ++i);

  int i;
#pragma acc serial loop gang(static:i)
  for(int i = 0; i < 5; ++i);
}

template<typename T>
void KernelsTempl() {
  T t;
  // expected-error@+1{{'dim' argument on 'gang' clause is not permitted on a 'kernels loop'}}
#pragma acc kernels loop gang(dim:t)
  for(int i = 0; i < 5; ++i);

#pragma acc kernels loop gang(static:t)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{'num_gangs' clause not allowed on a 'kernels loop' construct that has a 'gang' clause with a 'num' argument}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop gang(t) num_gangs(t)
  for(int i = 0; i < 5; ++i);

  // OK, kernels loop should block this.
#pragma acc kernels num_gangs(t)
#pragma acc kernels loop gang(static:t)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{'num' argument to 'gang' clause not allowed on a 'kernels loop' construct that has a 'num_gangs' clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop num_gangs(t) gang(t)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{'num_gangs' clause not allowed on a 'kernels loop' construct that has a 'gang' clause with a 'num' argument}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop gang(num:t) num_gangs(t)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{'num' argument to 'gang' clause not allowed on a 'kernels loop' construct that has a 'num_gangs' clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop num_gangs(t) gang(num:t)
  for(int i = 0; i < 5; ++i);
}

void Kernels() {
  KernelsTempl<unsigned>();

  // expected-error@+1{{'dim' argument on 'gang' clause is not permitted on a 'kernels loop'}}
#pragma acc kernels loop gang(dim:1)
  for(int i = 0; i < 5; ++i);

  unsigned t;
#pragma acc kernels loop gang(static:t)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{'num_gangs' clause not allowed on a 'kernels loop' construct that has a 'gang' clause with a 'num' argument}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop gang(t) num_gangs(t)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{'num' argument to 'gang' clause not allowed on a 'kernels loop' construct that has a 'num_gangs' clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop num_gangs(t) gang(t)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{'num_gangs' clause not allowed on a 'kernels loop' construct that has a 'gang' clause with a 'num' argument}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop gang(num:t) num_gangs(t)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{'num' argument to 'gang' clause not allowed on a 'kernels loop' construct that has a 'num_gangs' clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop num_gangs(t) gang(num:t)
  for(int i = 0; i < 5; ++i);

  // OK, intervening compute/combined construct.
#pragma acc kernels loop gang(num:1)
  for(int i = 0; i < 5; ++i) {
#pragma acc serial loop gang(static:1)
    for(int i = 0; i < 5; ++i);
  }
#pragma acc kernels loop gang(num:1)
  for(int i = 0; i < 5; ++i) {
#pragma acc serial
#pragma acc loop gang(static:1)
    for(int i = 0; i < 5; ++i);
  }
#pragma acc kernels
#pragma acc loop gang(num:1)
  for(int i = 0; i < 5; ++i) {
#pragma acc serial loop gang(static:1)
    for(int i = 0; i < 5; ++i);
  }

#pragma acc kernels loop gang(num:1)
  for(int i = 0; i < 5; ++i) {
    // expected-error@+2{{loop with a 'gang' clause may not exist in the region of a 'gang' clause on a 'kernels loop' construct}}
    // expected-note@-3{{previous clause is here}}
#pragma acc loop gang(static:1)
    for(int i = 0; i < 5; ++i);
  }

  // OK, on a different 'loop', not in the associated stmt.
#pragma acc kernels loop gang(num:1)
  for(int i = 0; i < 5; ++i);
#pragma acc loop gang(static:1)
  for(int i = 0; i < 5; ++i);

  // OK, on a different 'loop', not in the associated stmt.
#pragma acc kernels loop gang(num:1)
  for(int i = 0; i < 5; ++i);
#pragma acc kernels loop gang(static:1)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC 'gang' clause may have at most one unnamed or 'num' argument}}
  // expected-note@+1{{previous expression is here}}
#pragma acc kernels loop gang(5, num:1)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC 'gang' clause may have at most one unnamed or 'num' argument}}
  // expected-note@+1{{previous expression is here}}
#pragma acc kernels loop gang(num:5, 1)
  for(int i = 0; i < 5; ++i);

  // expected-error@+3{{OpenACC 'gang' clause may have at most one unnamed or 'num' argument}}
  // expected-note@+2{{previous expression is here}}
#pragma acc kernels
#pragma acc kernels loop gang(num:5, num:1)
  for(int i = 0; i < 5; ++i);
}

void MaxOneEntry() {
  // expected-error@+2{{OpenACC 'gang' clause may have at most one 'static' argument}}
  // expected-note@+1{{previous expression is here}}
#pragma acc kernels loop gang(static: 1, static:1)
    for(int i = 0; i < 5; ++i);

#pragma acc kernels loop gang gang(static:1)
    for(int i = 0; i < 5; ++i);
}


