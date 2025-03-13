// RUN: %clang_cc1 %s -fopenacc -verify

void Func();
void Func2();

#pragma acc routine(Func) worker
#pragma acc routine(Func) vector nohost
#pragma acc routine(Func) nohost seq
#pragma acc routine(Func) gang

// Only 1 of worker, vector, seq, gang.
// expected-error@+2{{OpenACC clause 'vector' may not appear on the same construct as a 'worker' clause on a 'routine' construct}}
// expected-note@+1{{previous clause is here}}
#pragma acc routine(Func) worker vector
// expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'worker' clause on a 'routine' construct}}
// expected-note@+1{{previous clause is here}}
#pragma acc routine(Func) worker seq
// expected-error@+2{{OpenACC clause 'gang' may not appear on the same construct as a 'worker' clause on a 'routine' construct}}
// expected-note@+1{{previous clause is here}}
#pragma acc routine(Func) worker gang
// expected-error@+2{{OpenACC clause 'worker' may not appear on the same construct as a 'worker' clause on a 'routine' construct}}
// expected-note@+1{{previous clause is here}}
#pragma acc routine(Func) worker worker
// expected-error@+2{{OpenACC clause 'worker' may not appear on the same construct as a 'vector' clause on a 'routine' construct}}
// expected-note@+1{{previous clause is here}}
#pragma acc routine(Func) vector worker
// expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'vector' clause on a 'routine' construct}}
// expected-note@+1{{previous clause is here}}
#pragma acc routine(Func) vector seq
// expected-error@+2{{OpenACC clause 'gang' may not appear on the same construct as a 'vector' clause on a 'routine' construct}}
// expected-note@+1{{previous clause is here}}
#pragma acc routine(Func) vector gang
// expected-error@+2{{OpenACC clause 'vector' may not appear on the same construct as a 'vector' clause on a 'routine' construct}}
// expected-note@+1{{previous clause is here}}
#pragma acc routine(Func) vector vector
// expected-error@+2{{OpenACC clause 'worker' may not appear on the same construct as a 'seq' clause on a 'routine' construct}}
// expected-note@+1{{previous clause is here}}
#pragma acc routine(Func) seq worker
// expected-error@+2{{OpenACC clause 'vector' may not appear on the same construct as a 'seq' clause on a 'routine' construct}}
// expected-note@+1{{previous clause is here}}
#pragma acc routine(Func) seq vector
// expected-error@+2{{OpenACC clause 'gang' may not appear on the same construct as a 'seq' clause on a 'routine' construct}}
// expected-note@+1{{previous clause is here}}
#pragma acc routine(Func) seq gang
// expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'seq' clause on a 'routine' construct}}
// expected-note@+1{{previous clause is here}}
#pragma acc routine(Func) seq seq
// expected-error@+2{{OpenACC clause 'worker' may not appear on the same construct as a 'gang' clause on a 'routine' construct}}
// expected-note@+1{{previous clause is here}}
#pragma acc routine(Func) gang worker
// expected-error@+2{{OpenACC clause 'vector' may not appear on the same construct as a 'gang' clause on a 'routine' construct}}
// expected-note@+1{{previous clause is here}}
#pragma acc routine(Func) gang vector
// expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'gang' clause on a 'routine' construct}}
// expected-note@+1{{previous clause is here}}
#pragma acc routine(Func) gang seq
// expected-error@+2{{OpenACC clause 'gang' may not appear on the same construct as a 'gang' clause on a 'routine' construct}}
// expected-note@+1{{previous clause is here}}
#pragma acc routine(Func) gang gang
// expected-error@+1{{OpenACC 'routine' construct must have at least one 'gang', 'worker', 'vector' or 'seq' clause}}
#pragma acc routine(Func)
// expected-error@+1{{OpenACC 'routine' construct must have at least one 'gang', 'worker', 'vector' or 'seq' clause}}
#pragma acc routine(Func) nohost

// only the 'dim' syntax for gang is legal.
#pragma acc routine(Func) gang(dim:1)
// expected-error@+1{{'num' argument on 'gang' clause is not permitted on a 'routine' construct}}
#pragma acc routine(Func) gang(1)
// expected-error@+1{{'num' argument on 'gang' clause is not permitted on a 'routine' construct}}
#pragma acc routine(Func) gang(num:1)
// expected-error@+1{{'static' argument on 'gang' clause is not permitted on a 'routine' construct}}
#pragma acc routine(Func) gang(static:1)
// expected-error@+2{{OpenACC 'gang' clause may have at most one 'dim' argument}}
// expected-note@+1{{previous expression is here}}
#pragma acc routine(Func) gang(dim:1, dim:2)

// worker, vector, seq don't allow arguments.
// expected-error@+1{{'num' argument on 'worker' clause is not permitted on a 'routine' construct}}
#pragma acc routine(Func) worker(1)
// expected-error@+1{{'length' argument on 'vector' clause is not permitted on a 'routine' construct}}
#pragma acc routine(Func) vector(1)
// expected-error@+1{{expected identifier}}
#pragma acc routine(Func) seq(1)

int getSomeInt();
// dim must be a constant positive integer.
// expected-error@+1{{argument to 'gang' clause dimension must be a constant expression}}
#pragma acc routine(Func) gang(dim:getSomeInt())

struct HasFuncs {
static constexpr int Neg() { return -5; }
static constexpr int Zero() { return 0; }
static constexpr int One() { return 1; }
static constexpr int Two() { return 2; }
static constexpr int Three() { return 3; }
static constexpr int Four() { return 4; }
};
// 'dim' must be 1, 2, or 3.
// expected-error@+1{{argument to 'gang' clause dimension must be 1, 2, or 3: evaluated to -5}}
#pragma acc routine(Func) gang(dim:HasFuncs::Neg())
// expected-error@+1{{argument to 'gang' clause dimension must be 1, 2, or 3: evaluated to 0}}
#pragma acc routine(Func) gang(dim:HasFuncs::Zero())
#pragma acc routine(Func) gang(dim:HasFuncs::One())
#pragma acc routine(Func) gang(dim:HasFuncs::Two())
#pragma acc routine(Func) gang(dim:HasFuncs::Three())
// expected-error@+1{{argument to 'gang' clause dimension must be 1, 2, or 3: evaluated to 4}}
#pragma acc routine(Func) gang(dim:HasFuncs::Four())

template<typename T>
struct DependentT {
// expected-error@+1{{argument to 'gang' clause dimension must be 1, 2, or 3: evaluated to -5}}
#pragma acc routine(Func) gang(dim:T::Neg())
// expected-error@+1{{argument to 'gang' clause dimension must be 1, 2, or 3: evaluated to 0}}
#pragma acc routine(Func) gang(dim:T::Zero()) nohost
#pragma acc routine(Func) nohost gang(dim:T::One())
#pragma acc routine(Func) gang(dim:T::Two())
#pragma acc routine(Func) gang(dim:T::Three())
// expected-error@+1{{argument to 'gang' clause dimension must be 1, 2, or 3: evaluated to 4}}
#pragma acc routine(Func) gang(dim:T::Four())
};

void Inst() {
  DependentT<HasFuncs> T;// expected-note{{in instantiation of}}
}

#pragma acc routine(Func) gang device_type(Inst)
#pragma acc routine(Func) gang dtype(Inst)
#pragma acc routine(Func) device_type(*) worker
#pragma acc routine(Func) dtype(*) worker

#pragma acc routine(Func) dtype(*) gang
#pragma acc routine(Func) device_type(*) worker
#pragma acc routine(Func) device_type(*) vector
#pragma acc routine(Func) dtype(*) seq
#pragma acc routine(Func) seq device_type(*) bind("asdf")
#pragma acc routine(Func2) seq device_type(*) bind(WhateverElse)
#pragma acc routine(Func) seq dtype(*) device_type(*)
// expected-error@+2{{OpenACC clause 'nohost' may not follow a 'dtype' clause in a 'routine' construct}}
// expected-note@+1{{previous clause is here}}
#pragma acc routine(Func) seq dtype(*) nohost
// expected-error@+2{{OpenACC clause 'nohost' may not follow a 'device_type' clause in a 'routine' construct}}
// expected-note@+1{{previous clause is here}}
#pragma acc routine(Func) seq device_type(*) nohost

// 2.15: a bind clause may not bind to a routine name that has a visible bind clause.
void Func3();
#pragma acc routine(Func3) seq bind("asdf")
// OK: Doesn't have a bind
#pragma acc routine(Func3) seq
// expected-error@+2{{multiple 'routine' directives with 'bind' clauses are not permitted to refer to the same function}}
// expected-note@-4{{previous clause is here}}
#pragma acc routine(Func3) seq bind("asdf")

void Func4();
#pragma acc routine(Func4) seq
// OK: Doesn't have a bind
#pragma acc routine(Func4) seq bind("asdf")
// expected-error@+2{{multiple 'routine' directives with 'bind' clauses are not permitted to refer to the same function}}
// expected-note@-2{{previous clause is here}}
#pragma acc routine(Func4) seq bind("asdf")
void Func5();
#pragma acc routine(Func5) seq bind("asdf")
// expected-error@+2{{multiple 'routine' directives with 'bind' clauses are not permitted to refer to the same function}}
// expected-note@-2{{previous clause is here}}
#pragma acc routine(Func5) seq bind("asdf")
// OK: Doesn't have a bind
#pragma acc routine(Func5) seq
