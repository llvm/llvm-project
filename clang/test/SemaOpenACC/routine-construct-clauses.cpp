// RUN: %clang_cc1 %s -fopenacc -verify

void Func();
void Func2();

#pragma acc routine(Func) worker
#pragma acc routine(Func) vector nohost
#pragma acc routine(Func) nohost seq
#pragma acc routine(Func) gang
// expected-error@+2{{OpenACC clause 'bind' may not appear on the same construct as a 'bind' clause on a 'routine' construct}}
// expected-note@+1{{previous 'bind' clause is here}}
#pragma acc routine(Func) gang bind(a) bind(a)

// expected-error@+2{{OpenACC clause 'bind' may not appear on the same construct as a 'bind' clause on a 'routine' construct}}
// expected-note@+1{{previous 'bind' clause is here}}
#pragma acc routine gang bind(a) bind(a)
void DupeImplName();

// Only 1 of worker, vector, seq, gang.
// expected-error@+2{{OpenACC clause 'vector' may not appear on the same construct as a 'worker' clause on a 'routine' construct}}
// expected-note@+1{{previous 'worker' clause is here}}
#pragma acc routine(Func) worker vector
// expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'worker' clause on a 'routine' construct}}
// expected-note@+1{{previous 'worker' clause is here}}
#pragma acc routine(Func) worker seq
// expected-error@+2{{OpenACC clause 'gang' may not appear on the same construct as a 'worker' clause on a 'routine' construct}}
// expected-note@+1{{previous 'worker' clause is here}}
#pragma acc routine(Func) worker gang
// expected-error@+2{{OpenACC clause 'worker' may not appear on the same construct as a 'worker' clause on a 'routine' construct}}
// expected-note@+1{{previous 'worker' clause is here}}
#pragma acc routine(Func) worker worker
// expected-error@+2{{OpenACC clause 'worker' may not appear on the same construct as a 'vector' clause on a 'routine' construct}}
// expected-note@+1{{previous 'vector' clause is here}}
#pragma acc routine(Func) vector worker
// expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'vector' clause on a 'routine' construct}}
// expected-note@+1{{previous 'vector' clause is here}}
#pragma acc routine(Func) vector seq
// expected-error@+2{{OpenACC clause 'gang' may not appear on the same construct as a 'vector' clause on a 'routine' construct}}
// expected-note@+1{{previous 'vector' clause is here}}
#pragma acc routine(Func) vector gang
// expected-error@+2{{OpenACC clause 'vector' may not appear on the same construct as a 'vector' clause on a 'routine' construct}}
// expected-note@+1{{previous 'vector' clause is here}}
#pragma acc routine(Func) vector vector
// expected-error@+2{{OpenACC clause 'worker' may not appear on the same construct as a 'seq' clause on a 'routine' construct}}
// expected-note@+1{{previous 'seq' clause is here}}
#pragma acc routine(Func) seq worker
// expected-error@+2{{OpenACC clause 'vector' may not appear on the same construct as a 'seq' clause on a 'routine' construct}}
// expected-note@+1{{previous 'seq' clause is here}}
#pragma acc routine(Func) seq vector
// expected-error@+2{{OpenACC clause 'gang' may not appear on the same construct as a 'seq' clause on a 'routine' construct}}
// expected-note@+1{{previous 'seq' clause is here}}
#pragma acc routine(Func) seq gang
// expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'seq' clause on a 'routine' construct}}
// expected-note@+1{{previous 'seq' clause is here}}
#pragma acc routine(Func) seq seq
// expected-error@+2{{OpenACC clause 'worker' may not appear on the same construct as a 'gang' clause on a 'routine' construct}}
// expected-note@+1{{previous 'gang' clause is here}}
#pragma acc routine(Func) gang worker
// expected-error@+2{{OpenACC clause 'vector' may not appear on the same construct as a 'gang' clause on a 'routine' construct}}
// expected-note@+1{{previous 'gang' clause is here}}
#pragma acc routine(Func) gang vector
// expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'gang' clause on a 'routine' construct}}
// expected-note@+1{{previous 'gang' clause is here}}
#pragma acc routine(Func) gang seq
// expected-error@+2{{OpenACC clause 'gang' may not appear on the same construct as a 'gang' clause on a 'routine' construct}}
// expected-note@+1{{previous 'gang' clause is here}}
#pragma acc routine(Func) gang gang
// expected-error@+1{{OpenACC 'routine' construct must have at least one 'gang', 'seq', 'vector', or 'worker' clause}}
#pragma acc routine(Func)
// expected-error@+1{{OpenACC 'routine' construct must have at least one 'gang', 'seq', 'vector', or 'worker' clause}}
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

  void MemFunc();
// expected-error@+1{{argument to 'gang' clause dimension must be 1, 2, or 3: evaluated to 4}}
#pragma acc routine(MemFunc) gang(dim:T::Four())

// expected-error@+1{{argument to 'gang' clause dimension must be 1, 2, or 3: evaluated to 4}}
#pragma acc routine gang(dim:T::Four())
  void MemFunc2();
};

void Inst() {
  DependentT<HasFuncs> T;// expected-note{{in instantiation of}}
}

#pragma acc routine(Func) gang device_type(host)
#pragma acc routine(Func) gang dtype(multicore)
#pragma acc routine(Func) device_type(*) worker
#pragma acc routine(Func) dtype(*) worker

#pragma acc routine(Func) dtype(*) gang
#pragma acc routine(Func) device_type(*) worker
#pragma acc routine(Func) device_type(*) vector
#pragma acc routine(Func) dtype(*) seq
#pragma acc routine(Func2) seq device_type(*) bind("asdf")
void Func6();
#pragma acc routine(Func6) seq device_type(*) bind(WhateverElse)
#pragma acc routine(Func) seq dtype(*) device_type(*)
// expected-error@+2{{OpenACC clause 'nohost' may not follow a 'dtype' clause in a 'routine' construct}}
// expected-note@+1{{active 'dtype' clause here}}
#pragma acc routine(Func) seq dtype(*) nohost
// expected-error@+2{{OpenACC clause 'nohost' may not follow a 'device_type' clause in a 'routine' construct}}
// expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine(Func) seq device_type(*) nohost

// 2.15: a bind clause may not bind to a routine name that has a visible bind clause.
void Func3();
#pragma acc routine(Func3) seq bind("asdf")
// OK: Doesn't have a bind
#pragma acc routine(Func3) seq
// expected-error@+2{{multiple 'routine' directives with 'bind' clauses are not permitted to refer to the same function}}
// expected-note@-4{{previous 'bind' clause is here}}
#pragma acc routine(Func3) seq bind("asdf")

void Func4();
// OK: Doesn't have a bind
#pragma acc routine(Func4) seq
#pragma acc routine(Func4) seq bind("asdf")
// expected-error@+2{{multiple 'routine' directives with 'bind' clauses are not permitted to refer to the same function}}
// expected-note@-2{{previous 'bind' clause is here}}
#pragma acc routine(Func4) seq bind("asdf")
void Func5();
#pragma acc routine(Func5) seq bind("asdf")
// expected-error@+2{{multiple 'routine' directives with 'bind' clauses are not permitted to refer to the same function}}
// expected-note@-2{{previous 'bind' clause is here}}
#pragma acc routine(Func5) seq bind("asdf")
// OK: Doesn't have a bind
#pragma acc routine(Func5) seq

// OK, same.
#pragma acc routine bind("asdf") seq
void DupeBinds1();
#pragma acc routine bind("asdf") seq
void DupeBinds1();

#pragma acc routine bind(asdf) seq
void DupeBinds2();
#pragma acc routine bind(asdf) seq
void DupeBinds2();

#pragma acc routine bind("asdf") seq
void DupeBinds3();
// expected-error@+2{{OpenACC 'bind' clause on a declaration must bind to the same name as previous 'bind' clauses}}
// expected-note@-3{{previous 'bind' clause is here}}
#pragma acc routine bind(asdf) seq
void DupeBinds3();

void DupeBinds4();
#pragma acc routine(DupeBinds4) bind(asdf) seq
// expected-error@+2{{multiple 'routine' directives with 'bind' clauses are not permitted to refer to the same function}}
// expected-note@-2{{previous 'bind' clause is here}}
#pragma acc routine bind(asdf) seq
void DupeBinds4();

void DupeBinds4b();
// expected-error@+1{{expected function or lambda declaration for 'routine' construct}}
#pragma acc routine bind(asdf) seq
#pragma acc routine(DupeBinds4b) bind(asdf) seq
void DupeBinds4b();

#pragma acc routine bind(asdf) seq
void DupeBinds5();
// expected-error@+2{{OpenACC 'bind' clause on a declaration must bind to the same name as previous 'bind' clauses}}
// expected-note@-3{{previous 'bind' clause is here}}
#pragma acc routine bind(asdfDiff) seq
void DupeBinds5();

#pragma acc routine bind("asdf") seq
void DupeBinds6();
// expected-error@+2{{OpenACC 'bind' clause on a declaration must bind to the same name as previous 'bind' clauses}}
// expected-note@-3{{previous 'bind' clause is here}}
#pragma acc routine bind("asdfDiff") seq
void DupeBinds6();

// The standard wasn't initially clear, but is being clarified that we require
// exactly one of (gang, worker, vector, or seq) to apply to each 'device_type'.
// The ones before any 'device_type' apply to all.
namespace FigureDupesAllowedAroundDeviceType {
  // Test conflicts without a device type
  // expected-error@+2{{OpenACC clause 'gang' may not appear on the same construct as a 'gang' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'gang' clause is here}}
#pragma acc routine gang gang
  void Func1();
  // expected-error@+2{{OpenACC clause 'worker' may not appear on the same construct as a 'gang' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'gang' clause is here}}
#pragma acc routine gang worker
  void Func2();
  // expected-error@+2{{OpenACC clause 'vector' may not appear on the same construct as a 'gang' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'gang' clause is here}}
#pragma acc routine gang vector
  void Func3();
  // expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'gang' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'gang' clause is here}}
#pragma acc routine gang seq
  void Func4();
  // expected-error@+2{{OpenACC clause 'gang' may not appear on the same construct as a 'worker' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'worker' clause is here}}
#pragma acc routine worker gang
  void Func5();
  // expected-error@+2{{OpenACC clause 'worker' may not appear on the same construct as a 'worker' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'worker' clause is here}}
#pragma acc routine worker worker
  void Func6();
  // expected-error@+2{{OpenACC clause 'vector' may not appear on the same construct as a 'worker' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'worker' clause is here}}
#pragma acc routine worker vector
  void Func7();
  // expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'worker' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'worker' clause is here}}
#pragma acc routine worker seq
  void Func8();
  // expected-error@+2{{OpenACC clause 'gang' may not appear on the same construct as a 'vector' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'vector' clause is here}}
#pragma acc routine vector gang
  void Func9();
  // expected-error@+2{{OpenACC clause 'worker' may not appear on the same construct as a 'vector' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'vector' clause is here}}
#pragma acc routine vector worker
  void Func10();
  // expected-error@+2{{OpenACC clause 'vector' may not appear on the same construct as a 'vector' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'vector' clause is here}}
#pragma acc routine vector vector
  void Func11();
  // expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'vector' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'vector' clause is here}}
#pragma acc routine vector seq
  void Func12();
  // expected-error@+2{{OpenACC clause 'gang' may not appear on the same construct as a 'seq' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'seq' clause is here}}
#pragma acc routine seq gang
  void Func13();
  // expected-error@+2{{OpenACC clause 'worker' may not appear on the same construct as a 'seq' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'seq' clause is here}}
#pragma acc routine seq worker
  void Func14();
  // expected-error@+2{{OpenACC clause 'vector' may not appear on the same construct as a 'seq' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'seq' clause is here}}
#pragma acc routine seq vector
  void Func15();
  // expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'seq' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'seq' clause is here}}
#pragma acc routine seq seq
  void Func16();
  // None included for one without a device type
  // expected-error@+1{{OpenACC 'routine' construct must have at least one 'gang', 'seq', 'vector', or 'worker' clause}}
#pragma acc routine
  void Func16();

  // Check same conflicts for 'before' the device_type
  // expected-error@+2{{OpenACC clause 'gang' may not appear on the same construct as a 'gang' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'gang' clause is here}}
#pragma acc routine gang gang device_type(*)
  void Func17();
  // expected-error@+2{{OpenACC clause 'worker' may not appear on the same construct as a 'gang' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'gang' clause is here}}
#pragma acc routine gang worker device_type(*)
  void Func18();
  // expected-error@+2{{OpenACC clause 'vector' may not appear on the same construct as a 'gang' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'gang' clause is here}}
#pragma acc routine gang vector device_type(*)
  void Func19();
  // expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'gang' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'gang' clause is here}}
#pragma acc routine gang seq device_type(*)
  void Func20();
  // expected-error@+2{{OpenACC clause 'gang' may not appear on the same construct as a 'worker' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'worker' clause is here}}
#pragma acc routine worker gang device_type(*)
  void Func21();
  // expected-error@+2{{OpenACC clause 'worker' may not appear on the same construct as a 'worker' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'worker' clause is here}}
#pragma acc routine worker worker device_type(*)
  void Func22();
  // expected-error@+2{{OpenACC clause 'vector' may not appear on the same construct as a 'worker' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'worker' clause is here}}
#pragma acc routine worker vector device_type(*)
  void Func23();
  // expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'worker' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'worker' clause is here}}
#pragma acc routine worker seq device_type(*)
  void Func24();
  // expected-error@+2{{OpenACC clause 'gang' may not appear on the same construct as a 'vector' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'vector' clause is here}}
#pragma acc routine vector gang device_type(*)
  void Func25();
  // expected-error@+2{{OpenACC clause 'worker' may not appear on the same construct as a 'vector' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'vector' clause is here}}
#pragma acc routine vector worker device_type(*)
  void Func26();
  // expected-error@+2{{OpenACC clause 'vector' may not appear on the same construct as a 'vector' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'vector' clause is here}}
#pragma acc routine vector vector device_type(*)
  void Func27();
  // expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'vector' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'vector' clause is here}}
#pragma acc routine vector seq device_type(*)
  void Func28();
  // expected-error@+2{{OpenACC clause 'gang' may not appear on the same construct as a 'seq' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'seq' clause is here}}
#pragma acc routine seq gang device_type(*)
  void Func29();
  // expected-error@+2{{OpenACC clause 'worker' may not appear on the same construct as a 'seq' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'seq' clause is here}}
#pragma acc routine seq worker device_type(*)
  void Func30();
  // expected-error@+2{{OpenACC clause 'vector' may not appear on the same construct as a 'seq' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'seq' clause is here}}
#pragma acc routine seq vector device_type(*)
  void Func31();
  // expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'seq' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'seq' clause is here}}
#pragma acc routine seq seq device_type(*)
  void Func32();
  // None included for the device_type
  // expected-error@+1{{OpenACC 'routine' construct must have at least one 'gang', 'seq', 'vector', or 'worker' clause that applies to each 'device_type'}}
#pragma acc routine device_type(*)
  void Func33();

  // Conflicts between 'global' and 'device_type'
  // expected-error@+3{{OpenACC clause 'gang' after 'device_type' clause on a 'routine' conflicts with the 'gang' clause before the first 'device_type'}}
  // expected-note@+2{{previous 'gang' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine gang device_type(*) gang
  void Func34();
  // expected-error@+3{{OpenACC clause 'worker' after 'device_type' clause on a 'routine' conflicts with the 'gang' clause before the first 'device_type'}}
  // expected-note@+2{{previous 'gang' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine gang device_type(*) worker
  void Func35();
  // expected-error@+3{{OpenACC clause 'vector' after 'device_type' clause on a 'routine' conflicts with the 'gang' clause before the first 'device_type'}}
  // expected-note@+2{{previous 'gang' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine gang device_type(*) vector
  void Func36();
  // expected-error@+3{{OpenACC clause 'seq' after 'device_type' clause on a 'routine' conflicts with the 'gang' clause before the first 'device_type'}}
  // expected-note@+2{{previous 'gang' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine gang device_type(*) seq
  void Func37();
  // expected-error@+3{{OpenACC clause 'gang' after 'device_type' clause on a 'routine' conflicts with the 'worker' clause before the first 'device_type'}}
  // expected-note@+2{{previous 'worker' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine worker device_type(*) gang
  void Func38();
  // expected-error@+3{{OpenACC clause 'worker' after 'device_type' clause on a 'routine' conflicts with the 'worker' clause before the first 'device_type'}}
  // expected-note@+2{{previous 'worker' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine worker device_type(*) worker
  void Func39();
  // expected-error@+3{{OpenACC clause 'vector' after 'device_type' clause on a 'routine' conflicts with the 'worker' clause before the first 'device_type'}}
  // expected-note@+2{{previous 'worker' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine worker device_type(*) vector
  void Func40();
  // expected-error@+3{{OpenACC clause 'seq' after 'device_type' clause on a 'routine' conflicts with the 'worker' clause before the first 'device_type'}}
  // expected-note@+2{{previous 'worker' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine worker device_type(*) seq
  void Func41();
  // expected-error@+3{{OpenACC clause 'gang' after 'device_type' clause on a 'routine' conflicts with the 'vector' clause before the first 'device_type'}}
  // expected-note@+2{{previous 'vector' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine vector device_type(*) gang
  void Func42();
  // expected-error@+3{{OpenACC clause 'worker' after 'device_type' clause on a 'routine' conflicts with the 'vector' clause before the first 'device_type'}}
  // expected-note@+2{{previous 'vector' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine vector device_type(*) worker
  void Func43();
  // expected-error@+3{{OpenACC clause 'vector' after 'device_type' clause on a 'routine' conflicts with the 'vector' clause before the first 'device_type'}}
  // expected-note@+2{{previous 'vector' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine vector device_type(*) vector
  void Func44();
  // expected-error@+3{{OpenACC clause 'seq' after 'device_type' clause on a 'routine' conflicts with the 'vector' clause before the first 'device_type'}}
  // expected-note@+2{{previous 'vector' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine vector device_type(*) seq
  void Func45();
  // expected-error@+3{{OpenACC clause 'gang' after 'device_type' clause on a 'routine' conflicts with the 'seq' clause before the first 'device_type'}}
  // expected-note@+2{{previous 'seq' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine seq device_type(*) gang
  void Func46();
  // expected-error@+3{{OpenACC clause 'worker' after 'device_type' clause on a 'routine' conflicts with the 'seq' clause before the first 'device_type'}}
  // expected-note@+2{{previous 'seq' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine seq device_type(*) worker
  void Func47();
  // expected-error@+3{{OpenACC clause 'vector' after 'device_type' clause on a 'routine' conflicts with the 'seq' clause before the first 'device_type'}}
  // expected-note@+2{{previous 'seq' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine seq device_type(*) vector
  void Func48();
  // expected-error@+3{{OpenACC clause 'seq' after 'device_type' clause on a 'routine' conflicts with the 'seq' clause before the first 'device_type'}}
  // expected-note@+2{{previous 'seq' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine seq device_type(*) seq
  void Func49();

  // Conflicts within same device_type
  // expected-error@+3{{OpenACC clause 'gang' on a 'routine' directive conflicts with the 'gang' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'gang' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) gang gang
  void Func50();
  // expected-error@+3{{OpenACC clause 'worker' on a 'routine' directive conflicts with the 'gang' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'gang' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) gang worker
  void Func51();
  // expected-error@+3{{OpenACC clause 'vector' on a 'routine' directive conflicts with the 'gang' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'gang' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) gang vector
  void Func52();
  // expected-error@+3{{OpenACC clause 'seq' on a 'routine' directive conflicts with the 'gang' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'gang' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) gang seq
  void Func53();
  // expected-error@+3{{OpenACC clause 'gang' on a 'routine' directive conflicts with the 'worker' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'worker' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) worker gang
  void Func54();
  // expected-error@+3{{OpenACC clause 'worker' on a 'routine' directive conflicts with the 'worker' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'worker' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) worker worker
  void Func55();
  // expected-error@+3{{OpenACC clause 'vector' on a 'routine' directive conflicts with the 'worker' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'worker' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) worker vector
  void Func56();
  // expected-error@+3{{OpenACC clause 'seq' on a 'routine' directive conflicts with the 'worker' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'worker' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) worker seq
  void Func57();
  // expected-error@+3{{OpenACC clause 'gang' on a 'routine' directive conflicts with the 'vector' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'vector' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) vector gang
  void Func58();
  // expected-error@+3{{OpenACC clause 'worker' on a 'routine' directive conflicts with the 'vector' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'vector' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) vector worker
  void Func59();
  // expected-error@+3{{OpenACC clause 'vector' on a 'routine' directive conflicts with the 'vector' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'vector' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) vector vector
  void Func60();
  // expected-error@+3{{OpenACC clause 'seq' on a 'routine' directive conflicts with the 'vector' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'vector' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) vector seq
  void Func61();
  // expected-error@+3{{OpenACC clause 'gang' on a 'routine' directive conflicts with the 'seq' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'seq' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) seq gang
  void Func62();
  // expected-error@+3{{OpenACC clause 'worker' on a 'routine' directive conflicts with the 'seq' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'seq' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) seq worker
  void Func63();
  // expected-error@+3{{OpenACC clause 'vector' on a 'routine' directive conflicts with the 'seq' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'seq' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) seq vector
  void Func64();
  // expected-error@+3{{OpenACC clause 'seq' on a 'routine' directive conflicts with the 'seq' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'seq' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) seq seq
  void Func65();

  // expected-error@+3{{OpenACC clause 'gang' on a 'routine' directive conflicts with the 'gang' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'gang' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) gang gang device_type(nvidia) seq
  void Func66();
  // expected-error@+3{{OpenACC clause 'worker' on a 'routine' directive conflicts with the 'gang' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'gang' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) gang worker device_type(nvidia) seq
  void Func67();
  // expected-error@+3{{OpenACC clause 'vector' on a 'routine' directive conflicts with the 'gang' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'gang' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) gang vector device_type(nvidia) seq
  void Func68();
  // expected-error@+3{{OpenACC clause 'seq' on a 'routine' directive conflicts with the 'gang' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'gang' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) gang seq device_type(nvidia) seq
  void Func69();
  // expected-error@+3{{OpenACC clause 'gang' on a 'routine' directive conflicts with the 'worker' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'worker' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) worker gang device_type(nvidia) seq
  void Func70();
  // expected-error@+3{{OpenACC clause 'worker' on a 'routine' directive conflicts with the 'worker' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'worker' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) worker worker device_type(nvidia) seq
  void Func71();
  // expected-error@+3{{OpenACC clause 'vector' on a 'routine' directive conflicts with the 'worker' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'worker' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) worker vector device_type(nvidia) seq
  void Func72();
  // expected-error@+3{{OpenACC clause 'seq' on a 'routine' directive conflicts with the 'worker' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'worker' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) worker seq device_type(nvidia) seq
  void Func73();
  // expected-error@+3{{OpenACC clause 'gang' on a 'routine' directive conflicts with the 'vector' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'vector' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) vector gang device_type(nvidia) seq
  void Func74();
  // expected-error@+3{{OpenACC clause 'worker' on a 'routine' directive conflicts with the 'vector' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'vector' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) vector worker device_type(nvidia) seq
  void Func75();
  // expected-error@+3{{OpenACC clause 'vector' on a 'routine' directive conflicts with the 'vector' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'vector' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) vector vector device_type(nvidia) seq
  void Func76();
  // expected-error@+3{{OpenACC clause 'seq' on a 'routine' directive conflicts with the 'vector' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'vector' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) vector seq device_type(nvidia) seq
  void Func77();
  // expected-error@+3{{OpenACC clause 'gang' on a 'routine' directive conflicts with the 'seq' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'seq' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) seq gang device_type(nvidia) seq
  void Func78();
  // expected-error@+3{{OpenACC clause 'worker' on a 'routine' directive conflicts with the 'seq' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'seq' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) seq worker device_type(nvidia) seq
  void Func79();
  // expected-error@+3{{OpenACC clause 'vector' on a 'routine' directive conflicts with the 'seq' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'seq' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) seq vector device_type(nvidia) seq
  void Func80();
  // expected-error@+3{{OpenACC clause 'seq' on a 'routine' directive conflicts with the 'seq' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'seq' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(*) seq seq device_type(nvidia) seq
  void Func81();

  // expected-error@+3{{OpenACC clause 'gang' on a 'routine' directive conflicts with the 'gang' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'gang' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(nvidia) seq device_type(*) gang gang
  void Func82();
  // expected-error@+3{{OpenACC clause 'worker' on a 'routine' directive conflicts with the 'gang' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'gang' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(nvidia) seq device_type(*) gang worker
  void Func83();
  // expected-error@+3{{OpenACC clause 'vector' on a 'routine' directive conflicts with the 'gang' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'gang' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(nvidia) seq device_type(*) gang vector
  void Func84();
  // expected-error@+3{{OpenACC clause 'seq' on a 'routine' directive conflicts with the 'gang' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'gang' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(nvidia) seq device_type(*) gang seq
  void Func85();
  // expected-error@+3{{OpenACC clause 'gang' on a 'routine' directive conflicts with the 'worker' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'worker' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(nvidia) seq device_type(*) worker gang
  void Func86();
  // expected-error@+3{{OpenACC clause 'worker' on a 'routine' directive conflicts with the 'worker' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'worker' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(nvidia) seq device_type(*) worker worker
  void Func87();
  // expected-error@+3{{OpenACC clause 'vector' on a 'routine' directive conflicts with the 'worker' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'worker' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(nvidia) seq device_type(*) worker vector
  void Func88();
  // expected-error@+3{{OpenACC clause 'seq' on a 'routine' directive conflicts with the 'worker' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'worker' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(nvidia) seq device_type(*) worker seq
  void Func89();
  // expected-error@+3{{OpenACC clause 'gang' on a 'routine' directive conflicts with the 'vector' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'vector' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(nvidia) seq device_type(*) vector gang
  void Func90();
  // expected-error@+3{{OpenACC clause 'worker' on a 'routine' directive conflicts with the 'vector' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'vector' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(nvidia) seq device_type(*) vector worker
  void Func91();
  // expected-error@+3{{OpenACC clause 'vector' on a 'routine' directive conflicts with the 'vector' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'vector' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(nvidia) seq device_type(*) vector vector
  void Func92();
  // expected-error@+3{{OpenACC clause 'seq' on a 'routine' directive conflicts with the 'vector' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'vector' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(nvidia) seq device_type(*) vector seq
  void Func93();
  // expected-error@+3{{OpenACC clause 'gang' on a 'routine' directive conflicts with the 'seq' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'seq' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(nvidia) seq device_type(*) seq gang
  void Func94();
  // expected-error@+3{{OpenACC clause 'worker' on a 'routine' directive conflicts with the 'seq' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'seq' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(nvidia) seq device_type(*) seq worker
  void Func95();
  // expected-error@+3{{OpenACC clause 'vector' on a 'routine' directive conflicts with the 'seq' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'seq' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(nvidia) seq device_type(*) seq vector
  void Func96();
  // expected-error@+3{{OpenACC clause 'seq' on a 'routine' directive conflicts with the 'seq' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'seq' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine device_type(nvidia) seq device_type(*) seq seq
  void Func97();

  // All fine?
#pragma acc routine device_type(*) gang device_type(nvidia) gang
  void Func98();
#pragma acc routine device_type(*) gang device_type(nvidia) worker
  void Func99();
#pragma acc routine device_type(*) gang device_type(nvidia) vector
  void Func100();
#pragma acc routine device_type(*) gang device_type(nvidia) seq
  void Func101();
#pragma acc routine device_type(*) worker device_type(nvidia) gang
  void Func102();
#pragma acc routine device_type(*) worker device_type(nvidia) worker
  void Func103();
#pragma acc routine device_type(*) worker device_type(nvidia) vector
  void Func104();
#pragma acc routine device_type(*) worker device_type(nvidia) seq
  void Func105();
#pragma acc routine device_type(*) vector device_type(nvidia) gang
  void Func106();
#pragma acc routine device_type(*) vector device_type(nvidia) worker
  void Func107();
#pragma acc routine device_type(*) vector device_type(nvidia) vector
  void Func108();
#pragma acc routine device_type(*) vector device_type(nvidia) seq
  void Func109();
#pragma acc routine device_type(*) seq device_type(nvidia) gang
  void Func110();
#pragma acc routine device_type(*) seq device_type(nvidia) worker
  void Func111();
#pragma acc routine device_type(*) seq device_type(nvidia) vector
  void Func112();
#pragma acc routine device_type(*) seq device_type(nvidia) seq
  void Func113();

}

namespace BindDupes {
  // expected-error@+2{{OpenACC clause 'bind' may not appear on the same construct as a 'bind' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'bind' clause is here}}
#pragma acc routine seq bind(asdf) bind(asdf)
  void Func1();
  // expected-error@+2{{OpenACC clause 'bind' may not appear on the same construct as a 'bind' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'bind' clause is here}}
#pragma acc routine seq bind(asdf) bind(asdf) device_type(*)
  void Func2();
  // expected-error@+3{{OpenACC clause 'bind' after 'device_type' clause on a 'routine' conflicts with the 'bind' clause before the first 'device_type'}}
  // expected-note@+2{{previous 'bind' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine seq bind(asdf) device_type(*) bind(asdf)
  void Func3();
  // expected-error@+3{{OpenACC clause 'bind' after 'device_type' clause on a 'routine' conflicts with the 'bind' clause before the first 'device_type'}}
  // expected-note@+2{{previous 'bind' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine seq bind(asdf) device_type(*) bind(asdf) device_type(*)
  void Func4();
  // expected-error@+3{{OpenACC clause 'bind' on a 'routine' directive conflicts with the 'bind' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'bind' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine seq device_type(*) bind(asdf) bind(asdf)
  void Func5();
  // expected-error@+3{{OpenACC clause 'bind' on a 'routine' directive conflicts with the 'bind' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'bind' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine seq device_type(*) bind(asdf) bind(asdf) device_type(*)
  void Func6();
  // expected-error@+3{{OpenACC clause 'bind' on a 'routine' directive conflicts with the 'bind' clause applying to the same 'device_type'}}
  // expected-note@+2{{previous 'bind' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc routine seq device_type(*) bind(asdf) device_type(*) bind(asdf) bind(asdf)
  void Func7();

#pragma acc routine seq device_type(*) bind(asdf) device_type(*) bind(asdf)
  void Func8();
}
