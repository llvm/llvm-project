// Diagnosing ODR violations can load declarations, and a deserialization that
// finishes there used to call ASTReader::diagnoseOdrViolations() again for the
// remaining pending ODR merge checks, adding stack frames for each check. With
// enough pending checks the compiler ran out of stack.
//
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -std=c++17 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -I%t -fsyntax-only -verify %t/tu.cpp

//--- module.modulemap
module Chain { textual header "chain.h" }
module A1 { header "a1.h" export * }
module A2 { header "a2.h" export * }

//--- chain.h
#ifndef CHAIN_H
#define CHAIN_H
#define N 10000
template <int I> struct S { using T = int; };
template <class T, T... I> struct Seq {};
template <class... T> struct List {};
template <int... I> List<typename S<I>::T...> list(Seq<int, I...>);
template <class T, T... I> struct Holder : S<I>... {};
#endif

//--- a1.h
#include "chain.h"
using A1Holder = __make_integer_seq<Holder, int, N>;
constexpr unsigned long A1HolderSize = sizeof(A1Holder);

//--- a2.h
#include "chain.h"
using A2List = decltype(list(__make_integer_seq<Seq, int, N>{}));

//--- tu.cpp
// expected-no-diagnostics
#include "a1.h"
// Load A1's definitions of S<0> ... S<N-1>.
unsigned long Size = sizeof(A1Holder);
#include "a2.h"
// Read A2's S<I>::T for every I. Each is a member of a definition merged into
// A1's S<I>, so each is queued as an ODR merge check.
A2List L;
