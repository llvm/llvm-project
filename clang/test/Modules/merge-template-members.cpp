// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/merge-template-members -verify -emit-llvm-only %s -DTEST=1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/merge-template-members -verify -emit-llvm-only %s -DTEST=2
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/merge-template-members -verify -emit-llvm-only %s -DTEST=3
// expected-no-diagnostics

#if TEST == 1

template<typename> struct A { int n; };
template<typename> struct B { typedef A<void> C; };
template class B<int>;

#include "update.h"
B<int>::C use2;

#elif TEST == 2

#include "c.h"
N::A<int> ai;

#elif TEST == 3

#include "merge.h"

#else
#error Unknown test
#endif
