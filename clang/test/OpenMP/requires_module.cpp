// RUN: rm -rf %t && split-file %s %t
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fmodules -fmodule-name=rev \
// RUN:   -x c++ -emit-module %t/module.modulemap -o %t/rev.pcm
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fmodules -fmodule-file=%t/rev.pcm \
// RUN:   -verify -fsyntax-only %t/use.cpp
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-targets=x86_64 -triple x86_64 \
// RUN:   -fmodules -fmodule-name=rev -x c++ -emit-module %t/module.modulemap -o %t/rev2.pcm
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-targets=x86_64 -triple x86_64 \
// RUN:   -fmodules -fmodule-file=%t/rev2.pcm -verify -fsyntax-only %t/use.cpp

// A 'requires' directive read from a module must keep its effect on the
// translation unit importing it.

//--- module.modulemap
module rev { header "rev.h" export * }

//--- rev.h
#pragma omp requires reverse_offload
void foo();

//--- use.cpp
#include "rev.h"

// expected-no-diagnostics
void bar(int argc) {
#pragma omp target device(ancestor : argc)
  foo();
}
