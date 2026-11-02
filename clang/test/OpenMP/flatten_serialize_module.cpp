// C++20 module interface with `#pragma omp flatten` — emit BMI + import; AST retains directive.
//
// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: %clang_cc1 -std=c++20 -fopenmp -fopenmp-version=61 -triple x86_64-unknown-linux-gnu %t/FlattenMod.cppm -emit-module-interface -o %t/FlattenMod.pcm
// RUN: %clang_cc1 -std=c++20 -fopenmp -fopenmp-version=61 -triple x86_64-unknown-linux-gnu %t/UseFlattenMod.cpp -fmodule-file=FlattenMod=%t/FlattenMod.pcm -ast-dump-all | FileCheck %t/FlattenMod.cppm

// expected-no-diagnostics

//--- FlattenMod.cppm
module;
export module FlattenMod;

export void flattenfoo(int n, int m) {
// CHECK: OMPFlattenDirective
// CHECK: ForStmt
#pragma omp flatten
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j) {
    }
}

//--- UseFlattenMod.cpp
import FlattenMod;

void g(void) { flattenfoo(10, 20); }
