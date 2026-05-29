// C++20 module interface with `#pragma omp split` — emit BMI + import; AST retains directive.
//
// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: %clang_cc1 -std=c++20 -fopenmp -fopenmp-version=60 -triple x86_64-unknown-linux-gnu %t/SplitMod.cppm -emit-module-interface -o %t/SplitMod.pcm
// RUN: %clang_cc1 -std=c++20 -fopenmp -fopenmp-version=60 -triple x86_64-unknown-linux-gnu %t/UseSplitMod.cpp -fmodule-file=SplitMod=%t/SplitMod.pcm -ast-dump-all | FileCheck %t/SplitMod.cppm

// expected-no-diagnostics

//--- SplitMod.cppm
module;
export module SplitMod;

export void splitfoo(int n) {
// CHECK: OMPSplitDirective
// CHECK: OMPCountsClause
#pragma omp split counts(2, omp_fill)
  for (int i = 0; i < n; ++i) {
  }
}

//--- UseSplitMod.cpp
import SplitMod;

void g(void) { splitfoo(10); }
