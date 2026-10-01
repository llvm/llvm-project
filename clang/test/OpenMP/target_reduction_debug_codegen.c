// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux-gnu \
// RUN:   -fopenmp-targets=amdgpu-amd-amdhsa -debug-info-kind=limited \
// RUN:   -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c -triple amdgpu-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgpu-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc -debug-info-kind=limited \
// RUN:   -emit-llvm %s -o - | FileCheck %s

// expected-no-diagnostics

// The reduction combiners are emitted into helper functions created by
// OpenMPIRBuilder that have no DISubprogram of their own. They must therefore
// not carry any debug locations: a !dbg there would name the scope of the
// enclosing outlined function, which is invalid IR. The Verifier cannot see
// this while the helper has no DISubprogram, so it only surfaced once the
// helper was inlined into the kernel, as an assertion in LexicalScopes.
//
// If these helpers ever gain a DISubprogram of their own, the checks below
// should be relaxed to require that the scope describes the helper itself.

void foo() {
  int T = 0;
#pragma omp target teams distribute parallel for reduction(+ : T)
  for (int i = 0; i < 1024; ++i)
    T += i;
}

// Sanity check that debug info is being emitted at all: the outlined function
// does have a DISubprogram attached.
// CHECK: define internal void @{{.*}}_debug__(
// CHECK-SAME: !dbg ![[#]]

// The teams-level and the parallel-level combiner. Neither may reference a
// debug location anywhere in its body.
// CHECK:     define internal void @"{{.*}}_omp$reduction$reduction_func"(
// CHECK-NOT:   !dbg
// CHECK:     }

// CHECK:     define internal void @"{{.*}}_omp$reduction$reduction_func"(
// CHECK-NOT:   !dbg
// CHECK:     }
