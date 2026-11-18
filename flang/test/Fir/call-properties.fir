// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s
// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=LLVM
// RUN: split-file %S/../Lower/OpenMP/assumption.f90 %t
// RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=52 -module-dir %t -o - %t/bare.f90 | FileCheck %s --check-prefix=SCOPE
// RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=60 -module-dir %t -o - %t/conditional.f90 | FileCheck %s --check-prefix=CONDITIONAL
// REQUIRES: x86-registered-target

// Preserve the direct callee alongside converted fast-math properties and
// discardable attributes. An indirect call must keep its function operand.
func.func private @callee(f32) -> f32

func.func @direct(%arg: f32) -> f32 {
  %result = fir.call @callee(%arg) fastmath<contract> {default_func_attrs = {llvm.assume = "omp_no_openmp"}, test.tag = "direct"} : (f32) -> f32
  return %result : f32
}

// CHECK-LABEL: llvm.func @direct(
// CHECK: llvm.call {{.*}}callee
// CHECK-SAME: default_func_attrs = {llvm.assume = "omp_no_openmp"}{{.*}}fastmathFlags = #llvm.fastmath<contract>{{.*}}test.tag = "direct"

func.func @indirect(%callee: (f32) -> f32, %arg: f32) -> f32 {
  %result = fir.call %callee(%arg) fastmath<contract> {default_func_attrs = {llvm.assume = "omp_no_openmp"}, test.tag = "indirect"} : (f32) -> f32
  return %result : f32
}

// CHECK-LABEL: llvm.func @indirect(
// CHECK: llvm.call %{{[^ (]+}}
// CHECK-SAME: default_func_attrs = {llvm.assume = "omp_no_openmp"}{{.*}}fastmathFlags = #llvm.fastmath<contract>{{.*}}test.tag = "indirect"

// LLVM-LABEL: define float {{.*}}direct
// LLVM: call contract float {{.*}}callee
// LLVM-SAME: #[[ASSUMPTIONS:[0-9]+]]
// LLVM-LABEL: define float {{.*}}indirect
// LLVM: call contract float %{{[^ (]+}}
// LLVM-SAME: #[[ASSUMPTIONS]]
// LLVM: attributes #[[ASSUMPTIONS]] = { "llvm.assume"="omp_no_openmp" }

// SCOPE-LABEL: define void {{.*}}assume_no_openmp
// SCOPE-NOT: #
// SCOPE-SAME: {{$}}
// SCOPE: call void {{.*}}external_helper
// SCOPE-NOT: #
// SCOPE-SAME: {{$}}
// SCOPE: call void {{.*}}external_helper
// SCOPE-SAME: #[[ASSUMPTIONS:[0-9]+]]
// SCOPE: call void {{.*}}external_helper
// SCOPE-NOT: #
// SCOPE-SAME: {{$}}
// SCOPE-LABEL: define void {{.*}}assume_no_openmp_loose
// SCOPE-NOT: #
// SCOPE-SAME: {{$}}
// SCOPE: call void @llvm.assume
// SCOPE: call void {{.*}}external_helper
// SCOPE-SAME: #[[ASSUMPTIONS]]
// SCOPE-LABEL: define void {{.*}}assume_no_openmp_holds_first
// SCOPE-NOT: #
// SCOPE-SAME: {{$}}
// SCOPE: call void @llvm.assume
// SCOPE: call void {{.*}}external_helper
// SCOPE-SAME: #[[ASSUMPTIONS]]
// SCOPE-LABEL: define void {{.*}}assume_no_openmp_nested
// SCOPE-NOT: #
// SCOPE-SAME: {{$}}
// SCOPE: call void %{{[^ (]+}}
// SCOPE-NOT: #
// SCOPE-SAME: {{$}}
// SCOPE: call void %{{[^ (]+}}
// SCOPE-SAME: #[[ASSUMPTIONS]]
// SCOPE: call void %{{[^ (]+}}
// SCOPE-SAME: #[[ASSUMPTIONS]]
// SCOPE: call void %{{[^ (]+}}
// SCOPE-SAME: #[[ASSUMPTIONS]]
// SCOPE: call void %{{[^ (]+}}
// SCOPE-NOT: #
// SCOPE-SAME: {{$}}
// SCOPE: attributes #[[ASSUMPTIONS]] = { "llvm.assume"="omp_no_openmp" }

// CONDITIONAL-LABEL: define void {{.*}}assume_no_openmp_conditional
// CONDITIONAL-NOT: #
// CONDITIONAL-SAME: {{$}}
// CONDITIONAL: call void {{.*}}external_helper
// CONDITIONAL-NOT: #
// CONDITIONAL-SAME: {{$}}
// CONDITIONAL: call void @llvm.assume
// CONDITIONAL: call void {{.*}}external_helper
// CONDITIONAL-SAME: #[[ASSUMPTIONS:[0-9]+]]
// CONDITIONAL: call void %{{[^ (]+}}
// CONDITIONAL-SAME: #[[ASSUMPTIONS]]
// CONDITIONAL: call void @llvm.assume
// CONDITIONAL: call void {{.*}}external_helper
// CONDITIONAL-NOT: #
// CONDITIONAL-SAME: {{$}}
// CONDITIONAL: call void %{{[^ (]+}}
// CONDITIONAL-SAME: #[[ASSUMPTIONS]]
// CONDITIONAL: call void {{.*}}external_helper
// CONDITIONAL-NOT: #
// CONDITIONAL-SAME: {{$}}
// CONDITIONAL: call void {{.*}}external_helper
// CONDITIONAL-NOT: #
// CONDITIONAL-SAME: {{$}}
// CONDITIONAL: ret void
// CONDITIONAL: attributes #[[ASSUMPTIONS]] = { "llvm.assume"="omp_no_openmp" }
