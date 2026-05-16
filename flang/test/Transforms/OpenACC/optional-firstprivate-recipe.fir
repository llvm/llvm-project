// RUN: fir-opt %s --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-data)" -split-input-file | FileCheck %s

// Optional REAL dummy referenced inside acc.parallel (Fortran OPTIONAL) must get
// @firstprivatization_optional_ref_f32 from acc-implicit-data

// -----

func.func @test_optional_firstprivate_f32(%arg0: !fir.ref<f32> {fir.bindc_name = "copt", fir.optional}) {
  acc.parallel {
    %load = fir.load %arg0 : !fir.ref<f32>
    acc.yield
  }
  return
}

// CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_optional_ref_f32 : !fir.ref<f32> init {
// CHECK: ^bb0(%{{.*}}: !fir.ref<f32>):
// CHECK:   %{{.*}} = fir.is_present %{{.*}} : (!fir.ref<f32>) -> i1
// CHECK:   fir.if %{{.*}} -> (!fir.ref<f32>) {
// CHECK:     %{{.*}} = fir.alloca f32
// CHECK:     fir.result %{{.*}} : !fir.ref<f32>
// CHECK:   } else {
// CHECK:     %{{.*}} = fir.absent !fir.ref<f32>
// CHECK:     fir.result %{{.*}} : !fir.ref<f32>
// CHECK:   }
// CHECK:   acc.yield %{{.*}} : !fir.ref<f32>
// CHECK: } copy {
// CHECK: ^bb0(%{{.*}}: !fir.ref<f32>, %{{.*}}: !fir.ref<f32>):
// CHECK:   %{{.*}} = fir.is_present %{{.*}} : (!fir.ref<f32>) -> i1
// CHECK:   fir.if %{{.*}} {
// CHECK:     %{{.*}} = fir.load %{{.*}} : !fir.ref<f32>
// CHECK:     hlfir.assign %{{.*}} to %{{.*}} temporary_lhs : f32, !fir.ref<f32>
// CHECK:   }
// CHECK:   acc.terminator
// CHECK: }

// bindc_name on the arg does not always flow into acc.firstprivate's name=; minimal FIR gets name = "".
// CHECK: acc.firstprivate varPtr({{.*}} : !fir.ref<f32>) recipe(@firstprivatization_optional_ref_f32) -> !fir.ref<f32> {implicit = true, name = ""}
// CHECK: acc.parallel firstprivate(
