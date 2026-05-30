// RUN: fir-opt %s --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-data,acc-recipe-materialization)" -split-input-file | FileCheck %s

// Verify that optional scalar (Fortran OPTIONAL dummy) gets the optional
// firstprivate recipe from acc-implicit-data and that acc-recipe-materialization
// inlines the fir.is_present / fir.if / fir.load / hlfir.assign body.

// -----

func.func @test_optional_firstprivate(%arg0: !fir.ref<i32> {fir.bindc_name = "x", fir.optional}) {
  acc.parallel {
    %load = fir.load %arg0 : !fir.ref<i32>
    acc.yield
  }
  return
}

// Recipe exists right after acc-implicit-data (before acc-recipe-materialization)
// IMPLICIT: acc.firstprivate.recipe @firstprivatization_optional_ref_i32

// After acc-recipe-materialization: recipe is materialized, check the inlined body
// CHECK: acc.firstprivate_map varPtr
// CHECK: acc.parallel {
// CHECK: fir.is_present
// CHECK: fir.if %{{.*}} -> (!fir.ref<i32>) {
// CHECK: fir.alloca i32
// CHECK: fir.result
// CHECK: } else {
// CHECK: fir.absent !fir.ref<i32>
// CHECK: fir.result
// CHECK: fir.is_present
// CHECK: fir.if %{{.*}} {
// CHECK: fir.load
// CHECK: hlfir.assign
// CHECK: acc.yield
