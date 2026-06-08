// RUN: fir-opt --function-attr="set-noalias=true" %s | FileCheck %s

// Test that noalias is not set on arguments of functions containing
// OpenMP synchronization or work-sharing constructs, because shared
// variables may cause multiple threads to pass the same address.

// CHECK-LABEL:   func.func @test_omp_single(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<f64> {fir.bindc_name = "tmp"}) {
func.func @test_omp_single(%arg0: !fir.ref<f64> {fir.bindc_name = "tmp"}) {
  omp.single {
    omp.terminator
  }
  return
}

// CHECK-LABEL:   func.func @test_omp_critical(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<f64> {fir.bindc_name = "tmp"}) {
func.func @test_omp_critical(%arg0: !fir.ref<f64> {fir.bindc_name = "tmp"}) {
  omp.critical {
    omp.terminator
  }
  return
}

// CHECK-LABEL:   func.func @test_omp_barrier(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<f64> {fir.bindc_name = "tmp"}) {
func.func @test_omp_barrier(%arg0: !fir.ref<f64> {fir.bindc_name = "tmp"}) {
  omp.barrier
  return
}

// CHECK-LABEL:   func.func @test_omp_master(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<f64> {fir.bindc_name = "tmp"}) {
func.func @test_omp_master(%arg0: !fir.ref<f64> {fir.bindc_name = "tmp"}) {
  omp.master {
    omp.terminator
  }
  return
}

// CHECK-LABEL:   func.func @test_omp_parallel(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<f64> {fir.bindc_name = "tmp"}) {
func.func @test_omp_parallel(%arg0: !fir.ref<f64> {fir.bindc_name = "tmp"}) {
  omp.parallel {
    omp.terminator
  }
  return
}

// A function without OMP constructs should still get noalias.
// CHECK-LABEL:   func.func @test_no_omp(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<f64> {fir.bindc_name = "tmp", llvm.noalias}) {
func.func @test_no_omp(%arg0: !fir.ref<f64> {fir.bindc_name = "tmp"}) {
  return
}
