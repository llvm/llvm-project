// RUN: fir-opt %s           | FileCheck %s --enable-var-scope
// RUN: fir-opt %s | fir-opt | FileCheck %s --enable-var-scope

// CHECK-LABEL: @omp_canonloop_multiregion(
func.func @omp_canonloop_multiregion(%c : i1) -> () {
  %c42_i32 = arith.constant 42: i32
  %canonloop1 = omp.new_cli
  %canonloop2 = omp.new_cli
  %canonloop3 = omp.new_cli
  fir.if %c {
    // CHECK: omp.canonical_loop(%canonloop_r0) %iv_r0 : i32 in range(%c42_i32) {
    omp.canonical_loop(%canonloop1) %iv1 : i32 in range(%c42_i32) {
      omp.terminator
    }
  } else {
    // CHECK: omp.canonical_loop(%canonloop_r1_s0) %iv_r1_s0 : i32 in range(%c42_i32) {
    omp.canonical_loop(%canonloop2)  %iv2 : i32 in range(%c42_i32) {
      omp.terminator
    }
    // CHECK: omp.canonical_loop(%canonloop_r1_s1) %iv_r1_s1 : i32 in range(%c42_i32) {
    omp.canonical_loop(%canonloop3)  %iv3 : i32 in range(%c42_i32) {
      omp.terminator
    }
  }

  // CHECK:   fir.unreachable
  fir.unreachable
}
