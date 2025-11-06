// RUN: fir-opt %s --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-data{enable-implicit-reduction-copy=true})" -split-input-file | FileCheck %s --check-prefix=COPY
// RUN: fir-opt %s --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-data{enable-implicit-reduction-copy=false})" -split-input-file | FileCheck %s --check-prefix=FIRSTPRIVATE

// Test case: integer reduction in parallel loop
// This corresponds to Fortran code:
//   integer :: r, i
//   r = 0
//   !$acc parallel
//   !$acc loop gang reduction(+:r)
//   do i = 1, N
//     r = r + 1
//   enddo
//   !$acc end parallel

acc.reduction.recipe @reduction_add_ref_i32 : !fir.ref<i32> reduction_operator <add> init {
^bb0(%arg0: !fir.ref<i32>):
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.alloca i32
  %1 = fir.declare %0 {uniq_name = "acc.reduction.init"} : (!fir.ref<i32>) -> !fir.ref<i32>
  fir.store %c0_i32 to %1 : !fir.ref<i32>
  acc.yield %1 : !fir.ref<i32>
} combiner {
^bb0(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>):
  %0 = fir.load %arg0 : !fir.ref<i32>
  %1 = fir.load %arg1 : !fir.ref<i32>
  %2 = arith.addi %0, %1 : i32
  fir.store %2 to %arg0 : !fir.ref<i32>
  acc.yield %arg0 : !fir.ref<i32>
}

func.func @test_reduction_implicit_copy() {
  %c1_i32 = arith.constant 1 : i32
  %cN = arith.constant 100 : i32
  %r = fir.alloca i32 {bindc_name = "r", uniq_name = "_QFEr"}
  %i = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  %r_decl = fir.declare %r {uniq_name = "_QFEr"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %i_decl = fir.declare %i {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %c0_i32 = arith.constant 0 : i32
  fir.store %c0_i32 to %r_decl : !fir.ref<i32>

  acc.parallel {
    %red_var = acc.reduction varPtr(%r_decl : !fir.ref<i32>) -> !fir.ref<i32> {name = "r"}
    acc.loop reduction(@reduction_add_ref_i32 -> %red_var : !fir.ref<i32>) control(%iv : i32) = (%c1_i32 : i32) to (%cN : i32) step (%c1_i32 : i32) {
      fir.store %iv to %i_decl : !fir.ref<i32>
      %cur_r = fir.load %red_var : !fir.ref<i32>
      %new_r = arith.addi %cur_r, %c1_i32 : i32
      fir.store %new_r to %red_var : !fir.ref<i32>
      acc.yield
    } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
    acc.yield
  }
  return
}

// When enable-implicit-reduction-copy=true: expect copyin/copyout for reduction variable
// COPY: %[[COPYIN:.*]] = acc.copyin varPtr({{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {dataClause = #acc<data_clause acc_reduction>, implicit = true, name = "r"}
// COPY: acc.copyout accPtr(%[[COPYIN]] : !fir.ref<i32>) to varPtr({{.*}} : !fir.ref<i32>) {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "r"}

// When enable-implicit-reduction-copy=false: expect firstprivate for reduction variable
// FIRSTPRIVATE: acc.firstprivate varPtr({{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "r"}
// FIRSTPRIVATE-NOT: acc.copyin
// FIRSTPRIVATE-NOT: acc.copyout

// -----

// Test case: reduction variable used both in loop and outside (should be firstprivate)
// This corresponds to Fortran code:
//   integer :: r = 0, i, out
//   !$acc parallel num_gangs(1)
//   !$acc loop reduction(+:r) copyout(out)
//   do i = 1, N
//     r = r + 1
//   enddo
//   out = r
//   !$acc end parallel

acc.reduction.recipe @reduction_add_ref_i32 : !fir.ref<i32> reduction_operator <add> init {
^bb0(%arg0: !fir.ref<i32>):
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.alloca i32
  %1 = fir.declare %0 {uniq_name = "acc.reduction.init"} : (!fir.ref<i32>) -> !fir.ref<i32>
  fir.store %c0_i32 to %1 : !fir.ref<i32>
  acc.yield %1 : !fir.ref<i32>
} combiner {
^bb0(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>):
  %0 = fir.load %arg0 : !fir.ref<i32>
  %1 = fir.load %arg1 : !fir.ref<i32>
  %2 = arith.addi %0, %1 : i32
  fir.store %2 to %arg0 : !fir.ref<i32>
  acc.yield %arg0 : !fir.ref<i32>
}

func.func @test_reduction_with_usage_outside_loop() {
  %c1_i32 = arith.constant 1 : i32
  %cN = arith.constant 100 : i32
  %c0_i32 = arith.constant 0 : i32

  %r = fir.alloca i32 {bindc_name = "r", uniq_name = "_QFEr"}
  %i = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  %out = fir.alloca i32 {bindc_name = "out", uniq_name = "_QFEout"}

  %r_decl = fir.declare %r {uniq_name = "_QFEr"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %i_decl = fir.declare %i {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %out_decl = fir.declare %out {uniq_name = "_QFEout"} : (!fir.ref<i32>) -> !fir.ref<i32>
  fir.store %c0_i32 to %r_decl : !fir.ref<i32>

  %out_copyout = acc.create varPtr(%out_decl : !fir.ref<i32>) -> !fir.ref<i32> {dataClause = #acc<data_clause acc_copyout>, name = "out"}
  acc.parallel dataOperands(%out_copyout : !fir.ref<i32>) {
    %red_var = acc.reduction varPtr(%r_decl : !fir.ref<i32>) -> !fir.ref<i32> {name = "r"}
    acc.loop reduction(@reduction_add_ref_i32 -> %red_var : !fir.ref<i32>) control(%iv : i32) = (%c1_i32 : i32) to (%cN : i32) step (%c1_i32 : i32) {
      fir.store %iv to %i_decl : !fir.ref<i32>
      %cur_r = fir.load %red_var : !fir.ref<i32>
      %new_r = arith.addi %cur_r, %c1_i32 : i32
      fir.store %new_r to %red_var : !fir.ref<i32>
      acc.yield
    } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
    // out = r (usage of r outside the loop)
    %final_r = fir.load %r_decl : !fir.ref<i32>
    fir.store %final_r to %out_copyout : !fir.ref<i32>
    acc.yield
  }
  acc.copyout accPtr(%out_copyout : !fir.ref<i32>) to varPtr(%out_decl : !fir.ref<i32>) {dataClause = #acc<data_clause acc_copyout>, name = "out"}
  return
}

// In this case, r should be firstprivate regardless of the flag setting because it's used outside the reduction context
// COPY-LABEL: func.func @test_reduction_with_usage_outside_loop
// COPY: acc.firstprivate varPtr({{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "r"}
// COPY-NOT: acc.copyin varPtr({{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {{.*}} name = "r"

// FIRSTPRIVATE-LABEL: func.func @test_reduction_with_usage_outside_loop
// FIRSTPRIVATE: acc.firstprivate varPtr({{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "r"}
// FIRSTPRIVATE-NOT: acc.copyin varPtr({{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {{.*}} name = "r"

