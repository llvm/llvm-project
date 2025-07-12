! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s


subroutine omp_unroll_heuristic_nested02(outer_lb, outer_ub, outer_inc, inner_lb, inner_ub, inner_inc)
  integer res, i, j, inner_lb, inner_ub, inner_inc, outer_lb, outer_ub, outer_inc

  !$omp unroll
  do i = outer_lb, outer_ub, outer_inc
    !$omp unroll
    do j = inner_lb, inner_ub, inner_inc
      res = i + j
    end do
    !$omp end unroll
  end do
  !$omp end unroll

end subroutine omp_unroll_heuristic_nested02


!CHECK-LABEL: func.func @_QPomp_unroll_heuristic_nested02(%arg0: !fir.ref<i32> {fir.bindc_name = "outer_lb"}, %arg1: !fir.ref<i32> {fir.bindc_name = "outer_ub"}, %arg2: !fir.ref<i32> {fir.bindc_name = "outer_inc"}, %arg3: !fir.ref<i32> {fir.bindc_name = "inner_lb"}, %arg4: !fir.ref<i32> {fir.bindc_name = "inner_ub"}, %arg5: !fir.ref<i32> {fir.bindc_name = "inner_inc"}) {
!CHECK:      %c0_i32 = arith.constant 0 : i32
!CHECK-NEXT: %c1_i32 = arith.constant 1 : i32
!CHECK-NEXT: %18 = arith.cmpi slt, %17, %c0_i32 : i32
!CHECK-NEXT: %19 = arith.subi %c0_i32, %17 : i32
!CHECK-NEXT: %20 = arith.select %18, %19, %17 : i32
!CHECK-NEXT: %21 = arith.select %18, %16, %15 : i32
!CHECK-NEXT: %22 = arith.select %18, %15, %16 : i32
!CHECK-NEXT: %23 = arith.subi %22, %21 overflow<nuw> : i32
!CHECK-NEXT: %24 = arith.divui %23, %20 : i32
!CHECK-NEXT: %25 = arith.addi %24, %c1_i32 overflow<nuw> : i32
!CHECK-NEXT: %26 = arith.cmpi slt, %22, %21 : i32
!CHECK-NEXT: %27 = arith.select %26, %c0_i32, %25 : i32
!CHECK-NEXT: %canonloop_s0 = omp.new_cli
!CHECK-NEXT: omp.canonical_loop(%canonloop_s0) %iv : i32 in range(%27) {
!CHECK-NEXT:   %28 = arith.muli %iv, %17 : i32
!CHECK-NEXT:   %29 = arith.addi %15, %28 : i32
!CHECK-NEXT:   hlfir.assign %29 to %14#0 : i32, !fir.ref<i32>
!CHECK-NEXT:   %30 = fir.alloca i32 {bindc_name = "j", pinned, uniq_name = "_QFomp_unroll_heuristic_nested02Ej"}
!CHECK-NEXT:   %31:2 = hlfir.declare %30 {uniq_name = "_QFomp_unroll_heuristic_nested02Ej"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:   %32 = fir.load %4#0 : !fir.ref<i32>
!CHECK-NEXT:   %33 = fir.load %5#0 : !fir.ref<i32>
!CHECK-NEXT:   %34 = fir.load %3#0 : !fir.ref<i32>
!CHECK-NEXT:   %c0_i32_0 = arith.constant 0 : i32
!CHECK-NEXT:   %c1_i32_1 = arith.constant 1 : i32
!CHECK-NEXT:   %35 = arith.cmpi slt, %34, %c0_i32_0 : i32
!CHECK-NEXT:   %36 = arith.subi %c0_i32_0, %34 : i32
!CHECK-NEXT:   %37 = arith.select %35, %36, %34 : i32
!CHECK-NEXT:   %38 = arith.select %35, %33, %32 : i32
!CHECK-NEXT:   %39 = arith.select %35, %32, %33 : i32
!CHECK-NEXT:   %40 = arith.subi %39, %38 overflow<nuw> : i32
!CHECK-NEXT:   %41 = arith.divui %40, %37 : i32
!CHECK-NEXT:   %42 = arith.addi %41, %c1_i32_1 overflow<nuw> : i32
!CHECK-NEXT:   %43 = arith.cmpi slt, %39, %38 : i32
!CHECK-NEXT:   %44 = arith.select %43, %c0_i32_0, %42 : i32
!CHECK-NEXT:   %canonloop_s0_s0 = omp.new_cli
!CHECK-NEXT:   omp.canonical_loop(%canonloop_s0_s0) %iv_2 : i32 in range(%44) {
!CHECK-NEXT:     %45 = arith.muli %iv_2, %34 : i32
!CHECK-NEXT:     %46 = arith.addi %32, %45 : i32
!CHECK-NEXT:     hlfir.assign %46 to %31#0 : i32, !fir.ref<i32>
!CHECK-NEXT:     %47 = fir.load %14#0 : !fir.ref<i32>
!CHECK-NEXT:     %48 = fir.load %31#0 : !fir.ref<i32>
!CHECK-NEXT:     %49 = arith.addi %47, %48 : i32
!CHECK-NEXT:     hlfir.assign %49 to %12#0 : i32, !fir.ref<i32>
!CHECK-NEXT:     omp.terminator
!CHECK-NEXT:   }
!CHECK-NEXT:   omp.unroll_heuristic(%canonloop_s0_s0)
!CHECK-NEXT:   omp.terminator
!CHECK-NEXT: }
!CHECK-NEXT: omp.unroll_heuristic(%canonloop_s0)
!CHECK-NEXT: return
