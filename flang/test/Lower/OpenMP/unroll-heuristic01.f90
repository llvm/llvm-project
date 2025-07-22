! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s


subroutine omp_unroll_heuristic01(lb, ub, inc)
  integer res, i, lb, ub, inc

  !$omp unroll
  do i = lb, ub, inc
    res = i
  end do
  !$omp end unroll

end subroutine omp_unroll_heuristic01


!CHECK-LABEL: func.func @_QPomp_unroll_heuristic01(
!CHECK:      %c0_i32 = arith.constant 0 : i32
!CHECK-NEXT: %c1_i32 = arith.constant 1 : i32
!CHECK-NEXT: %13 = arith.cmpi slt, %12, %c0_i32 : i32
!CHECK-NEXT: %14 = arith.subi %c0_i32, %12 : i32
!CHECK-NEXT: %15 = arith.select %13, %14, %12 : i32
!CHECK-NEXT: %16 = arith.select %13, %11, %10 : i32
!CHECK-NEXT: %17 = arith.select %13, %10, %11 : i32
!CHECK-NEXT: %18 = arith.subi %17, %16 overflow<nuw> : i32
!CHECK-NEXT: %19 = arith.divui %18, %15 : i32
!CHECK-NEXT: %20 = arith.addi %19, %c1_i32 overflow<nuw> : i32
!CHECK-NEXT: %21 = arith.cmpi slt, %17, %16 : i32
!CHECK-NEXT: %22 = arith.select %21, %c0_i32, %20 : i32
!CHECK-NEXT: %canonloop_s0 = omp.new_cli
!CHECK-NEXT: omp.canonical_loop(%canonloop_s0) %iv : i32 in range(%22) {
!CHECK-NEXT:   %23 = arith.muli %iv, %12 : i32
!CHECK-NEXT:   %24 = arith.addi %10, %23 : i32
!CHECK-NEXT:   hlfir.assign %24 to %9#0 : i32, !fir.ref<i32>
!CHECK-NEXT:   %25 = fir.load %9#0 : !fir.ref<i32>
!CHECK-NEXT:   hlfir.assign %25 to %6#0 : i32, !fir.ref<i32>
!CHECK-NEXT:   omp.terminator
!CHECK-NEXT: }
!CHECK-NEXT: omp.unroll_heuristic(%canonloop_s0)
!CHECK-NEXT: return
