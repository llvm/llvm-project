! This test checks lowering of OpenMP DISTRIBUTE SIMD composite constructs.

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPdistribute_simd_aligned(
subroutine distribute_simd_aligned(A)
  use iso_c_binding
  type(c_ptr) :: A

  !$omp teams

  ! CHECK:      omp.distribute
  ! CHECK-NOT:  aligned({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: aligned({{.*}})
  !$omp distribute simd aligned(A:256)
  do index_ = 1, 10
    call c_test_call(A)
  end do
  !$omp end distribute simd

  !$omp end teams
end subroutine distribute_simd_aligned

! CHECK-LABEL: func.func @_QPdistribute_simd_safelen(
subroutine distribute_simd_safelen()
  !$omp teams

  ! CHECK:      omp.distribute
  ! CHECK-NOT:  safelen({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: safelen({{.*}})
  !$omp distribute simd safelen(4)
  do index_ = 1, 10
  end do
  !$omp end distribute simd

  !$omp end teams
end subroutine distribute_simd_safelen

! CHECK-LABEL: func.func @_QPdistribute_simd_simdlen(
subroutine distribute_simd_simdlen()
  !$omp teams

  ! CHECK:      omp.distribute
  ! CHECK-NOT:  simdlen({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: simdlen({{.*}})
  !$omp distribute simd simdlen(4)
  do index_ = 1, 10
  end do
  !$omp end distribute simd

  !$omp end teams
end subroutine distribute_simd_simdlen

! CHECK-LABEL: func.func @_QPdistribute_simd_private(
subroutine distribute_simd_private()
  integer, allocatable :: tmp
  ! CHECK:      omp.teams
  !$omp teams
  ! CHECK:      omp.distribute
  ! CHECK:      omp.simd
  ! CHECK-SAME: private(@[[PRIV_BOX_SYM:.*]] %{{.*}} -> %[[PRIV_BOX:.*]], @[[PRIV_IVAR_SYM:.*]] %{{.*}} -> %[[PRIV_IVAR:.*]] : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<i32>)
  ! CHECK-NEXT: omp.loop_nest (%[[IVAR:.*]]) : i32
  !$omp distribute simd private(tmp)
  do index_ = 1, 10
  ! CHECK:      %[[PRIV_BOX_DECL:.*]]:2 = hlfir.declare %[[PRIV_BOX]]
  ! CHECK:      %[[PRIV_IVAR_DECL:.*]]:2 = hlfir.declare %[[PRIV_IVAR]]
  ! CHECK:      hlfir.assign %[[IVAR]] to %[[PRIV_IVAR_DECL]]#0
  end do
  !$omp end distribute simd
  !$omp end teams
end subroutine distribute_simd_private
