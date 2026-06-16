!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

! Test lowering of groupprivate directive to omp.groupprivate.

! CHECK-DAG: fir.global common @blk_
! CHECK-DAG: fir.global common @blka_
! CHECK-DAG: fir.global common @blkb_

! Test 1: Basic groupprivate with single module variable.
module m
  implicit none
  integer, save :: x
  !$omp groupprivate(x)
end module

! CHECK-LABEL: func.func @_QPtest_groupprivate
! CHECK:         omp.target {
! CHECK:           omp.teams {
! CHECK:             %[[GP:.*]] = omp.groupprivate @_QMmEx device_type (any) : !fir.ref<i32>
! CHECK:             %[[DECL:.*]]:2 = hlfir.declare %[[GP]] {uniq_name = "_QMmEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             %[[C10:.*]] = arith.constant 10 : i32
! CHECK:             hlfir.assign %[[C10]] to %[[DECL]]#0 : i32, !fir.ref<i32>
subroutine test_groupprivate()
  use m

  !$omp target
    !$omp teams
      x = 10
    !$omp end teams
  !$omp end target
end subroutine

! Test 2: Groupprivate with common block.
module m2
  implicit none
  integer :: cb_x, cb_y
  real :: cb_z
  common /blk/ cb_x, cb_y, cb_z
  !$omp groupprivate(/blk/)
end module

! CHECK-LABEL: func.func @_QPtest_common_block_groupprivate
! CHECK:         omp.target {
! CHECK:           omp.teams {
! CHECK:             %[[GP:.*]] = omp.groupprivate @blk_ device_type (any) : !fir.ref<!fir.array<12xi8>>
! CHECK:             %[[DECL_X:.*]]:2 = hlfir.declare %{{.*}} storage(%[[GP]][0]) {uniq_name = "_QMm2Ecb_x"}
! CHECK:             %[[DECL_Y:.*]]:2 = hlfir.declare %{{.*}} storage(%[[GP]][4]) {uniq_name = "_QMm2Ecb_y"}
! CHECK:             %[[DECL_Z:.*]]:2 = hlfir.declare %{{.*}} storage(%[[GP]][8]) {uniq_name = "_QMm2Ecb_z"}
! CHECK:             hlfir.assign %{{.*}} to %[[DECL_X]]#0 : i32, !fir.ref<i32>
! CHECK:             hlfir.assign %{{.*}} to %[[DECL_Y]]#0 : i32, !fir.ref<i32>
! CHECK:             hlfir.assign %{{.*}} to %[[DECL_Z]]#0 : f32, !fir.ref<f32>
subroutine test_common_block_groupprivate()
  use m2

  !$omp target
    !$omp teams
      cb_x = 1
      cb_y = 2
      cb_z = 3.0
    !$omp end teams
  !$omp end target
end subroutine

! Test 3: Local SAVE variable promoted to fir.global by globalInitialization.
! CHECK-LABEL: func.func @_QPtest_local_save_groupprivate
! CHECK:         omp.teams {
! CHECK:           %[[GP:.*]] = omp.groupprivate @_QFtest_local_save_groupprivateElocal_x device_type (any) : !fir.ref<i32>
! CHECK:           %[[DECL:.*]]:2 = hlfir.declare %[[GP]] {uniq_name = "_QFtest_local_save_groupprivateElocal_x"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[C42:.*]] = arith.constant 42 : i32
! CHECK:           hlfir.assign %[[C42]] to %[[DECL]]#0 : i32, !fir.ref<i32>
subroutine test_local_save_groupprivate()
  integer, save :: local_x
  !$omp groupprivate(local_x)

  !$omp teams
    local_x = 42
  !$omp end teams
end subroutine

! Test 4: Multiple groupprivate variables in same teams region.
module m_multi
  implicit none
  integer, save :: gp_a
  integer, save :: gp_b
  real,    save :: gp_c
  !$omp groupprivate(gp_a, gp_b, gp_c)
end module

! CHECK-LABEL: func.func @_QPtest_multiple_groupprivate
! CHECK:         omp.target {
! CHECK:           omp.teams {
! CHECK:             %[[GP_A:.*]] = omp.groupprivate @_QMm_multiEgp_a device_type (any) : !fir.ref<i32>
! CHECK:             %[[DECL_A:.*]]:2 = hlfir.declare %[[GP_A]] {uniq_name = "_QMm_multiEgp_a"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             %[[GP_B:.*]] = omp.groupprivate @_QMm_multiEgp_b device_type (any) : !fir.ref<i32>
! CHECK:             %[[DECL_B:.*]]:2 = hlfir.declare %[[GP_B]] {uniq_name = "_QMm_multiEgp_b"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             %[[GP_C:.*]] = omp.groupprivate @_QMm_multiEgp_c device_type (any) : !fir.ref<f32>
! CHECK:             %[[DECL_C:.*]]:2 = hlfir.declare %[[GP_C]] {uniq_name = "_QMm_multiEgp_c"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:             hlfir.assign %{{.*}} to %[[DECL_A]]#0 : i32, !fir.ref<i32>
! CHECK:             hlfir.assign %{{.*}} to %[[DECL_B]]#0 : i32, !fir.ref<i32>
! CHECK:             hlfir.assign %{{.*}} to %[[DECL_C]]#0 : f32, !fir.ref<f32>
subroutine test_multiple_groupprivate()
  use m_multi

  !$omp target
    !$omp teams
      gp_a = 1
      gp_b = 2
      gp_c = 3.0
    !$omp end teams
  !$omp end target
end subroutine

! Test 5: Same variable referenced multiple times produces only one op, and
! every reference accesses the per-team copy via the same hlfir.declare.
! CHECK-LABEL: func.func @_QPtest_repeated_ref_groupprivate
! CHECK:         omp.teams {
! CHECK:           %[[GP:.*]] = omp.groupprivate @_QMmEx device_type (any) : !fir.ref<i32>
! CHECK:           %[[DECL:.*]]:2 = hlfir.declare %[[GP]] {uniq_name = "_QMmEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-NOT:       omp.groupprivate @_QMmEx
! CHECK:           hlfir.assign %{{.*}} to %[[DECL]]#0 : i32, !fir.ref<i32>
! CHECK:           %{{.*}} = fir.load %[[DECL]]#0 : !fir.ref<i32>
! CHECK:           hlfir.assign %{{.*}} to %[[DECL]]#0 : i32, !fir.ref<i32>
! CHECK:           %{{.*}} = fir.load %[[DECL]]#0 : !fir.ref<i32>
! CHECK:           hlfir.assign %{{.*}} to %[[DECL]]#0 : i32, !fir.ref<i32>
! CHECK:           omp.terminator
subroutine test_repeated_ref_groupprivate()
  use m

  !$omp target
    !$omp teams
      x = 10
      x = x + 5
      x = x * 2
    !$omp end teams
  !$omp end target
end subroutine

! Test 6: Standalone teams (no enclosing target) still triggers groupprivate.
! CHECK-LABEL: func.func @_QPtest_standalone_teams_groupprivate
! CHECK-NOT:     omp.target
! CHECK:         omp.teams {
! CHECK:           %[[GP:.*]] = omp.groupprivate @_QMmEx device_type (any) : !fir.ref<i32>
! CHECK:           %[[DECL:.*]]:2 = hlfir.declare %[[GP]] {uniq_name = "_QMmEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[C100:.*]] = arith.constant 100 : i32
! CHECK:           hlfir.assign %[[C100]] to %[[DECL]]#0 : i32, !fir.ref<i32>
subroutine test_standalone_teams_groupprivate()
  use m

  !$omp teams
    x = 100
  !$omp end teams
end subroutine

! Test 7: Groupprivate variable is not added to target's implicit map_entries,
! and the access inside the teams region goes through the omp.groupprivate
! result rather than the original host global.
! CHECK-LABEL: func.func @_QPtest_target_skip_map_groupprivate
! CHECK-NOT:     omp.map.info
! CHECK:         omp.target {
! CHECK:           omp.teams {
! CHECK:             %[[GP:.*]] = omp.groupprivate @_QMmEx device_type (any) : !fir.ref<i32>
! CHECK:             %[[DECL:.*]]:2 = hlfir.declare %[[GP]] {uniq_name = "_QMmEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             %[[C200:.*]] = arith.constant 200 : i32
! CHECK:             hlfir.assign %[[C200]] to %[[DECL]]#0 : i32, !fir.ref<i32>
subroutine test_target_skip_map_groupprivate()
  use m

  !$omp target
    !$omp teams
      x = 200
    !$omp end teams
  !$omp end target
end subroutine

! Test 8: Groupprivate works with various element types (scalar, array, real).
module m_types
  implicit none
  real(8), save :: gp_r8
  integer, save :: gp_iarr(4)
  !$omp groupprivate(gp_r8, gp_iarr)
end module

! CHECK-LABEL: func.func @_QPtest_types_groupprivate
! CHECK:         omp.teams {
! CHECK:           %[[GP_R8:.*]] = omp.groupprivate @_QMm_typesEgp_r8 device_type (any) : !fir.ref<f64>
! CHECK:           %[[DECL_R8:.*]]:2 = hlfir.declare %[[GP_R8]] {uniq_name = "_QMm_typesEgp_r8"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
! CHECK:           %[[GP_IARR:.*]] = omp.groupprivate @_QMm_typesEgp_iarr device_type (any) : !fir.ref<!fir.array<4xi32>>
! CHECK:           %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:           %[[DECL_IARR:.*]]:2 = hlfir.declare %[[GP_IARR]](%[[SHAPE]]) {uniq_name = "_QMm_typesEgp_iarr"}
! CHECK:           hlfir.assign %{{.*}} to %[[DECL_R8]]#0 : f64, !fir.ref<f64>
! CHECK:           %[[ELT:.*]] = hlfir.designate %[[DECL_IARR]]#0 (%{{.*}}) : (!fir.ref<!fir.array<4xi32>>, index) -> !fir.ref<i32>
! CHECK:           hlfir.assign %{{.*}} to %[[ELT]] : i32, !fir.ref<i32>
subroutine test_types_groupprivate()
  use m_types

  !$omp teams
    gp_r8 = 1.0d0
    gp_iarr(1) = 99
  !$omp end teams
end subroutine

! Test 9: Multiple distinct common blocks each get their own omp.groupprivate.
module m_blocks
  implicit none
  integer :: a1, a2
  integer :: b1, b2
  common /blka/ a1, a2
  common /blkb/ b1, b2
  !$omp groupprivate(/blka/, /blkb/)
end module

! CHECK-LABEL: func.func @_QPtest_multi_common_groupprivate
! CHECK:         omp.teams {
! CHECK:           %[[GP_A:.*]] = omp.groupprivate @blka_ device_type (any) : !fir.ref<!fir.array<8xi8>>
! CHECK:           %[[DECL_A1:.*]]:2 = hlfir.declare %{{.*}} storage(%[[GP_A]][0]) {uniq_name = "_QMm_blocksEa1"}
! CHECK:           %[[GP_B:.*]] = omp.groupprivate @blkb_ device_type (any) : !fir.ref<!fir.array<8xi8>>
! CHECK:           %[[DECL_B1:.*]]:2 = hlfir.declare %{{.*}} storage(%[[GP_B]][0]) {uniq_name = "_QMm_blocksEb1"}
! CHECK:           hlfir.assign %{{.*}} to %[[DECL_A1]]#0 : i32, !fir.ref<i32>
! CHECK:           hlfir.assign %{{.*}} to %[[DECL_B1]]#0 : i32, !fir.ref<i32>
subroutine test_multi_common_groupprivate()
  use m_blocks

  !$omp teams
    a1 = 1
    b1 = 2
  !$omp end teams
end subroutine

! Test 10: device_type(host) and device_type(nohost) clauses are honored.
module m_dt
  implicit none
  integer, save :: gp_h
  integer, save :: gp_nh
  !$omp groupprivate(gp_h)  device_type(host)
  !$omp groupprivate(gp_nh) device_type(nohost)
end module

! CHECK-LABEL: func.func @_QPtest_device_type_groupprivate
! CHECK:         omp.teams {
! CHECK:           %[[GP_H:.*]] = omp.groupprivate @_QMm_dtEgp_h device_type (host) : !fir.ref<i32>
! CHECK:           %[[DECL_H:.*]]:2 = hlfir.declare %[[GP_H]] {uniq_name = "_QMm_dtEgp_h"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[GP_NH:.*]] = omp.groupprivate @_QMm_dtEgp_nh device_type (nohost) : !fir.ref<i32>
! CHECK:           %[[DECL_NH:.*]]:2 = hlfir.declare %[[GP_NH]] {uniq_name = "_QMm_dtEgp_nh"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           hlfir.assign %{{.*}} to %[[DECL_H]]#0 : i32, !fir.ref<i32>
! CHECK:           hlfir.assign %{{.*}} to %[[DECL_NH]]#0 : i32, !fir.ref<i32>
subroutine test_device_type_groupprivate()
  use m_dt

  !$omp teams
    gp_h = 1
    gp_nh = 2
  !$omp end teams
end subroutine

! Test 11: The module owning the !$omp groupprivate directive
! is declared AFTER a subroutine that already
! references the variable inside a teams region.
! CHECK-LABEL: func.func @_QPtest_module_after_subroutine_groupprivate
! CHECK:         omp.teams {
! CHECK:           %[[GP:.*]] = omp.groupprivate @_QMm_lateEgp_late device_type (host) : !fir.ref<i32>
! CHECK:           %[[DECL:.*]]:2 = hlfir.declare %[[GP]] {uniq_name = "_QMm_lateEgp_late"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[C7:.*]] = arith.constant 7 : i32
! CHECK:           hlfir.assign %[[C7]] to %[[DECL]]#0 : i32, !fir.ref<i32>
subroutine test_module_after_subroutine_groupprivate()
  use m_late

  !$omp teams
    gp_late = 7
  !$omp end teams
end subroutine

module m_late
  implicit none
  integer, save :: gp_late
  !$omp groupprivate(gp_late) device_type(host)
end module
