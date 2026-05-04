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
! CHECK: omp.target
! CHECK:   omp.teams
! CHECK:     %{{.*}} = omp.groupprivate @_QMmEx device_type (any) : !fir.ref<i32>
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
! CHECK: omp.target
! CHECK:   omp.teams
! CHECK:     %{{.*}} = omp.groupprivate @blk_ device_type (any) : !fir.ref<!fir.array<12xi8>>
! CHECK:     fir.coordinate_of
! CHECK:     fir.convert
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
! CHECK: omp.teams
! CHECK:   %{{.*}} = omp.groupprivate @_QFtest_local_save_groupprivateElocal_x device_type (any) : !fir.ref<i32>
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
! CHECK: omp.target
! CHECK:   omp.teams
! CHECK-DAG: omp.groupprivate @_QMm_multiEgp_a
! CHECK-DAG: omp.groupprivate @_QMm_multiEgp_b
! CHECK-DAG: omp.groupprivate @_QMm_multiEgp_c
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

! Test 5: Same variable referenced multiple times produces only one op.
! CHECK-LABEL: func.func @_QPtest_repeated_ref_groupprivate
! CHECK: omp.teams
! CHECK: omp.groupprivate @_QMmEx
! CHECK-NOT: omp.groupprivate @_QMmEx
! CHECK: omp.terminator
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
! CHECK-NOT: omp.target
! CHECK: omp.teams
! CHECK:   %{{.*}} = omp.groupprivate @_QMmEx device_type (any) : !fir.ref<i32>
subroutine test_standalone_teams_groupprivate()
  use m

  !$omp teams
    x = 100
  !$omp end teams
end subroutine

! Test 7: Groupprivate variable is not added to target's implicit map_entries.
! CHECK-LABEL: func.func @_QPtest_target_skip_map_groupprivate
! CHECK-NOT: omp.map.info {{.*}}@_QMmEx
! CHECK: omp.target
! CHECK:   omp.teams
! CHECK:     omp.groupprivate @_QMmEx
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
! CHECK: omp.teams
! CHECK-DAG: omp.groupprivate @_QMm_typesEgp_r8 device_type (any) : !fir.ref<f64>
! CHECK-DAG: omp.groupprivate @_QMm_typesEgp_iarr device_type (any) : !fir.ref<!fir.array<4xi32>>
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
! CHECK: omp.teams
! CHECK-DAG: omp.groupprivate @blka_ device_type (any)
! CHECK-DAG: omp.groupprivate @blkb_ device_type (any)
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
! CHECK: omp.teams
! CHECK-DAG: omp.groupprivate @_QMm_dtEgp_h device_type (host) : !fir.ref<i32>
! CHECK-DAG: omp.groupprivate @_QMm_dtEgp_nh device_type (nohost) : !fir.ref<i32>
subroutine test_device_type_groupprivate()
  use m_dt

  !$omp teams
    gp_h = 1
    gp_nh = 2
  !$omp end teams
end subroutine
