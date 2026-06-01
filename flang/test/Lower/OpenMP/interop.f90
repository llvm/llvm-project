! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

!===============================================================================
! Interop Init — target
!===============================================================================

!CHECK-LABEL: func.func @_QPtest_interop_init(
!CHECK-SAME:    %[[ARG:.*]]: !fir.ref<i64> {fir.bindc_name = "obj"})
!CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ARG]]
!CHECK:         omp.interop.init %[[DECL]]#0 : !fir.ref<i64> interop_types([#omp<interop_type(target)>])
subroutine test_interop_init(obj)
  integer(8) :: obj
  !$omp interop init(target: obj)
end subroutine

!===============================================================================
! Interop Init — targetsync
!===============================================================================

!CHECK-LABEL: func.func @_QPtest_interop_init_targetsync(
!CHECK:         omp.interop.init %{{.*}} : !fir.ref<i64> interop_types([#omp<interop_type(targetsync)>])
subroutine test_interop_init_targetsync(obj)
  integer(8) :: obj
  !$omp interop init(targetsync: obj)
end subroutine

!===============================================================================
! Interop Init — targetsync, target (both)
!===============================================================================

!CHECK-LABEL: func.func @_QPtest_interop_init_both(
!CHECK:         omp.interop.init %{{.*}} : !fir.ref<i64> interop_types([#omp<interop_type(targetsync)>, #omp<interop_type(target)>])
subroutine test_interop_init_both(obj)
  integer(8) :: obj
  !$omp interop init(targetsync, target: obj)
end subroutine

!===============================================================================
! Interop Init — untyped (OpenMP 6.0: interop-type modifier is optional)
!===============================================================================

!CHECK-LABEL: func.func @_QPtest_interop_init_untyped(
!CHECK:         omp.interop.init %{{.*}} : !fir.ref<i64> interop_types([])
subroutine test_interop_init_untyped(obj)
  integer(8) :: obj
  !$omp interop init(obj)
end subroutine

!===============================================================================
! Interop Init — nowait
!===============================================================================

!CHECK-LABEL: func.func @_QPtest_interop_init_nowait(
!CHECK:         omp.interop.init %{{.*}} : !fir.ref<i64> interop_types([#omp<interop_type(target)>]) nowait
subroutine test_interop_init_nowait(obj)
  integer(8) :: obj
  !$omp interop init(target: obj) nowait
end subroutine

!===============================================================================
! Interop Init — device clause
!===============================================================================

!CHECK-LABEL: func.func @_QPtest_interop_device(
!CHECK:         %[[DEV:.*]] = fir.load %{{.*}} : !fir.ref<i32>
!CHECK:         omp.interop.init %{{.*}} : !fir.ref<i64> interop_types([#omp<interop_type(target)>]) device(%[[DEV]] : i32)
subroutine test_interop_device(obj, dev)
  integer(8) :: obj
  integer :: dev
  !$omp interop device(dev) init(target: obj)
end subroutine

!===============================================================================
! Interop Init — prefer_type with string FR identifiers
!===============================================================================

!CHECK-LABEL: func.func @_QPtest_interop_prefer_str(
!CHECK:         omp.interop.init %{{.*}} : !fir.ref<i64> interop_types([#omp<interop_type(targetsync)>]) prefer_type([1, 6])
subroutine test_interop_prefer_str(obj)
  integer(8) :: obj
  !$omp interop init(prefer_type("cuda", "level_zero"), targetsync: obj)
end subroutine

!===============================================================================
! Interop Init — prefer_type with string FR identifier (hip)
!===============================================================================

!CHECK-LABEL: func.func @_QPtest_interop_prefer_hip(
!CHECK:         omp.interop.init %{{.*}} : !fir.ref<i64> interop_types([#omp<interop_type(target)>]) prefer_type([5])
subroutine test_interop_prefer_hip(obj)
  integer(8) :: obj
  !$omp interop init(prefer_type("hip"), target: obj)
end subroutine

!===============================================================================
! Interop Init — prefer_type with integer constant FR identifiers
!===============================================================================

!CHECK-LABEL: func.func @_QPtest_interop_prefer_int(
!CHECK:         omp.interop.init %{{.*}} : !fir.ref<i64> interop_types([#omp<interop_type(targetsync)>]) prefer_type([4])
subroutine test_interop_prefer_int(obj)
  integer(8) :: obj
  integer, parameter :: omp_ifr_sycl = 4
  !$omp interop init(prefer_type(omp_ifr_sycl), targetsync: obj)
end subroutine

!===============================================================================
! Interop Use
!===============================================================================

!CHECK-LABEL: func.func @_QPtest_interop_use(
!CHECK:         omp.interop.use %{{.*}} : !fir.ref<i64>
subroutine test_interop_use(obj)
  integer(8) :: obj
  !$omp interop use(obj)
end subroutine

!===============================================================================
! Interop Use — nowait
!===============================================================================

!CHECK-LABEL: func.func @_QPtest_interop_use_nowait(
!CHECK:         omp.interop.use %{{.*}} : !fir.ref<i64> nowait
subroutine test_interop_use_nowait(obj)
  integer(8) :: obj
  !$omp interop use(obj) nowait
end subroutine

!===============================================================================
! Interop Use — device clause
!===============================================================================

!CHECK-LABEL: func.func @_QPtest_interop_use_device(
!CHECK:         %[[DEV:.*]] = fir.load %{{.*}} : !fir.ref<i32>
!CHECK:         omp.interop.use %{{.*}} : !fir.ref<i64> device(%[[DEV]] : i32)
subroutine test_interop_use_device(obj, dev)
  integer(8) :: obj
  integer :: dev
  !$omp interop use(obj) device(dev)
end subroutine

!===============================================================================
! Interop Destroy
!===============================================================================

!CHECK-LABEL: func.func @_QPtest_interop_destroy(
!CHECK:         omp.interop.destroy %{{.*}} : !fir.ref<i64>
subroutine test_interop_destroy(obj)
  integer(8) :: obj
  !$omp interop destroy(obj)
end subroutine

!===============================================================================
! Interop Destroy — nowait
!===============================================================================

!CHECK-LABEL: func.func @_QPtest_interop_destroy_nowait(
!CHECK:         omp.interop.destroy %{{.*}} : !fir.ref<i64> nowait
subroutine test_interop_destroy_nowait(obj)
  integer(8) :: obj
  !$omp interop destroy(obj) nowait
end subroutine

!===============================================================================
! Interop Destroy — device clause
!===============================================================================

!CHECK-LABEL: func.func @_QPtest_interop_destroy_device(
!CHECK:         %[[DEV:.*]] = fir.load %{{.*}} : !fir.ref<i32>
!CHECK:         omp.interop.destroy %{{.*}} : !fir.ref<i64> device(%[[DEV]] : i32)
subroutine test_interop_destroy_device(obj, dev)
  integer(8) :: obj
  integer :: dev
  !$omp interop destroy(obj) device(dev)
end subroutine
