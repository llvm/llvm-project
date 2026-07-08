! Check that the INDIRECT modifier is preserved (merged with logical OR) when a
! procedure is named by more than one DECLARE TARGET directive. A prior
! `indirect = true` must not be dropped, either by the device_type merge to
! `any` or by an early return when the device_type already matches.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

module m1
  implicit none
contains
  ! A later directive adds `indirect` with the same (default) device_type.
  ! CHECK: func.func @_QMm1Pfoo1() -> i32 attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), automap = false, indirect = true>}
  function foo1() result(i)
    !$omp declare target enter(foo1)
    !$omp declare target enter(foo1) indirect(.true.)
    integer :: i
    i = 1
  end function
end module

module m2
  implicit none
contains
  ! `indirect` first, plain second: it must stay set.
  ! CHECK: func.func @_QMm2Pfoo2() -> i32 attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), automap = false, indirect = true>}
  function foo2() result(i)
    !$omp declare target enter(foo2) indirect(.true.)
    !$omp declare target enter(foo2)
    integer :: i
    i = 1
  end function
end module

module m3
  implicit none
contains
  ! `indirect` (device_type any) followed by a device_type(nohost) declaration:
  ! the device type merges to `any` and the `indirect = true` must be carried
  ! over rather than overwritten.
  ! CHECK: func.func @_QMm3Pfoo3() -> i32 attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), automap = false, indirect = true>}
  function foo3() result(i)
    !$omp declare target enter(foo3) indirect(.true.)
    !$omp declare target enter(foo3) device_type(nohost)
    integer :: i
    i = 1
  end function
end module
