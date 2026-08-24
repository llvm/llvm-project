! A module variable marked `declare target` must keep that attribute when it is
! USE-associated in a separately compiled file. The directive lives in the
! module, so the USE-ing file must copy the attribute onto the external global
! it declares. Without it the device pass internalizes the global and
! `target update` reads stale data.

! RUN: split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -module-dir %t %t/mod.f90 -o /dev/null
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -fopenmp-is-target-device -module-dir %t %t/use.f90 -o - | FileCheck %s

!--- mod.f90
module dt_mod
  implicit none
  integer :: arr(100)
  !$omp declare target enter(arr)
end module

!--- use.f90
! Here the global is only an external declaration, but it still must carry the
! declare target attribute.
! CHECK: fir.global @_QMdt_modEarr {{.*}}omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)
subroutine sub(i)
  use dt_mod
  implicit none
  !$omp declare target
  integer :: i
  arr(i) = i
end subroutine
