! Test bbc's test-only implicit USE module hook.

! RUN: split-file %s %t
! RUN: bbc -module %t %t/implicit_mod.f90 -o /dev/null
! RUN: bbc -emit-hlfir -I %t -implicit-use-module implicit_mod %t/use_implicit.f90 -o - | FileCheck %s

!--- implicit_mod.f90
module implicit_mod
  implicit none
  integer :: module_value = 42
end module

!--- use_implicit.f90
subroutine use_implicit
  integer :: x
  x = module_value
end subroutine

! CHECK-LABEL: func.func @_QPuse_implicit()
! CHECK: fir.address_of(@_QMimplicit_modEmodule_value) : !fir.ref<i32>
