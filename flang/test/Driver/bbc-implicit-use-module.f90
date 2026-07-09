! Test bbc's test-only implicit USE module hook.

! RUN: split-file %s %t
! RUN: mkdir -p %t/self
! RUN: bbc -module %t/self -implicit-use-module implicit_mod %t/implicit_mod.f90 -o /dev/null
! RUN: bbc -module %t %t/implicit_mod.f90 -o /dev/null
! RUN: bbc -emit-hlfir -fopenacc -I %t -implicit-use-module implicit_mod %t/use_implicit.f90 -o - | FileCheck %s

!--- implicit_mod.f90
module implicit_mod
  implicit none
  integer :: common_value
  integer :: module_value = 42
  common /implicit_common/ common_value
contains
  subroutine contained
    integer :: x
    x = module_value
  end subroutine
end module

!--- use_implicit.f90
subroutine use_implicit
  real :: x
  !$acc data copy(/implicit_common/)
  x = module_value + common_value
  !$acc end data
end subroutine

! CHECK-LABEL: func.func @_QPuse_implicit()
! CHECK-DAG: fir.address_of(@_QMimplicit_modEmodule_value) : !fir.ref<i32>
! CHECK-DAG: %[[COMMON:.*]] = fir.address_of(@implicit_common_) : !fir.ref<!fir.array<4xi8>>
! CHECK-DAG: acc.copyin varPtr(%[[COMMON]] : !fir.ref<!fir.array<4xi8>>) -> !fir.ref<!fir.array<4xi8>>  {dataClause = #acc<data_clause acc_copy>, name = "implicit_common"}
