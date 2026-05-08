! Test that acc.declare attributes are propagated to fir.global when
! a module with !$acc declare is USEd from a separately compiled file.

! RUN: split-file %s %t
! RUN: bbc -fopenacc -emit-hlfir %t/mod.f90 -o %t/mod.mlir --module=%t
! RUN: bbc -fopenacc -emit-hlfir %t/use.f90 -o - -I %t | FileCheck %s

//--- mod.f90
module acc_declare_mod
  integer, parameter :: n = 100
  real, dimension(n) :: aa, bb
  real :: coef
  !$acc declare create(aa)
  !$acc declare copyin(coef)
end module

//--- use.f90
subroutine use_mod()
  use acc_declare_mod
  implicit none
  integer :: i
  !$acc parallel loop
  do i = 1, n
    aa(i) = bb(i) + coef
  end do
end subroutine

! CHECK: fir.global @_QMacc_declare_modEaa {acc.declare = #acc.declare<dataClause = acc_create>} : !fir.array<100xf32>
! CHECK: fir.global @_QMacc_declare_modEcoef {acc.declare = #acc.declare<dataClause = acc_copyin>} : f32
