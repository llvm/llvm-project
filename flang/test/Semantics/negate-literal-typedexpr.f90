!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! The GenericExprWrapper for the negated literal 2 is nullptr. Usually
! it would be non-null, but contain std::nullopt. Make sure we don't
! crash on this.

!CHECK: omp.teams

subroutine f(array)
  implicit none
  real :: array(:)
  integer s
  !$omp target teams distribute parallel do
  do s = 1, 3
    !The "2" in "Negate 2" does not have TypedExpr.
    array(-2 + s) = 1.0
  end do
  !$omp end target teams distribute parallel do
end

