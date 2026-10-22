!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

! OpenMP 5.0 2.17.11 depend Clause restrictions for the DEPOBJ dependence type:
!  - Array sections cannot be specified.
!  - List items must be depend objects (scalar integers of omp_depend_kind).

subroutine f00(x, y, obj, objs)
  use iso_c_binding, only: c_intptr_t
  integer :: x
  integer :: y(10)
  integer(c_intptr_t) :: obj
  integer(c_intptr_t) :: objs(10)

  ! A plain integer is not a depend object (wrong kind).
  !ERROR: A list item in a DEPEND clause with the DEPOBJ dependence type must be a depend object (a scalar integer variable of kind omp_depend_kind)
  !$omp task depend(depobj: x)
  !$omp end task

  ! An array section is not a depend object (not scalar).
  !ERROR: A list item in a DEPEND clause with the DEPOBJ dependence type must be a depend object (a scalar integer variable of kind omp_depend_kind)
  !$omp task depend(depobj: objs(1:3))
  !$omp end task

  ! A whole array is not a depend object (not scalar).
  !ERROR: A list item in a DEPEND clause with the DEPOBJ dependence type must be a depend object (a scalar integer variable of kind omp_depend_kind)
  !$omp task depend(depobj: y)
  !$omp end task

  ! A valid depend object (scalar) and a depend-object array element are accepted
  ! by semantics (lowering is not yet implemented).
  !$omp task depend(depobj: obj)
  !$omp end task
  !$omp task depend(depobj: objs(2))
  !$omp end task
end
