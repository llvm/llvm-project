! RUN: export FLANG_DEBUG_BOUNDS=1 && %python %S/test_errors.py %s %flang_fc1
program declaration_array_bounds
  implicit none

  ! ---- Valid cases (no errors expected) ----

  ! Scalar bounds (baseline)
  integer :: a(10)
  integer :: b(2:10)

  ! Array upper bound only
  integer :: c([3, 4, 5])

  ! Array lower and upper bounds, same size
  integer :: d([2, 3] : [10, 20])

  ! Scalar lower, array upper
  integer :: e(2 : [10, 20])

  ! Array lower, scalar upper
  integer :: f([2, 3] : 10)

  ! Using non-literal PARAMETER variables
  integer, parameter :: rank1_parameter_array(3) = [5,5,5]
  integer :: g(rank1_parameter_array)
  integer :: ggg(rank1_parameter_array * 2 : rank1_parameter_array - 1)


  ! ! Negative cases (erros expected)
  integer :: rank1_array(3) = [5,5,5]
  !ERROR: Array (upper) bound must be a constant expression
  integer :: gg(rank1_array)
  integer :: scalar
  !ERROR: Array (lower) bound must be a constant expression
  !ERROR: Array (upper) bound must be a constant expression
  integer :: gggg(rank1_parameter_array + rank1_array : rank1_parameter_array * scalar)

  !ERROR: Must have INTEGER type, but is REAL(4)
  integer :: h([1.2,2.2,3.2]:[1,2,3])
  !ERROR: DECLARATION bounds integer rank-1 arrays must have the same size; lower bounds has 3 elements, upper bounds has 2 elements
  integer :: i([1,2,3]:[3,3])

end program 