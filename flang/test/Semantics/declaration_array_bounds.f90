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

  ! Test error for rank > 1, fulfilling constness
  integer, parameter :: rank2_parameter_array(2,2) = reshape([[1,2],[3,4]], [2,2])
  !ERROR: Integer array used as upper bounds in DECLARATION must be rank-1 but is rank-2
  integer :: j(rank2_parameter_array)
  ! Test combined bounds error, first bound as before but second bound as wrong rank
  ! and nonconst
  integer :: rank3_array(2,2,2)
  !ERROR: Integer array used as lower bounds in DECLARATION must be rank-1 but is rank-2
  !ERROR: Array (upper) bound must be a constant expression
  !ERROR: Integer array used as upper bounds in DECLARATION must be rank-1 but is rank-3
  integer :: k(rank2_parameter_array : rank3_array)

  ! Test that any comma list is parsed as ExplicitShapeSpecList and not rewritten 
  ! to ExplicitShapeBonudsSpec, giving error messages expecting same number of 
  ! aruments as rank of test_array and scalar integers
  !ERROR: Must be a scalar value, but is a rank-1 array
  !ERROR: Must be a scalar value, but is a rank-1 array
  !ERROR: Must be a scalar value, but is a rank-1 array
  !ERROR: Must have INTEGER type, but is REAL(4)
  integer :: test_array([1,2,3] : [2,3,4], 3, [1,2,3], 5.2)
end program 