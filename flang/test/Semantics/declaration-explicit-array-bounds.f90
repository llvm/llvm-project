! RUN: %python %S/test_errors.py %s %flang_fc1 -Wautomatic-in-main-program -Wsaved-local-in-spec-expr
! ---- Module with rank-1 array-bounded declarations, USE'd elsewhere ----
module bounds_provider
  implicit none
  integer, parameter :: dims(3) = [5, 5, 5]
  integer, parameter :: lo(2) = [2, 3]
  integer, parameter :: hi(2) = [10, 20]
end module
module consumer
  use bounds_provider
  implicit none
  ! Declare arrays using USE-associated rank-1 parameter arrays
  !ERROR: TODO: Analyze overload for ExplicitShapeBoundsSpec
  integer :: arr_upper(dims)
  !ERROR: TODO: Analyze overload for ExplicitShapeBoundsSpec
  integer :: arr_both(lo : hi)
end module
subroutine sub_consumer()
  use bounds_provider, only: dims, lo, hi
  implicit none
  ! USE'd parameter arrays as bounds in a subroutine
  !ERROR: TODO: Analyze overload for ExplicitShapeBoundsSpec
  integer :: local_arr(dims)
  !ERROR: TODO: Analyze overload for ExplicitShapeBoundsSpec
  integer :: local_arr2(lo : hi)
end subroutine
subroutine sub_use_consumer()
  use consumer, only: arr_upper, arr_both
  implicit none
  ! USE the arrays that were themselves declared with rank-1 array bounds
  arr_upper = 1
  arr_both = 2
end subroutine

module data 
  integer :: rank1_array_module(3) = [5, 5, 5]
  !future_ERROR: Automatic data object 'gg2' may not appear in a module
  !ERROR: TODO: Analyze overload for ExplicitShapeBoundsSpec
  integer :: gg2(rank1_array_module)
  integer, allocatable :: nonconstsize(:)
  !future_ERROR: Rank-1 integer array used as lower bounds in DECLARATION must have constant size
  !future_ERROR: Rank-1 integer array used as upper bounds in DECLARATION must have constant size
  !ERROR: TODO: Analyze overload for ExplicitShapeBoundsSpec
  integer :: gg3(nonconstsize : nonconstsize)
end module 
program declaration_array_bounds
  implicit none

  ! Valid cases (no errors expected)

  ! Array upper bound only
  !ERROR: TODO: Analyze overload for ExplicitShapeBoundsSpec
  integer :: c([3, 4, 5])
  !ERROR: TODO: Analyze overload for ExplicitShapeBoundsSpec
  integer, dimension([3, 4, 5]) :: cc

  ! Array lower and upper bounds, same size
  !ERROR: TODO: Analyze overload for ExplicitShapeBoundsSpec
  integer :: d((/2, 3/) : [10, 20])

  ! Scalar lower, array upper
  !ERROR: TODO: Analyze overload for ExplicitShapeBoundsSpec
  integer :: e(2 : [10, 20])

  ! Array lower, scalar upper
  !ERROR: TODO: Analyze overload for ExplicitShapeBoundsSpec
  integer :: f([2, 3] : 10)

  ! Using non-literal PARAMETER variables
  integer, parameter :: rank1_parameter_array(3) = [5,5,5]
  !ERROR: TODO: Analyze overload for ExplicitShapeBoundsSpec
  integer :: g(rank1_parameter_array)
  !ERROR: TODO: Analyze overload for ExplicitShapeBoundsSpec
  integer :: ggg(rank1_parameter_array * 2 : rank1_parameter_array - 1)


  ! Negative cases (errors expected)
  integer :: rank1_array(3) = [5,5,5]
  ! Use existing error message for constness checking
  !future_PORTABILITY: specification expression refers to local object 'rank1_array' (initialized and saved) [-Wsaved-local-in-spec-expr]
  !future_PORTABILITY: Automatic data object 'gg' should not appear in the specification part of a main program [-Wautomatic-in-main-program]
  !ERROR: TODO: Analyze overload for ExplicitShapeBoundsSpec
  integer :: gg(rank1_array)
  integer :: scalar
  !future_ERROR: Invalid specification expression: reference to local entity 'scalar'
  !future_PORTABILITY: Automatic data object 'gggg' should not appear in the specification part of a main program [-Wautomatic-in-main-program]
  !ERROR: TODO: Analyze overload for ExplicitShapeBoundsSpec
  integer :: gggg(rank1_parameter_array : scalar)

  !ERROR: Must have INTEGER type, but is REAL(4)
  !ERROR: TODO: Analyze overload for ExplicitShapeBoundsSpec
  integer :: h([1.2,2.2,3.2]:[1,2,3])
  !future_ERROR: DECLARATION bounds integer rank-1 arrays must have the same size; lower bounds has 3 elements, upper bounds has 2 elements
  !ERROR: TODO: Analyze overload for ExplicitShapeBoundsSpec
  integer :: i([1,2,3]:[3,3])

  ! Test error for rank > 1, fulfilling constness
  integer, parameter :: rank2_parameter_array(2,2) = reshape([[1,2],[3,4]], [2,2])
  !future_ERROR: Integer array used as upper bounds in DECLARATION must be rank-1 but is rank-2
  !ERROR: TODO: Analyze overload for ExplicitShapeBoundsSpec
  integer :: j(rank2_parameter_array)
  ! Test combined bounds error, first bound as before but second bound as wrong rank
  ! and nonconst
  integer :: rank3_array(2,2,2)
  !future_ERROR: Integer array used as lower bounds in DECLARATION must be rank-1 but is rank-2
  !future_ERROR: Integer array used as upper bounds in DECLARATION must be rank-1 but is rank-3
  !ERROR: TODO: Analyze overload for ExplicitShapeBoundsSpec
  integer :: k(rank2_parameter_array : rank3_array)

  ! Test that any comma list is parsed as ExplicitShapeSpecList and not rewritten 
  ! to ExplicitShapeBoundsSpec, giving error messages expecting same number of 
  ! aruments as rank of test_array and scalar integers
  !ERROR: Must be a scalar value, but is a rank-1 array
  !ERROR: Must be a scalar value, but is a rank-1 array
  !ERROR: Must be a scalar value, but is a rank-1 array
  !ERROR: Must have INTEGER type, but is REAL(4)
  integer :: test_array([1,2,3] : [2,3,4], 3, [1,2,3], 5.2)
end program
