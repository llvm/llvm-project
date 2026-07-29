! RUN: %python %S/test_errors.py %s %flang_fc1 -Wautomatic-in-main-program -Wsaved-local-in-spec-expr
! ---- Module with rank-1 array-bounded declarations, USE'd elsewhere ----
subroutine array_flatten(int)
  integer, intent(IN) :: int
  !Array Constructors produce rank-1 arrays, even with nested arrays,
  !so neither of these should produce an error or warning.
  integer :: fff([int, int])
  integer :: ff([[int, [int, int]]])
  integer :: arr([(int+i, integer(8) :: i=1_8, 2_8)])
end subroutine
module getter 
contains 
  pure function get_bounds() result(r)
    integer :: r(2)
    r = [8, 9]
  end function
  subroutine foo() 
    ! Function result (rank-1 integer array) as explicit shape bounds
    integer :: from_func(get_bounds())
  end subroutine 
end module
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
  integer :: arr_upper(dims)
  integer :: arr_both(lo : hi)
end module
subroutine sub_consumer()
  use bounds_provider, only: dims, lo, hi
  implicit none
  ! USE'd parameter arrays as bounds in a subroutine
  integer :: local_arr(dims)
  integer :: local_arr2(lo : hi)
end subroutine
subroutine sub_use_consumer()
  use consumer, only: arr_upper, arr_both
  implicit none
  ! USE the arrays that were themselves declared with rank-1 array bounds
  arr_upper = 1
  arr_both = 2
end subroutine
module m
contains
  impure integer function impf()
    impf = 1
  end function
end module
subroutine s1()
  use m
  !ERROR: Invalid specification expression: reference to impure function 'impf'
  integer :: z(impf() : [integer::])
end subroutine
subroutine bar(n, bounds, rank_bounds)
  integer, intent(IN) :: n 
  integer, intent(IN) :: bounds(:)
  integer, intent(IN) :: rank_bounds(..)
  integer :: bounds2(n)
  !ERROR: Rank-1 integer array used as upper bounds in DECLARATION must have constant size
  integer :: arr(bounds)
  !ERROR: Rank-1 integer array used as upper bounds in DECLARATION must have constant size
  integer :: arr2(bounds2)
  !ERROR: Rank-1 integer array used as upper bounds in DECLARATION must have constant size
  integer :: arr3(rank_bounds)
end subroutine

module data 
  integer :: rank1_array_module(3) = [5, 5, 5]
  !ERROR: Automatic data object 'gg2' may not appear in a module
  integer :: gg2(rank1_array_module)
  integer, allocatable :: nonconstsize(:)
  !ERROR: Rank-1 integer array used as lower bounds in DECLARATION must have constant size
  !ERROR: Rank-1 integer array used as upper bounds in DECLARATION must have constant size
  integer :: gg3(nonconstsize : nonconstsize)
end module 
program declaration_array_bounds
  use getter
  implicit none

  ! Valid cases (no errors expected)

  ! Array upper bound only
  integer :: c([3, 4, 5])
  integer, dimension([3, 4, 5]) :: cc

  ! Array lower and upper bounds, same size
  integer :: d((/2, 3/) : [10, 20])

  ! Scalar lower, array upper
  integer :: e(2 : [10, 20])

  ! Array lower, scalar upper
  integer :: f([2, 3] : 10)

  ! Using non-literal PARAMETER variables
  integer, parameter :: rank1_parameter_array(3) = [5,5,5]
  integer :: g(rank1_parameter_array)
  integer :: ggg(rank1_parameter_array * 2 : rank1_parameter_array - 1)


  ! Negative cases (errors expected)
  integer :: rank1_array(3) = [5,5,5]
  ! Use existing error message for constness checking
  !PORTABILITY: specification expression refers to local object 'rank1_array' (initialized and saved) [-Wsaved-local-in-spec-expr]
  !PORTABILITY: Automatic data object 'gg' should not appear in the specification part of a main program [-Wautomatic-in-main-program]
  integer :: gg(rank1_array)
  integer :: scalar
  !ERROR: Invalid specification expression: reference to local entity 'scalar'
  !PORTABILITY: Automatic data object 'gggg' should not appear in the specification part of a main program [-Wautomatic-in-main-program]
  integer :: gggg(rank1_parameter_array : scalar)

  !ERROR: Must have INTEGER type, but is REAL(4)
  integer :: h([1.2,2.2,3.2]:[1,2,3])
  !ERROR: DECLARATION bounds integer rank-1 arrays must have the same size; lower bounds has 3 elements, upper bounds has 2 elements
  integer :: i([1,2,3]:[3,3])
  !Previously uncaught bug: array of size 1 is being treated as a scalar, and broadcast. This is incorrect.
  !It should be treated as a size mismatch error like the one above.
  !ERROR: DECLARATION bounds integer rank-1 arrays must have the same size; lower bounds has 1 elements, upper bounds has 2 elements
  integer :: ii([1] : [1,2]) 
  !Test same behavior with vector subscripts
  !ERROR: DECLARATION bounds integer rank-1 arrays must have the same size; lower bounds has 1 elements, upper bounds has 2 elements
  integer :: abc(rank1_array([scalar]) : rank1_array([scalar, scalar]))
  !Test same behavior with array slices
  !ERROR: DECLARATION bounds integer rank-1 arrays must have the same size; lower bounds has 2 elements, upper bounds has 1 elements
  integer :: abcd(rank1_array(1:3:2) : rank1_array(1:1))
  ! using a nonconst upper bound or stride for array slices makes the size nonconst. Should error
  !ERROR: Rank-1 integer array used as upper bounds in DECLARATION must have constant size
  integer :: abcde(rank1_parameter_array(1:scalar:1))
  !ERROR: Rank-1 integer array used as upper bounds in DECLARATION must have constant size
  integer :: abcdef(rank1_parameter_array(1:1:scalar))

  ! Test error for rank > 1, fulfilling constness
  integer, parameter :: rank2_parameter_array(2,2) = reshape([[1,2],[3,4]], [2,2])
  !ERROR: Integer array used as upper bounds in DECLARATION must be rank-1 but is rank-2
  integer :: j(rank2_parameter_array)
  ! Test combined bounds error, first bound as before but second bound as wrong rank
  ! and nonconst
  integer :: rank3_array(2,2,2)
  !ERROR: Integer array used as lower bounds in DECLARATION must be rank-1 but is rank-2
  !ERROR: Integer array used as upper bounds in DECLARATION must be rank-1 but is rank-3
  integer :: k(rank2_parameter_array : rank3_array)

  ! Test that any comma list is parsed as ExplicitShapeSpecList and not rewritten 
  ! to ExplicitShapeBoundsSpec, giving error messages expecting scalar integer values.
  !ERROR: Must be a scalar value, but is a rank-1 array
  !ERROR: Must be a scalar value, but is a rank-1 array
  !ERROR: Must be a scalar value, but is a rank-1 array
  !ERROR: Must have INTEGER type, but is REAL(4)
  integer :: test_array([1,2,3] : [2,3,4], 3, [1,2,3], 5.2)

  ! Test that Integer type constraint wrapper correctly errors without aborting.
  !ERROR: Must have INTEGER type, but is CHARACTER(KIND=1,LEN=1_8)
  integer :: a1("a" : [1, 2])
  !ERROR: Must have INTEGER type, but is CHARACTER(KIND=1,LEN=2_8)
  integer :: a2(["ab", "cd"])
  character(2), parameter :: ccc(2) = ["ab", "cd"]
  !ERROR: Must have INTEGER type, but is CHARACTER(KIND=1,LEN=2_8)
  integer :: a3(ccc)
  !ERROR: Must have INTEGER type, but is typeless
  integer :: a4(z'4' : [1, 2])
  !ERROR: Must have INTEGER type, but is LOGICAL(4)
  integer :: a5([.true., .false.])
  !ERROR: Must have INTEGER type, but is REAL(4)
  integer :: a6([1.2, 2.2, 3.2] : [1, 2, 3])
  !ERROR: Must have INTEGER type, but is REAL(4)
  !ERROR: Must have INTEGER type, but is REAL(4)
  integer :: a7([1.2, 2.2, 3.2] : [1.2, 2.2, 3.2])
  !ERROR: No explicit type declared for 'undef'
  integer :: a8(undef : [1, 2])
  !ERROR: No explicit type declared for 'undef'
  integer :: a9([1, 2] : undef)

  integer :: dim0(0) = [integer::]
  integer :: zerosize1([integer::])
  !PORTABILITY: specification expression refers to local object 'dim0' (initialized and saved) [-Wsaved-local-in-spec-expr]
  integer :: zerosize2(dim0 : [integer::])
  !PORTABILITY: specification expression refers to local object 'dim0' (initialized and saved) [-Wsaved-local-in-spec-expr]
  integer :: zerosize3(999 : dim0)
  integer :: local9 = 9
  !PORTABILITY: specification expression refers to local object 'local9' (initialized and saved) [-Wsaved-local-in-spec-expr]
  integer :: zerosize4([integer::] : local9)
  !ERROR: DECLARATION bounds integer rank-1 arrays must have the same size; lower bounds has 0 elements, upper bounds has 2 elements
  integer :: zerosize_n([integer::] : [1,2])

  ! Test that maximum supported rank error fires. At the time of writing this test,
  ! that is 15. See flang/include/flang/Common/Fortran-consts.h.
  integer :: maxrank([(1,i=1,15)])
  !ERROR: DECLARATION rank-1 integer array bound(s) imply rank 16, which is greater than the maximum supported rank 15
  integer :: maxrank_n([(1,i=1,16)])
  !Test a rank size between the implementation's max rank and signed 32 bit integer overflow.
  !Previously, this was crashing the compiler, despite the size (maxRank + 1 == 16) test case above
  !erroring gracefully as expected. This is because pre-existing Rank() API on ArraySpec
  !(and several other classes) use C++ `int`s (despite calling inherited vector<>'s .size()
  !which is a size_t, so there is an implicit cast from unsigned int to a possibly smaller signed int).
  !Furthermore, this allows us to error out before pushing an absurd amount of ShapeSpec's to a 
  !vector. This was previously not a problem, but now is due to the nature of the change this feature introduces 
  !to a variable's rank. Previously, this was a context-free parse-time property given by 
  !(number of commas - 1), constrained by the literal source text. Now we can use any compile-time
  !constant value, and this can require a symbol table lookup (which makes rank a context-sensitive property, 
  !no longer strictly as parse-time).
  !Consider the following:
  !integer :: dims(100)
  !Before we'd have to declare 
  !integer :: array(dims(1),dims(2), ..., dims(100))
  !parsing every ',' along the way. This made overflow theoretically still possible but 
  !practically/computationally impossible. Now we can declare
  !integer :: array(dims)
  !and overflow much more easily.
  integer :: n(2147483648_8) !2^31
  !ERROR: DECLARATION rank-1 integer array bound(s) imply rank 2147483648, which is greater than the maximum supported rank 15
  integer :: too_big(n)

end program
