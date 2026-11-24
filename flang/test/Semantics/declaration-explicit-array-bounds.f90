! RUN: %python %S/test_errors.py %s %flang_fc1 -Wautomatic-in-main-program -Wsaved-local-in-spec-expr -Wrank-one-bounds-on-function-result
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
  impure function impureZeroSize() result(r)
    integer :: r(0)
  end function
end module
subroutine s1()
  use m
  !ERROR: Invalid specification expression: reference to impure function 'impf'
  integer :: z(impf() : [integer::])
  ! A bare zero-size upper bound declares a scalar (implicit lower bound 1),
  ! but the bound is still a specification expression: the deferred check must
  ! reject the impure function reference.
  !ERROR: Invalid specification expression: reference to impure function 'impurezerosize'
  integer :: z2(impureZeroSize())
  ! Both bounds of a zero-size explicit-shape-bounds-spec are still validated,
  ! even though the entity is scalar.
  !ERROR: Invalid specification expression: reference to impure function 'impf'
  !ERROR: Invalid specification expression: reference to impure function 'impurezerosize'
  integer :: z3(impf() : impureZeroSize())
  
  !Now test using the dimension attribute, and a mixture of both (overriding the dimension attribute).
  !We must validate dimension even if overridden. 
  !ERROR: Invalid specification expression: reference to impure function 'impf'
  integer, dimension(impf() : [integer::]) :: z4

  !The overridden dimension spec should still be checked and error
  !ERROR: Invalid specification expression: reference to impure function 'impf'
  integer, dimension(impf() : [integer::]) :: z5(5)

  !The overridden dimension spec should still be checked as above with z2, but x
  !should also error. So, it should be emitted once for z6 and once for z7.
  !ERROR: Invalid specification expression: reference to impure function 'impf'
  !ERROR: Invalid specification expression: reference to impure function 'impf'
  integer, dimension(impf() : [integer::]) :: z6(5), z7

  !ERROR: Invalid specification expression: reference to impure function 'impf'
  integer, dimension([impf(),2,3]) :: z9
  !ERROR: Invalid specification expression: reference to impure function 'impf'
  integer, dimension([1,2,3] : [4,5,impf()]) :: z10

  ! An entity-decl's own array-spec overrides the DIMENSION attribute, but the
  ! attribute's bounds remain specification expressions that must be validated.
  !ERROR: Invalid specification expression: reference to impure function 'impf'
  integer, dimension([impf(),2,3]) :: z11([1,2,3])

  ! A zero-size (scalar) entity-decl array-spec also overrides the DIMENSION
  ! attribute, but the attribute's bounds must still be validated.
  !ERROR: Invalid specification expression: reference to impure function 'impf'
  integer, dimension(impf() : 5) :: z12(1 : [integer::])

  ! A duplicate DIMENSION attribute is an error even when one of them is a
  ! zero-size bounds array (F2023): such an attribute declares a scalar and so
  ! leaves the statement-level array-spec empty, but it is still a DIMENSION.
  !ERROR: Attribute 'DIMENSION' cannot be used more than once
  integer, dimension(1 : [integer::]), dimension(3) :: z13
  !ERROR: Attribute 'DIMENSION' cannot be used more than once
  integer, dimension(3), dimension([integer::]) :: z14([integer::])

end subroutine
! The bounds of a zero-size (scalar) declaration are validated during
! declaration checking, when the scope is complete -- not during name
! resolution.  The later DATA statement makes 'k' saved and initialized;
! because the check happens once the scope is final, the zero-size case (zero)
! reports the same portability warning as the ordinary broadcast case
! (nonzero).  Checking 'k' early, before DATA had been processed, would have
! diagnosed these two declarations inconsistently.
subroutine timing()
  integer :: k
  !PORTABILITY: specification expression refers to local object 'k' (initialized and saved) [-Wsaved-local-in-spec-expr]
  integer :: zero(k : [integer::])
  !PORTABILITY: specification expression refers to local object 'k' (initialized and saved) [-Wsaved-local-in-spec-expr]
  integer :: nonzero(k : [1, 2])
  data k / 5 /
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
! A raw rank-1 bound element that survives folding in a's extents (merge() with
! a LOGICAL mask is not reduced to a per-element scalar, and the lower bound 0
! makes neither UBOUND nor LBOUND exact) must keep SIZE from folding, but must
! not keep it from validating DIM=: an out-of-range or non-positive DIM is
! still diagnosed.
subroutine size_dim_checked(n, m, c)
  integer, intent(in) :: n(2), m(2)
  logical, intent(in) :: c(2)
  real :: a(0:merge(n, m, c))
  !ERROR: DIM=9 dimension is out of range for rank-2 array
  print *, size(a, 9)
  !ERROR: DIM=2147483648 dimension is out of range for rank-2 array
  print *, size(a, 2147483648_8)
  !ERROR: DIM=0 dimension must be positive
  print *, size(a, 0)
end subroutine
! A constant bounds array is folded into per-dimension constant bounds at
! declaration, so no raw element remains and size(a) is a constant expression:
! 3*4 == 12 (a kind of -1 would be rejected).
subroutine size_constant_base()
  real :: a(0:[2, 3])
  integer, parameter :: count = size(a)
  integer(kind=merge(4, -1, count == 12)) :: ok
  ok = 0
end subroutine
! A constant rank-1 lower bound is folded into per-dimension constants at
! declaration, so a lower bound of 1 is recognized regardless of the (unknown)
! extent and LBOUND is a constant expression exactly as for a(1:n).  (A lower
! bound other than 1 is correctly not constant here: LBOUND is 1 for an empty
! dimension, and n(2) is not known.)
subroutine lbound_constant_base(n)
  integer, intent(in) :: n(2)
  integer, parameter :: ones(2) = 1
  real :: a(ones:n), b([1, 1]:n)
  integer, parameter :: la = lbound(a, 1), lb2 = lbound(b, 2)
  integer(kind=merge(4, -1, la == 1 .and. lb2 == 1)) :: ok
  ok = 0
end subroutine
! F2023 C816 permits only an explicit-shape-spec-list for a nonallocatable
! nonpointer function result, so a rank-1 integer array as the result's bounds
! is an extension and gets a portability warning.  The same bounds on a dummy
! (R814) or on an ALLOCATABLE result (not covered by C816) are standard.
module c816
contains
  function res_rank1(n, a) result(r)
    integer, intent(in) :: n(2)
    real, intent(in) :: a(n)
    !PORTABILITY: Function result 'r' should not be declared with a rank-1 integer array as its bounds (F2023 C816) [-Wrank-one-bounds-on-function-result]
    real :: r(n)
    r = a
  end function
  function res_rank1_lb(lo, hi) result(r)
    integer, intent(in) :: lo(2), hi(2)
    !PORTABILITY: Function result 'r' should not be declared with a rank-1 integer array as its bounds (F2023 C816) [-Wrank-one-bounds-on-function-result]
    real :: r(lo:hi)
    r = 0.0
  end function
  function res_zero_size() result(r)
    !PORTABILITY: Function result 'r' should not be declared with a rank-1 integer array as its bounds (F2023 C816) [-Wrank-one-bounds-on-function-result]
    real :: r([integer::])
    r = 0.0
  end function
  ! A constant bounds array is folded into ordinary bounds at declaration,
  ! but the shape still records that it was written as a rank-1 array.
  function res_constant_base() result(r)
    !PORTABILITY: Function result 'r' should not be declared with a rank-1 integer array as its bounds (F2023 C816) [-Wrank-one-bounds-on-function-result]
    real :: r([2, 3])
    r = 0.0
  end function
  ! The rank-1 bounds may also come from a DIMENSION attribute or statement.
  function res_dimension_attr() result(r)
    !PORTABILITY: Function result 'r' should not be declared with a rank-1 integer array as its bounds (F2023 C816) [-Wrank-one-bounds-on-function-result]
    real, dimension([2, 3]) :: r
    r = 0.0
  end function
  function res_dimension_attr_dummy(n) result(r)
    integer, intent(in) :: n(2)
    !PORTABILITY: Function result 'r' should not be declared with a rank-1 integer array as its bounds (F2023 C816) [-Wrank-one-bounds-on-function-result]
    real, dimension(n) :: r
    r = 0.0
  end function
  function res_dimension_stmt() result(r)
    !PORTABILITY: Function result 'r' should not be declared with a rank-1 integer array as its bounds (F2023 C816) [-Wrank-one-bounds-on-function-result]
    real :: r
    dimension r([2, 3])
    r = 0.0
  end function
  function res_alloc(n) result(r)
    integer, intent(in) :: n(2)
    real, allocatable :: r(:,:)
    allocate(r(n(1), n(2)))
    r = 0.0
  end function
  function res_explicit_list(n) result(r)
    integer, intent(in) :: n(2)
    real :: r(n(1), n(2))
    r = 0.0
  end function
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

  ! A rank-1 bound base gives the array its rank; an intrinsic inquiry on that
  ! array (ubound/lbound/size) -- the form such a bound takes when serialized to
  ! a module file -- is validated by the ordinary DIM-range check.  A DIM beyond
  ! the rank, including one that overflows signed 32-bit (2^31), is rejected
  ! without wrapping, so a consumer cannot be handed an out-of-range dimension.
  integer, parameter :: twodims(2) = [3, 4]
  integer :: a_dim(twodims) ! rank-2
  !ERROR: DIM=9 dimension is out of range for rank-2 array
  integer :: bad_dim(ubound(a_dim, 9))
  !ERROR: DIM=2147483648 dimension is out of range for rank-2 array
  integer :: bad_dim_ovf(ubound(a_dim, 2147483648_8))

end program
