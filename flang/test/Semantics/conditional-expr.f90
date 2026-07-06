! RUN: %python %S/test_errors.py %s %flang_fc1
! Test semantic analysis of conditional expressions (Fortran 2023)

! Valid cases with basic types
subroutine valid_basic_types(flag)
  logical :: flag
  integer :: i1, i2, i3
  real :: r1, r2, r3
  complex :: c1, c2, c3
  logical :: l1, l2, l3
  character(len=5) :: ch1, ch2, ch3

  ! INTEGER conditionals
  i3 = (flag ? i1 : i2)

  ! REAL conditionals
  r3 = (flag ? r1 : r2)

  ! COMPLEX conditionals
  c3 = (flag ? c1 : c2)

  ! LOGICAL conditionals
  l3 = (flag ? l1 : l2)

  ! CHARACTER conditionals
  ch3 = (flag ? ch1 : ch2)
end subroutine

! Valid cases with same kind
subroutine valid_same_kind(flag)
  logical :: flag
  integer(kind=4) :: i4a, i4b, i4c
  integer(kind=8) :: i8a, i8b, i8c
  real(kind=4) :: r4a, r4b, r4c
  real(kind=8) :: r8a, r8b, r8c

  ! Same kind - valid
  i4c = (flag ? i4a : i4b)
  i8c = (flag ? i8a : i8b)
  r4c = (flag ? r4a : r4b)
  r8c = (flag ? r8a : r8b)
end subroutine

! Valid cases with literals
subroutine valid_literals(flag)
  logical :: flag
  integer :: i
  real :: r
  character(len=10) :: ch

  i = (flag ? 10 : 20)
  r = (flag ? 1.0 : 2.0)
  ch = (flag ? "HELLO" : "WORLD")
end subroutine

! Valid cases with nested conditionals
subroutine valid_nested(flag1, flag2, x, y, z, w)
  logical :: flag1, flag2
  integer :: x, y, z, w, result

  ! Nested in value position
  result = (flag1 ? (flag2 ? x : y) : z)

  ! Nested in condition (condition is logical)
  result = ((x > y ? flag1 : flag2) ? w : z)

  ! Multi-branch
  result = (x > 10 ? 100 : x > 5 ? 50 : 0)
end subroutine

! Valid cases with arrays
subroutine valid_arrays(flag)
  logical :: flag
  integer :: arr1(10), arr2(10), arr3(10)
  real :: mat1(3,3), mat2(3,3), mat3(3,3)

  ! Whole array conditional
  arr3 = (flag ? arr1 : arr2)

  ! Multidimensional arrays
  mat3 = (flag ? mat1 : mat2)

  ! Array sections
  arr3(1:5) = (flag ? arr1(1:5) : arr2(1:5))
end subroutine

! Valid cases with derived types
subroutine valid_derived_types(flag)
  type :: point
    real :: x, y
  end type

  logical :: flag
  type(point) :: p1, p2, p3

  p3 = (flag ? p1 : p2)
end subroutine

! Valid cases with character lengths
subroutine valid_character_lengths(flag)
  logical :: flag
  character(len=5) :: short1, short2, short3
  character(len=10) :: medium
  character(len=20) :: long

  ! Same length
  short3 = (flag ? short1 : short2)

  ! Different lengths - padding/truncation applies
  medium = (flag ? short1 : medium)
  long = (flag ? short1 : "A LONGER STRING")
end subroutine

! Valid: deferred-length character scalars
subroutine valid_deferred_length_character(flag)
  logical :: flag
  character(len=:), allocatable :: str1, str2, result

  str1 = "SHORT"
  str2 = "A MUCH LONGER STRING"
  ! Result length is determined by selected branch
  result = (flag ? str1 : str2)
end subroutine

! Valid: assumed-length character arguments
subroutine valid_assumed_length_character(flag, str1, str2)
  logical :: flag
  character(len=*) :: str1, str2
  character(len=100) :: result

  result = (flag ? str1 : str2)
end subroutine

! Error: condition must be logical
subroutine error_non_logical_condition()
  integer :: i, x, y
  real :: r
  character :: ch

  !ERROR: Must have LOGICAL type, but is INTEGER(4)
  i = (i ? x : y)

  !ERROR: Must have LOGICAL type, but is REAL(4)
  i = (r ? x : y)

  !ERROR: Must have LOGICAL type, but is CHARACTER(KIND=1,LEN=1_8)
  i = (ch ? x : y)
end subroutine

! Error: type mismatch between branches
subroutine error_type_mismatch(flag)
  logical :: flag
  integer :: i1, i2
  real :: r
  character :: ch
  complex :: c

  !ERROR: All values in conditional expression must have the same type and kind; have INTEGER(4) and REAL(4)
  i1 = (flag ? i2 : r)

  !ERROR: All values in conditional expression must have the same type and kind; have INTEGER(4) and CHARACTER(KIND=1,LEN=1_8)
  i1 = (flag ? i2 : ch)

  !ERROR: All values in conditional expression must have the same type and kind; have REAL(4) and COMPLEX(4)
  r = (flag ? r : c)

  !ERROR: All values in conditional expression must have the same type and kind; have LOGICAL(4) and INTEGER(4)
  flag = (flag ? flag : i1)
end subroutine

! Error: kind mismatch (F2023 C1004)
subroutine error_kind_mismatch(flag)
  logical :: flag
  integer(kind=4) :: i4
  integer(kind=8) :: i8
  real(kind=4) :: r4
  real(kind=8) :: r8
  complex(kind=4) :: c4
  complex(kind=8) :: c8

  !ERROR: All values in conditional expression must have the same type and kind; have INTEGER(4) and INTEGER(8)
  i4 = (flag ? i4 : i8)

  !ERROR: All values in conditional expression must have the same type and kind; have REAL(4) and REAL(8)
  r4 = (flag ? r4 : r8)

  !ERROR: All values in conditional expression must have the same type and kind; have COMPLEX(4) and COMPLEX(8)
  c4 = (flag ? c4 : c8)
end subroutine

! Error: derived type mismatch
subroutine error_derived_type_mismatch(flag)
  type :: type1
    integer :: i
  end type

  type :: type2
    integer :: i
  end type

  logical :: flag
  type(type1) :: t1
  type(type2) :: t2

  !ERROR: All values in conditional expression must be the same derived type; have type1 and type2
  t1 = (flag ? t1 : t2)
end subroutine

! Error: derived type vs intrinsic type mismatch
subroutine error_derived_vs_intrinsic(flag)
  type :: my_type
    integer :: i
  end type

  logical :: flag
  type(my_type) :: t
  integer :: i
  real :: r

  !ERROR: All values in conditional expression must have the same type and kind; have my_type and INTEGER(4)
  t = (flag ? t : i)

  !ERROR: All values in conditional expression must have the same type and kind; have INTEGER(4) and my_type
  t = (flag ? i : t)

  !ERROR: All values in conditional expression must have the same type and kind; have my_type and REAL(4)
  t = (flag ? t : r)
end subroutine

! Error: array rank mismatch
subroutine error_array_rank_mismatch(flag)
  logical :: flag
  integer :: arr1(10), mat1(3,3), result(10)

  !ERROR: All values in conditional expression must have the same rank; have rank 1 and 2
  result = (flag ? arr1 : mat1)
end subroutine

! Error: scalar vs array mismatch
subroutine error_scalar_array_mismatch(flag)
  logical :: flag
  integer :: scalar, arr(10), result(10)

  !ERROR: All values in conditional expression must have the same rank; have rank 0 and 1
  result = (flag ? scalar : arr)
end subroutine

! Error: condition must be scalar
subroutine error_array_condition()
  logical :: flags(5)
  integer :: x(5), y(5), result(5)

  !ERROR: Must be a scalar value, but is a rank-1 array
  result = (flags ? x : y)
end subroutine

! Valid cases with intrinsic functions
subroutine valid_intrinsic_functions(x, y, flag)
  integer :: x, y
  logical :: flag
  integer :: result

  result = (flag ? abs(x) : abs(y))
  result = (flag ? max(x, y) : min(x, y))
end subroutine

! Valid: conditional in array constructor
subroutine valid_in_array_constructor(flag, x, y)
  logical :: flag
  integer :: x, y, arr(3)

  arr = [(flag ? x : y), (flag ? x + 1 : y + 1), (flag ? x + 2 : y + 2)]
end subroutine

! Valid: conditional in expression context
subroutine valid_in_expression(flag, x, y)
  logical :: flag
  integer :: x, y, z

  z = (flag ? x : y) + 10
  z = 2 * (flag ? x : y)

  if ((flag ? x : y) > 5) then
    z = 1
  end if
end subroutine

! Note: allocatable/pointer differences are handled by assignment semantics
! The conditional expression just requires matching types

! Valid: both branches allocatable
subroutine valid_both_allocatable(flag)
  logical :: flag
  integer, allocatable :: alloc1, alloc2, result

  allocate(result)
  result = (flag ? alloc1 : alloc2)
end subroutine

! Valid: both branches pointer
subroutine valid_both_pointer(flag)
  logical :: flag
  integer, pointer :: ptr1, ptr2, result

  result = (flag ? ptr1 : ptr2)
end subroutine

! Valid: elemental context
elemental integer function conditional_elemental(flag, x, y)
  logical, intent(in) :: flag
  integer, intent(in) :: x, y

  conditional_elemental = (flag ? x : y)
end function

! Valid: pure context
pure integer function conditional_pure(flag, x, y)
  logical, intent(in) :: flag
  integer, intent(in) :: x, y

  conditional_pure = (flag ? x : y)
end function

! Valid: recursive context
recursive integer function conditional_recursive(n, flag, x, y) result(res)
  integer, intent(in) :: n
  logical, intent(in) :: flag
  integer, intent(in) :: x, y

  if (n <= 0) then
    res = (flag ? x : y)
  else
    res = conditional_recursive(n - 1, flag, x, y)
  end if
end function

! Valid: nested multi-branch
subroutine valid_multi_branch(x)
  integer :: x, result

  ! Five-branch conditional
  result = (x > 20 ? 1 : x > 15 ? 2 : x > 10 ? 3 : x > 5 ? 4 : 5)
end subroutine

! Valid: polymorphic types
subroutine valid_polymorphic(flag)
  type :: base_t
    integer :: i
  end type

  logical :: flag
  class(base_t), allocatable :: poly1, poly2, result

  result = (flag ? poly1 : poly2)
end subroutine

! Error: unlimited polymorphic (CLASS(*)) not allowed
subroutine error_unlimited_polymorphic(flag)
  logical :: flag
  class(*), allocatable :: star1, star2, result

  !ERROR: Unlimited polymorphic types (CLASS(*)) not allowed in conditional expression
  result = (flag ? star1 : star2)
end subroutine

! Error: mismatched character kinds
subroutine error_character_kind_mismatch(flag)
  logical :: flag
  character(kind=1, len=5) :: ch1
  character(kind=4, len=5) :: ch4

  !ERROR: All values in conditional expression must have the same type and kind; have CHARACTER(KIND=1,LEN=5_8) and CHARACTER(KIND=4,LEN=5_8)
  ch1 = (flag ? ch1 : ch4)
end subroutine

! Valid: optional arguments
subroutine valid_optional_args(flag, opt_x, opt_y)
  logical :: flag
  integer, optional :: opt_x, opt_y
  integer :: result

  if (present(opt_x) .and. present(opt_y)) then
    result = (flag ? opt_x : opt_y)
  end if
end subroutine

! Valid: mix of expressions and designators
subroutine valid_mixed_expressions(flag, x, y)
  logical :: flag
  integer :: x, y, result

  result = (flag ? x + y : x - y)
  result = (flag ? 2 * x : y / 2)
end subroutine

! Constant-folding: when the condition is a constant, only the selected
! branch must be a constant expression (F2023 10.1.12).
subroutine constant_folding_cases()
  integer :: non_const = 99

  ! Valid: .true. selects 10; non_const is in the unselected else-branch.
  integer, parameter :: p_true_const  = (.true.  ? 10 : non_const)

  ! Valid: .false. selects 10; non_const is in the unselected then-branch.
  integer, parameter :: p_false_const = (.false. ? non_const : 10)

  ! Error: .false. selects non_const — not a constant expression.
  !ERROR: Must be a constant value
  integer, parameter :: p_false_nconst = (.false. ? 10 : non_const)

  ! Error: .true. selects non_const — not a constant expression.
  !ERROR: Must be a constant value
  integer, parameter :: p_true_nconst  = (.true.  ? non_const : 10)
end subroutine

! Module serialization: conditional expressions in a module must be correctly
! written to and read back from the .mod file.
module conditional_expr_mod
  implicit none
contains
  subroutine mod_mixed_expressions(flag, x, y, result)
    logical, intent(in) :: flag
    integer, intent(in) :: x, y
    integer, intent(out) :: result

    result = (flag ? x + y : x - y)
    result = (flag ? 2 * x : y / 2)
  end subroutine
end module

subroutine valid_use_from_module(flag, x, y)
  use conditional_expr_mod
  logical :: flag
  integer :: x, y, result

  call mod_mixed_expressions(flag, x, y, result)
end subroutine
