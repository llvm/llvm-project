! RUN: %python %S/test_errors.py %s %flang_fc1
! Inquiries of a conditional argument (F2023 R1526-R1528) whose result can
! differ between consequent-args must not fold through a representative
! consequent-arg when the condition isn't constant. Compare with the type-only
! inquiries in Evaluate/fold-conditional-arg.f90, which do fold.
module m
  use ieee_arithmetic, only: ieee_support_flag, ieee_invalid, ieee_overflow, &
      ieee_support_rounding, ieee_round_type
  implicit none
  type :: base
    integer :: i
  end type
  logical :: flag
  integer :: a1(3), a2(5), a3(2:4)
  integer :: m2(2, 3)
  integer, allocatable :: alloc1(:), alloc2(:)
  integer, pointer :: ptr1(:), ptr2(:)
  integer, target :: tgt(10)
  real :: x, y
  type(ieee_round_type) :: mode
  !ERROR: Must be a constant value
  integer, parameter :: p_size = size((flag ? a1 : a2))
  !ERROR: Must be a constant value
  integer, parameter :: p_size_dim = size((flag ? m2 : m2), dim=1)
  !ERROR: Must be a constant value
  integer, parameter :: p_shape(1) = shape((flag ? a1 : a2))
  !ERROR: Must be a constant value
  integer, parameter :: p_lbound = lbound((flag ? a1 : a3), 1)
  !ERROR: Must be a constant value
  integer, parameter :: p_ubound = ubound((flag ? a1 : a3), 1)
  !ERROR: Must be a constant value
  logical, parameter :: p_allocated = allocated((flag ? alloc1 : alloc2))
  !ERROR: Must be a constant value
  logical, parameter :: p_associated = associated((flag ? ptr1 : ptr2))
  !ERROR: Must be a constant value
  logical, parameter :: p_contiguous = is_contiguous((flag ? tgt(1:10:2) : tgt))
  ! FLAG= of IEEE_SUPPORT_FLAG isn't a type-only inquiry dummy, so the
  ! conditional argument must not fold through a representative there even
  ! though X= would qualify.
  !ERROR: Must be a constant value
  logical, parameter :: p_flag_pos = ieee_support_flag((flag ? ieee_invalid : ieee_overflow), x)
  ! One non-qualifying conditional argument stops the fold for the whole
  ! reference, even when the other one qualifies.
  !ERROR: Must be a constant value
  logical, parameter :: p_flag_both = ieee_support_flag((flag ? ieee_invalid : ieee_overflow), (flag ? x : y))
  ! X= qualifies, but the copy folded with the representative isn't a constant
  ! when ROUND_VALUE= is a variable, so the original reference is kept.
  !ERROR: Must be a constant value
  logical, parameter :: p_round = ieee_support_rounding(mode, (flag ? x : y))
contains
  ! LEN and STORAGE_SIZE depend on length type parameters and dynamic type,
  ! which C1538 doesn't require to agree between consequent-args.
  subroutine s(ca, cb, pa, pb)
    character(*) :: ca, cb
    class(base) :: pa, pb
    !ERROR: Must be a constant value
    integer, parameter :: p_len = len((flag ? ca : cb))
    !ERROR: Must be a constant value
    integer, parameter :: p_storage_size = storage_size((flag ? pa : pb))
  end subroutine
  ! C1539 permits consequent-args that are all assumed-rank, and their ranks
  ! can differ at run time.
  subroutine r(ar1, ar2)
    integer :: ar1(..), ar2(..)
    !ERROR: Value of named constant 'p_rank' (rank(( flag ? ar1 : ar2 ))) cannot be computed as a constant value
    integer, parameter :: p_rank = rank((flag ? ar1 : ar2))
  end subroutine
end module
