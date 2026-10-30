! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Wno-portability
! Check the copy-out analysis of conditional arguments through the diagnostics
! that depend on it: F2023 C1547 and C1548, and the warning about a temporary
! copy for a VOLATILE dummy argument. A conditional argument is simply
! contiguous if and only if all of its consequent-args are, and polymorphic if
! any of them is (F2023 15.5.2.3 p4).

module m
  type :: t
  end type
 contains
  subroutine volatile_contiguous(x)
    real, volatile, contiguous :: x(:)
  end subroutine
  subroutine volatile_contiguous_optional(x)
    real, volatile, contiguous, optional :: x(:)
  end subroutine
  subroutine volatile_contiguous_t(x)
    type(t), volatile, contiguous :: x(:)
  end subroutine
  subroutine volatile_scalar(x)
    real, volatile :: x
  end subroutine

  subroutine test_all_discontiguous(flag)
    logical, intent(in) :: flag
    real, volatile :: a(10), b(10)
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call volatile_contiguous((flag ? a(::2) : b(::2)))
  end subroutine

  subroutine test_last_discontiguous(flag)
    logical, intent(in) :: flag
    real, volatile :: a(10), b(10)
    ! Only the discontiguous consequent-arg needs a temporary.
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call volatile_contiguous((flag ? a : b(::2)))
  end subroutine

  subroutine test_all_contiguous(flag1, flag2)
    logical, intent(in) :: flag1, flag2
    real, volatile :: a(10), b(10)
    call volatile_contiguous((flag1 ? a : b(2:9)))
    call volatile_contiguous_optional((flag1 ? a : flag2 ? b : .nil.))
  end subroutine

  subroutine test_expression_condition(n)
    integer, intent(in) :: n
    real, volatile :: a(10), b(10)
    ! A condition that isn't a variable doesn't make the argument an expression.
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call volatile_contiguous((n > 0 ? a(::2) : b(::2)))
  end subroutine

  subroutine test_polymorphic(flag, a, b)
    logical, intent(in) :: flag
    type(t), allocatable :: a(:)
    class(t), allocatable, volatile :: b(:)
    ! Only the second consequent-arg is polymorphic.
    !WARNING: The actual argument 'b' with VOLATILE attribute should not be associated with dummy argument 'x=' with VOLATILE attribute, because a temporary copy is required during the call [-Wvolatile-or-asynchronous-temporary]
    call volatile_contiguous_t((flag ? a : b))
  end subroutine

  subroutine test_coindexed(flag, a, b)
    logical, intent(in) :: flag
    real :: a
    real, volatile :: b[*]
    ! Only the second consequent-arg is coindexed.
    !ERROR: Coindexed ASYNCHRONOUS or VOLATILE actual argument may not be associated with dummy argument 'x=' with ASYNCHRONOUS or VOLATILE attributes unless VALUE
    call volatile_scalar((flag ? a : b[1]))
  end subroutine
end module
