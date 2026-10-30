! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Wno-portability
! A conditional argument is simply contiguous if and only if all of its
! consequent-args are (F2023 15.5.2.3 p4). Check that F2023 C1548 is enforced
! for each consequent-arg that is not, and that no temporary copy is reported
! when all of them are.

module m
 contains
  subroutine volatile_contiguous(x)
    real, volatile, contiguous :: x(:)
  end subroutine
  subroutine volatile_contiguous_optional(x)
    real, volatile, contiguous, optional :: x(:)
  end subroutine

  subroutine test_all_discontiguous(flag)
    logical, intent(in) :: flag
    real, volatile :: a(10), b(10)
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call volatile_contiguous((flag ? a(::2) : b(::2)))
  end subroutine

  subroutine test_last_discontiguous(flag)
    logical, intent(in) :: flag
    real :: a(10)
    real, volatile :: b(10)
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call volatile_contiguous((flag ? a : b(::2)))
  end subroutine

  subroutine test_all_contiguous(flag1, flag2)
    logical, intent(in) :: flag1, flag2
    real, volatile :: a(10), b(10)
    call volatile_contiguous((flag1 ? a : b(2:9)))
    call volatile_contiguous_optional((flag1 ? a : flag2 ? b : .nil.))
  end subroutine
end module
