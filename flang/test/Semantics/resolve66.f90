! RUN: %S/test_errors.sh %s %t %f18
! Test that user-defined assignment is used in the right places

module m1
  type t1
  end type
  type t2
  end type
  interface assignment(=)
    subroutine assign_il(x, y)
      integer, intent(out) :: x
      logical, intent(in) :: y
    end
    subroutine assign_li(x, y)
      logical, intent(out) :: x
      integer, intent(in) :: y
    end
    subroutine assign_tt(x, y)
      import t1
      type(t1), intent(out) :: x
      type(t1), intent(in) :: y
    end
    subroutine assign_tz(x, y)
      import t1
      type(t1), intent(out) :: x
      complex, intent(in) :: y
    end
    subroutine assign_01(x, y)
      real, intent(out) :: x
      real, intent(in) :: y(:)
    end
  end interface
contains
  ! These are all intrinsic assignments
  pure subroutine test1()
    type(t2) :: a, b, b5(5)
    logical :: l
    integer :: i, i5(5)
    a = b
    b5 = a
    l = .true.
    i = z'1234'
    i5 = 1.0
  end

  ! These have invalid type combinations
  subroutine test2()
    type(t1) :: a
    type(t2) :: b
    logical :: l, l5(5)
    complex :: z, z5(5), z55(5,5)
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types TYPE(t1) and TYPE(t2)
    a = b
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types REAL(4) and LOGICAL(4)
    r = l
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types LOGICAL(4) and REAL(4)
    l = r
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types TYPE(t1) and REAL(4)
    a = r
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types TYPE(t2) and COMPLEX(4)
    b = z
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar COMPLEX(4) and rank 1 array of COMPLEX(4)
    z = z5
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches rank 1 array of LOGICAL(4) and scalar COMPLEX(4)
    l5 = z
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches rank 1 array of COMPLEX(4) and rank 2 array of COMPLEX(4)
    z5 = z55
  end

  ! These should all be defined assignments. Because the subroutines
  ! implementing them are not pure, they should all produce errors
  pure subroutine test3()
    type(t1) :: a, b
    integer :: i
    logical :: l
    complex :: z
    real :: r, r5(5)
    !ERROR: Procedure 'assign_tt' referenced in pure subprogram 'test3' must be pure too
    a = b
    !ERROR: Procedure 'assign_il' referenced in pure subprogram 'test3' must be pure too
    i = l
    !ERROR: Procedure 'assign_li' referenced in pure subprogram 'test3' must be pure too
    l = i
    !ERROR: Procedure 'assign_il' referenced in pure subprogram 'test3' must be pure too
    i = .true.
    !ERROR: Procedure 'assign_tz' referenced in pure subprogram 'test3' must be pure too
    a = z
    !ERROR: Procedure 'assign_01' referenced in pure subprogram 'test3' must be pure too
    r = r5
  end

  ! Like test3 but not in a pure subroutine so no errors.
  subroutine test4()
    type(t1) :: a, b
    integer :: i
    logical :: l
    complex :: z
    real :: r, r5(5)
    a = b
    i = l
    l = i
    i = .true.
    a = z
    r = r5
  end
end
