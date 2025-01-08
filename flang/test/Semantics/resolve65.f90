! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Test restrictions on what subprograms can be used for defined assignment.

module m1
  implicit none
  type :: t
  contains
    !ERROR: Generic 'assignment(=)' may not have specific procedures 't%assign_t4' and 't%assign_t5' as their interfaces are not distinguishable
    !ERROR: Generic 'assignment(=)' may not have specific procedures 't%assign_t4' and 't%assign_t6' as their interfaces are not distinguishable
    !ERROR: Generic 'assignment(=)' may not have specific procedures 't%assign_t5' and 't%assign_t6' as their interfaces are not distinguishable
    !ERROR: Defined assignment procedure 'binding' must be a subroutine
    generic :: assignment(=) => binding
    procedure :: binding => assign_t1
    procedure :: assign_t
    procedure :: assign_t2
    procedure :: assign_t3
    !ERROR: Defined assignment subroutine 'assign_t2' must have two dummy arguments
    !WARNING: In defined assignment subroutine 'assign_t3', second dummy argument 'y' should have INTENT(IN) or VALUE attribute
    !WARNING: In defined assignment subroutine 'assign_t4', first dummy argument 'x' should have INTENT(OUT) or INTENT(INOUT)
    !ERROR: In defined assignment subroutine 'assign_t5', first dummy argument 'x' may not have INTENT(IN)
    !ERROR: In defined assignment subroutine 'assign_t6', second dummy argument 'y' may not have INTENT(OUT)
    generic :: assignment(=) => assign_t, assign_t2, assign_t3, assign_t4, assign_t5, assign_t6
    procedure :: assign_t4
    procedure :: assign_t5
    procedure :: assign_t6
  end type
  type :: t2
  contains
    procedure, nopass :: assign_t
    !ERROR: Defined assignment procedure 'assign_t' may not have NOPASS attribute
    generic :: assignment(=) => assign_t
  end type
contains
  subroutine assign_t(x, y)
    class(t), intent(out) :: x
    type(t), intent(in) :: y
  end
  logical function assign_t1(x, y)
    class(t), intent(out) :: x
    type(t), intent(in) :: y
  end
  subroutine assign_t2(x)
    class(t), intent(out) :: x
  end
  subroutine assign_t3(x, y)
    class(t), intent(out) :: x
    real :: y
  end
  subroutine assign_t4(x, y)
    class(t) :: x
    integer, intent(in) :: y
  end
  subroutine assign_t5(x, y)
    class(t), intent(in) :: x
    integer, intent(in) :: y
  end
  subroutine assign_t6(x, y)
    class(t), intent(out) :: x
    integer, intent(out) :: y
  end
end

module m2
  type :: t
  end type
  !ERROR: Generic 'assignment(=)' may not have specific procedures 's3' and 's4' as their interfaces are not distinguishable
  interface assignment(=)
    !ERROR: In defined assignment subroutine 's1', dummy argument 'y' may not be OPTIONAL
    subroutine s1(x, y)
      import t
      type(t), intent(out) :: x
      real, optional, intent(in) :: y
    end
    !ERROR: In defined assignment subroutine 's2', dummy argument 'y' must be a data object
    subroutine s2(x, y)
      import t
      type(t), intent(out) :: x
      intent(in) :: y
      interface
        subroutine y()
        end
      end interface
    end
    !ERROR: In defined assignment subroutine 's3', second dummy argument 'y' must not be a pointer
    subroutine s3(x, y)
      import t
      type(t), intent(out) :: x
      type(t), intent(in), pointer :: y
    end
    !ERROR: In defined assignment subroutine 's4', second dummy argument 'y' must not be an allocatable
    subroutine s4(x, y)
      import t
      type(t), intent(out) :: x
      type(t), intent(in), allocatable :: y
    end
  end interface
end

! Detect defined assignment that conflicts with intrinsic assignment
module m5
  type :: t
  end type
  interface assignment(=)
    ! OK - lhs is derived type
    subroutine assign_tt(x, y)
      import t
      type(t), intent(out) :: x
      type(t), intent(in) :: y
    end
    !OK - incompatible types
    subroutine assign_il(x, y)
      integer, intent(out) :: x
      logical, intent(in) :: y
    end
    !OK - different ranks
    subroutine assign_23(x, y)
      integer, intent(out) :: x(:,:)
      integer, intent(in) :: y(:,:,:)
    end
    !OK - scalar = array
    subroutine assign_01(x, y)
      integer, intent(out) :: x
      integer, intent(in) :: y(:)
    end
    !ERROR: Defined assignment subroutine 'assign_10' conflicts with intrinsic assignment
    subroutine assign_10(x, y)
      integer, intent(out) :: x(:)
      integer, intent(in) :: y
    end
    !ERROR: Defined assignment subroutine 'assign_ir' conflicts with intrinsic assignment
    subroutine assign_ir(x, y)
      integer, intent(out) :: x
      real, intent(in) :: y
    end
    !ERROR: Defined assignment subroutine 'assign_ii' conflicts with intrinsic assignment
    subroutine assign_ii(x, y)
      integer(2), intent(out) :: x
      integer(1), intent(in) :: y
    end
  end interface
end
