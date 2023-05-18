! RUN: %python %S/test_errors.py %s %flang_fc1
! Tests for C760:
! The passed-object dummy argument shall be a scalar, nonpointer, nonallocatable
! dummy data object with the same declared type as the type being defined;
! all of its length type parameters shall be assumed; it shall be polymorphic
! (7.3.2.3) if and only if the type being defined is extensible (7.5.7).
! It shall not have the VALUE attribute.
!
! C757 If the procedure pointer component has an implicit interface or has no
! arguments, NOPASS shall be specified.
!
! C758 If PASS (arg-name) appears, the interface of the procedure pointer
! component shall have a dummy argument named arg-name.


module m1
  type :: t
    procedure(real), pointer, nopass :: a
    !ERROR: Procedure component 'b' must have NOPASS attribute or explicit interface
    procedure(real), pointer :: b
  end type
end

module m2
  type :: t
    !ERROR: Procedure component 'a' with no dummy arguments must have NOPASS attribute
    procedure(s1), pointer :: a
    !ERROR: Procedure component 'b' with no dummy arguments must have NOPASS attribute
    procedure(s1), pointer, pass :: b
  contains
    !ERROR: Procedure binding 'p1' with no dummy arguments must have NOPASS attribute
    procedure :: p1 => s1
    !ERROR: Procedure binding 'p2' with no dummy arguments must have NOPASS attribute
    procedure, pass :: p2 => s1
  end type
contains
  subroutine s1()
  end
end

module m3
  type :: t
    !ERROR: 'y' is not a dummy argument of procedure interface 's'
    procedure(s), pointer, pass(y) :: a
  contains
    !ERROR: 'z' is not a dummy argument of procedure interface 's'
    procedure, pass(z) :: p => s
  end type
contains
  subroutine s(x)
    class(t) :: x
  end
  subroutine test
    type(t) x
    !ERROR: Dummy argument 'x=' (#1) is not OPTIONAL and is not associated with an actual argument in this procedure reference
    call x%p
  end
end

module m4
  type :: t
    !ERROR: Passed-object dummy argument 'x' of procedure 'a' may not have the POINTER attribute
    procedure(s1), pointer :: a
    !ERROR: Passed-object dummy argument 'x' of procedure 'b' may not have the ALLOCATABLE attribute
    procedure(s2), pointer, pass(x) :: b
    !ERROR: Passed-object dummy argument 'f' of procedure 'c' must be a data object
    procedure(s3), pointer, pass :: c
    !ERROR: Passed-object dummy argument 'x' of procedure 'd' must be scalar
    procedure(s4), pointer, pass :: d
  end type
contains
  subroutine s1(x)
    class(t), pointer :: x
  end
  subroutine s2(w, x)
    real :: x
    !ERROR: The type of 'x' has already been declared
    class(t), allocatable :: x
  end
  subroutine s3(f)
    interface
      real function f()
      end function
    end interface
  end
  subroutine s4(x)
    class(t) :: x(10)
  end
end

module m5
  type :: t1
    sequence
    !ERROR: Passed-object dummy argument 'x' of procedure 'a' must be of type 't1' but is 'REAL(4)'
    procedure(s), pointer :: a
  end type
  type :: t2
  contains
    !ERROR: Passed-object dummy argument 'y' of procedure 's' must be of type 't2' but is 'TYPE(t1)'
    procedure, pass(y) :: s
  end type
contains
  subroutine s(x, y)
    real :: x
    type(t1) :: y
  end
end

module m6
  type :: t(k, l)
    integer, kind :: k
    integer, len :: l
    !ERROR: Passed-object dummy argument 'x' of procedure 'a' has non-assumed length parameter 'l'
    procedure(s1), pointer :: a
  end type
contains
  subroutine s1(x)
    class(t(1, 2)) :: x
  end
end

module m7
  type :: t
    sequence  ! t is not extensible
    !ERROR: Passed-object dummy argument 'x' of procedure 'a' may not be polymorphic because 't' is not extensible
    procedure(s), pointer :: a
  end type
contains
  subroutine s(x)
    !ERROR: Non-extensible derived type 't' may not be used with CLASS keyword
    class(t) :: x
  end
end

module m8
  type :: t
  contains
    !ERROR: Passed-object dummy argument 'x' of procedure 's' must be polymorphic because 't' is extensible
    procedure :: s
  end type
contains
  subroutine s(x)
    type(t) :: x  ! x is not polymorphic
  end
end
