! RUN: %python %S/test_errors.py %s %flang_fc1
! C772
module m1
  type t1
  contains
    procedure, nopass :: s1
    !ERROR: Binding name 's2' not found in this derived type
    generic :: g1 => s2
  end type
  type t2
    integer :: s1
  contains
    !ERROR: 's1' is not the name of a specific binding of this type
    generic :: g2 => s1
  end type
contains
  subroutine s1
  end
end

module m2
  type :: t3
  contains
    private
    procedure, nopass :: s3
    generic, public :: g3 => s3
    generic :: h3 => s3
  end type
contains
  subroutine s3(i)
  end
end

! C771
module m3
  use m2
  type, extends(t3) :: t4
  contains
    procedure, nopass :: s4
    procedure, nopass :: s5
    !ERROR: 'g3' does not have the same accessibility as its previous declaration
    generic, private :: g3 => s4
    !ERROR: 'h3' does not have the same accessibility as its previous declaration
    generic, public :: h3 => s4
    generic :: i3 => s4
    !ERROR: 'i3' does not have the same accessibility as its previous declaration
    generic, private :: i3 => s5
  end type
  type :: t5
  contains
    private
    procedure, nopass :: s3
    procedure, nopass :: s4
    procedure, nopass :: s5
    generic :: g5 => s3, s4
    !ERROR: 'g5' does not have the same accessibility as its previous declaration
    generic, public :: g5 => s5
  end type
contains
  subroutine s4(r)
  end
  subroutine s5(z)
    complex :: z
  end
end

! Test forward reference in type-bound generic to binding is allowed
module m4
  type :: t1
  contains
    generic :: g => s1
    generic :: g => s2
    procedure, nopass :: s1
    procedure, nopass :: s2
  end type
  type :: t2
  contains
    generic :: g => p1
    generic :: g => p2
    procedure, nopass :: p1 => s1
    procedure, nopass :: p2 => s2
  end type
contains
  subroutine s1()
  end
  subroutine s2(x)
  end
end

! C773 - duplicate binding names
module m5
  type :: t1
  contains
    generic :: g => s1
    generic :: g => s2
    procedure, nopass :: s1
    procedure, nopass :: s2
    !ERROR: Binding name 's1' was already specified for generic 'g'
    generic :: g => s1
  end type
contains
  subroutine s1()
  end
  subroutine s2(x)
  end
end

module m6
  type t
  contains
    procedure :: f1
    procedure :: f2
    generic :: operator(.eq.) => f1
    !ERROR: Binding name 'f1' was already specified for generic 'operator(.eq.)'
    generic :: operator(==) => f2, f1
  end type
contains
  logical function f1(x, y) result(result)
    class(t), intent(in) :: x
    real, intent(in) :: y
    result = .true.
  end
  logical function f2(x, y) result(result)
    class(t), intent(in) :: x
    integer, intent(in) :: y
    result = .true.
  end
end
