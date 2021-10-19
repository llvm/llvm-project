! RUN: %python %S/test_errors.py %s %flang_fc1
! 15.4.3.4.5 Restrictions on generic declarations
! Specific procedures of generic interfaces must be distinguishable.

module m1
  !ERROR: Generic 'g' may not have specific procedures 's2' and 's4' as their interfaces are not distinguishable
  interface g
    procedure s1
    procedure s2
    procedure s3
    procedure s4
  end interface
contains
  subroutine s1(x)
    integer(8) x
  end
  subroutine s2(x)
    integer x
  end
  subroutine s3
  end
  subroutine s4(x)
    integer x
  end
end

module m2
  !ERROR: Generic 'g' may not have specific procedures 's1' and 's2' as their interfaces are not distinguishable
  interface g
    subroutine s1(x)
    end subroutine
    subroutine s2(x)
      real x
    end subroutine
  end interface
end

module m3
  !ERROR: Generic 'g' may not have specific procedures 'f1' and 'f2' as their interfaces are not distinguishable
  interface g
    integer function f1()
    end function
    real function f2()
    end function
  end interface
end

module m4
  type :: t1
  end type
  type, extends(t1) :: t2
  end type
  interface g
    subroutine s1(x)
      import :: t1
      type(t1) :: x
    end
    subroutine s2(x)
      import :: t2
      type(t2) :: x
    end
  end interface
end

! These are all different ranks so they are distinguishable
module m5
  interface g
    subroutine s1(x)
      real x
    end subroutine
    subroutine s2(x)
      real x(:)
    end subroutine
    subroutine s3(x)
      real x(:,:)
    end subroutine
  end interface
end

module m6
  use m5
  !ERROR: Generic 'g' may not have specific procedures 's1' and 's4' as their interfaces are not distinguishable
  interface g
    subroutine s4(x)
    end subroutine
  end interface
end

module m7
  use m5
  !ERROR: Generic 'g' may not have specific procedures 's1' and 's5' as their interfaces are not distinguishable
  !ERROR: Generic 'g' may not have specific procedures 's2' and 's5' as their interfaces are not distinguishable
  !ERROR: Generic 'g' may not have specific procedures 's3' and 's5' as their interfaces are not distinguishable
  interface g
    subroutine s5(x)
      real x(..)
    end subroutine
  end interface
end
    

! Two procedures that differ only by attributes are not distinguishable
module m8
  !ERROR: Generic 'g' may not have specific procedures 's1' and 's2' as their interfaces are not distinguishable
  interface g
    pure subroutine s1(x)
      real, intent(in) :: x
    end subroutine
    subroutine s2(x)
      real, intent(in) :: x
    end subroutine
  end interface
end

module m9
  !ERROR: Generic 'g' may not have specific procedures 's1' and 's2' as their interfaces are not distinguishable
  interface g
    subroutine s1(x)
      real :: x(10)
    end subroutine
    subroutine s2(x)
      real :: x(100)
    end subroutine
  end interface
end

module m10
  !ERROR: Generic 'g' may not have specific procedures 's1' and 's2' as their interfaces are not distinguishable
  interface g
    subroutine s1(x)
      real :: x(10)
    end subroutine
    subroutine s2(x)
      real :: x(..)
    end subroutine
  end interface
end

program m11
  interface g1
    subroutine s1(x)
      real, pointer, intent(out) :: x
    end subroutine
    subroutine s2(x)
      real, allocatable :: x
    end subroutine
  end interface
  !ERROR: Generic 'g2' may not have specific procedures 's3' and 's4' as their interfaces are not distinguishable
  interface g2
    subroutine s3(x)
      real, pointer, intent(in) :: x
    end subroutine
    subroutine s4(x)
      real, allocatable :: x
    end subroutine
  end interface
end

module m12
  !ERROR: Generic 'g1' may not have specific procedures 's1' and 's2' as their interfaces are not distinguishable
  generic :: g1 => s1, s2  ! rank-1 and assumed-rank
  !ERROR: Generic 'g2' may not have specific procedures 's2' and 's3' as their interfaces are not distinguishable
  generic :: g2 => s2, s3  ! scalar and assumed-rank
  !ERROR: Generic 'g3' may not have specific procedures 's1' and 's4' as their interfaces are not distinguishable
  generic :: g3 => s1, s4  ! different shape, same rank
contains
  subroutine s1(x)
    real :: x(10)
  end
  subroutine s2(x)
    real :: x(..)
  end
  subroutine s3(x)
    real :: x
  end
  subroutine s4(x)
    real :: x(100)
  end
end

! Procedures that are distinguishable by return type of a dummy argument
module m13
  interface g1
    procedure s1
    procedure s2
  end interface
  interface g2
    procedure s1
    procedure s3
  end interface
contains
  subroutine s1(x)
    procedure(real), pointer :: x
  end
  subroutine s2(x)
    procedure(integer), pointer :: x
  end
  subroutine s3(x)
    interface
      function x()
        procedure(real), pointer :: x
      end function
    end interface
  end
end

! Check user-defined operators
module m14
  interface operator(*)
    module procedure f1
    module procedure f2
  end interface
  !ERROR: Generic 'OPERATOR(+)' may not have specific procedures 'f1' and 'f3' as their interfaces are not distinguishable
  interface operator(+)
    module procedure f1
    module procedure f3
  end interface
  interface operator(.foo.)
    module procedure f1
    module procedure f2
  end interface
  !ERROR: Generic 'OPERATOR(.bar.)' may not have specific procedures 'f1' and 'f3' as their interfaces are not distinguishable
  interface operator(.bar.)
    module procedure f1
    module procedure f3
  end interface
contains
  real function f1(x, y)
    real, intent(in) :: x
    logical, intent(in) :: y
  end
  integer function f2(x, y)
    integer, intent(in) :: x
    logical, intent(in) :: y
  end
  real function f3(x, y)
    real, value :: x
    logical, value :: y
  end
end module

! Types distinguished by kind (but not length) parameters
module m15
  type :: t1(k1, l1)
    integer, kind :: k1 = 1
    integer, len :: l1 = 101
  end type

  type, extends(t1) :: t2(k2a, l2, k2b)
    integer, kind :: k2a = 2
    integer, kind :: k2b = 3
    integer, len :: l2 = 102
  end type

  type, extends(t2) :: t3(l3, k3)
    integer, kind :: k3 = 4
    integer, len :: l3 = 103
  end type

  interface g1
    procedure s1
    procedure s2
  end interface
  !ERROR: Generic 'g2' may not have specific procedures 's1' and 's3' as their interfaces are not distinguishable
  interface g2
    procedure s1
    procedure s3
  end interface
  !ERROR: Generic 'g3' may not have specific procedures 's4' and 's5' as their interfaces are not distinguishable
  interface g3
    procedure s4
    procedure s5
  end interface
  interface g4
    procedure s5
    procedure s6
    procedure s9
  end interface
  interface g5
    procedure s4
    procedure s7
    procedure s9
  end interface
  interface g6
    procedure s5
    procedure s8
    procedure s9
  end interface
  !ERROR: Generic 'g7' may not have specific procedures 's6' and 's7' as their interfaces are not distinguishable
  interface g7
    procedure s6
    procedure s7
  end interface
  !ERROR: Generic 'g8' may not have specific procedures 's6' and 's8' as their interfaces are not distinguishable
  interface g8
    procedure s6
    procedure s8
  end interface
  !ERROR: Generic 'g9' may not have specific procedures 's7' and 's8' as their interfaces are not distinguishable
  interface g9
    procedure s7
    procedure s8
  end interface

contains
  subroutine s1(x)
    type(t1(1, 5)) :: x
  end
  subroutine s2(x)
    type(t1(2, 4)) :: x
  end
  subroutine s3(x)
    type(t1(l1=5)) :: x
  end
  subroutine s4(x)
    type(t3(1, 101, 2, 102, 3, 103, 4)) :: x
  end subroutine
  subroutine s5(x)
    type(t3) :: x
  end subroutine
  subroutine s6(x)
    type(t3(1, 99, k2b=2, k2a=3, l2=*, l3=103, k3=4)) :: x
  end subroutine
  subroutine s7(x)
    type(t3(k1=1, l1=99, k2a=3, k2b=2, k3=4)) :: x
  end subroutine
  subroutine s8(x)
    type(t3(1, :, 3, :, 2, :, 4)), allocatable :: x
  end subroutine
  subroutine s9(x)
    type(t3(k1=2)) :: x
  end subroutine
end

! Check that specifics for type-bound generics can be distinguished
module m16
  type :: t
  contains
    procedure, nopass :: s1
    procedure, nopass :: s2
    procedure, nopass :: s3
    generic :: g1 => s1, s2
    !ERROR: Generic 'g2' may not have specific procedures 's1' and 's3' as their interfaces are not distinguishable
    generic :: g2 => s1, s3
  end type
contains
  subroutine s1(x)
    real :: x
  end
  subroutine s2(x)
    integer :: x
  end
  subroutine s3(x)
    real :: x
  end
end

! Check polymorphic types
module m17
  type :: t
  end type
  type, extends(t) :: t1
  end type
  type, extends(t) :: t2
  end type
  type, extends(t2) :: t2a
  end type
  interface g1
    procedure s1
    procedure s2
  end interface
  !ERROR: Generic 'g2' may not have specific procedures 's3' and 's4' as their interfaces are not distinguishable
  interface g2
    procedure s3
    procedure s4
  end interface
  interface g3
    procedure s1
    procedure s4
  end interface
  !ERROR: Generic 'g4' may not have specific procedures 's2' and 's3' as their interfaces are not distinguishable
  interface g4
    procedure s2
    procedure s3
  end interface
  !ERROR: Generic 'g5' may not have specific procedures 's2' and 's5' as their interfaces are not distinguishable
  interface g5
    procedure s2
    procedure s5
  end interface
  !ERROR: Generic 'g6' may not have specific procedures 's2' and 's6' as their interfaces are not distinguishable
  interface g6
    procedure s2
    procedure s6
  end interface
contains
  subroutine s1(x)
    type(t) :: x
  end
  subroutine s2(x)
    type(t2a) :: x
  end
  subroutine s3(x)
    class(t) :: x
  end
  subroutine s4(x)
    class(t2) :: x
  end
  subroutine s5(x)
    class(*) :: x
  end
  subroutine s6(x)
    type(*) :: x
  end
end

! Test C1514 rule 3 -- distinguishable passed-object dummy arguments
module m18
  type :: t(k)
    integer, kind :: k
  contains
    procedure, pass(x) :: p1 => s
    procedure, pass    :: p2 => s
    procedure          :: p3 => s
    procedure, pass(y) :: p4 => s
    generic :: g1 => p1, p4
    generic :: g2 => p2, p4
    generic :: g3 => p3, p4
  end type
contains
  subroutine s(x, y)
    class(t(1)) :: x
    class(t(2)) :: y
  end
end

! C1511 - rules for operators
module m19
  interface operator(.foo.)
    module procedure f1
    module procedure f2
  end interface
  !ERROR: Generic 'OPERATOR(.bar.)' may not have specific procedures 'f2' and 'f3' as their interfaces are not distinguishable
  interface operator(.bar.)
    module procedure f2
    module procedure f3
  end interface
contains
  integer function f1(i)
    integer, intent(in) :: i
  end
  integer function f2(i, j)
    integer, value :: i, j
  end
  integer function f3(i, j)
    integer, intent(in) :: i, j
  end
end

module m20
  interface operator(.not.)
    real function f(x)
      character(*),intent(in) :: x
    end function
  end interface
  interface operator(+)
    procedure f
  end interface
end module

subroutine s1()
  use m20
  interface operator(.not.)
    !ERROR: Procedure 'f' from module 'm20' is already specified in generic 'OPERATOR(.NOT.)'
    procedure f
  end interface
  interface operator(+)
    !ERROR: Procedure 'f' from module 'm20' is already specified in generic 'OPERATOR(+)'
    procedure f
  end interface
end subroutine s1
