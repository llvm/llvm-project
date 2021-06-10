! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Pointer assignment constraints 10.2.2.2

module m1
  type :: t(k)
    integer, kind :: k
  end type
  type t2
    sequence
    real :: t2Field
  end type
contains

  ! C852
  subroutine s0
    !ERROR: 'p1' may not have both the POINTER and TARGET attributes
    real, pointer :: p1, p3
    allocatable :: p2
    !ERROR: 'sin' may not have both the POINTER and INTRINSIC attributes
    real, intrinsic, pointer :: sin
    target :: p1
    !ERROR: 'p2' may not have both the POINTER and ALLOCATABLE attributes
    pointer :: p2
    !ERROR: 'a' may not have the POINTER attribute because it is a coarray
    real, pointer :: a(:)[*]
  end

  ! C1015
  subroutine s1
    real, target :: r
    real(8), target :: r8
    logical, target :: l
    real, pointer :: p
    p => r
    !ERROR: Target type REAL(8) is not compatible with pointer type REAL(4)
    p => r8
    !ERROR: Target type LOGICAL(4) is not compatible with pointer type REAL(4)
    p => l
  end

  ! C1019
  subroutine s2
    real, target :: r1(4), r2(4,4)
    real, pointer :: p(:)
    p => r1
    !ERROR: Pointer has rank 1 but target has rank 2
    p => r2
  end

  ! C1015
  subroutine s3
    type(t(1)), target :: x1
    type(t(2)), target :: x2
    type(t(1)), pointer :: p
    p => x1
    !ERROR: Target type t(k=2_4) is not compatible with pointer type t(k=1_4)
    p => x2
  end

  ! C1016
  subroutine s4(x)
    class(*), target :: x
    type(t(1)), pointer :: p1
    type(t2), pointer :: p2
    class(*), pointer :: p3
    real, pointer :: p4
    p2 => x  ! OK - not extensible
    p3 => x  ! OK - unlimited polymorphic
    !ERROR: Pointer type must be unlimited polymorphic or non-extensible derived type when target is unlimited polymorphic
    p1 => x
    !ERROR: Pointer type must be unlimited polymorphic or non-extensible derived type when target is unlimited polymorphic
    p4 => x
  end

  ! C1020
  subroutine s5
    real, target :: x[*]
    real, target, volatile :: y[*]
    real, pointer :: p
    real, pointer, volatile :: q
    p => x
    !ERROR: Pointer must be VOLATILE when target is a VOLATILE coarray
    p => y
    !ERROR: Pointer may not be VOLATILE when target is a non-VOLATILE coarray
    q => x
    q => y
  end

  ! C1021, C1023
  subroutine s6
    real, target :: x
    real :: p
    type :: tp
      real, pointer :: a
      real :: b
    end type
    type(tp) :: y
    !ERROR: 'p' is not a pointer
    p => x
    y%a => x
    !ERROR: 'b' is not a pointer
    y%b => x
  end

  !C1025 (R1037) The expr shall be a designator that designates a
  !variable with either the TARGET or POINTER attribute and is not
  !an array section with a vector subscript, or it shall be a reference
  !to a function that returns a data pointer.
  subroutine s7
    real, target :: a
    real, pointer :: b
    real, pointer :: c
    real :: d
    b => a
    c => b
    !ERROR: In assignment to object pointer 'b', the target 'd' is not an object with POINTER or TARGET attributes
    b => d
  end

  ! C1025
  subroutine s8
    real :: a(10)
    integer :: b(10)
    real, pointer :: p(:)
    !ERROR: An array section with a vector subscript may not be a pointer target
    p => a(b)
  end

  ! C1025
  subroutine s9
    real, target :: x
    real, pointer :: p
    p => f1()
    !ERROR: pointer 'p' is associated with the result of a reference to function 'f2' that is a not a pointer
    p => f2()
  contains
    function f1()
      real, pointer :: f1
      f1 => x
    end
    function f2()
      real :: f2
      f2 = x
    end
  end

  ! C1026 (R1037) A data-target shall not be a coindexed object.
  subroutine s10
    real, target :: a[*]
    real, pointer :: b
    !ERROR: A coindexed object may not be a pointer target
    b => a[1]
  end

end

module m2
  type :: t1
    real :: a
  end type
  type :: t2
    type(t1) :: b
    type(t1), pointer :: c
    real :: d
  end type
end

subroutine s2
  use m2
  real, pointer :: p
  type(t2), target :: x
  type(t2) :: y
  !OK: x has TARGET attribute
  p => x%b%a
  !OK: c has POINTER attribute
  p => y%c%a
  !ERROR: In assignment to object pointer 'p', the target 'y%b%a' is not an object with POINTER or TARGET attributes
  p => y%b%a
  associate(z => x%b)
    !OK: x has TARGET attribute
    p => z%a
  end associate
  associate(z => y%c)
    !OK: c has POINTER attribute
    p => z%a
  end associate
  associate(z => y%b)
    !ERROR: In assignment to object pointer 'p', the target 'z%a' is not an object with POINTER or TARGET attributes
    p => z%a
  end associate
  associate(z => y%b%a)
    !ERROR: In assignment to object pointer 'p', the target 'z' is not an object with POINTER or TARGET attributes
    p => z
  end associate
end
