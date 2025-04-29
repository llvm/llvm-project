! RUN: %python %S/test_errors.py %s %flang_fc1
! More coarray error tests.
module m1
  integer :: local[*] ! ok in module
end
program main
  use iso_fortran_env
  !ERROR: Coarray 'namedconst' may not be a named constant
  !ERROR: Local coarray must have the SAVE or ALLOCATABLE attribute
  integer, parameter :: namedConst = 123
  codimension namedConst[*]
  !ERROR: Coarray 'coarr1' may not be in COMMON block '//'
  real :: coarr1[*]
  common//coarr1
  !ERROR: Variable 'event' with EVENT_TYPE or LOCK_TYPE must be a coarray
  type(event_type) event
  !ERROR: Variable 'lock' with EVENT_TYPE or LOCK_TYPE must be a coarray
  type(lock_type) lock
  integer :: local[*] ! ok in main
end

function func1()
  !ERROR: Function result may not be a coarray
  integer :: func1[*]
  !ERROR: Local coarray must have the SAVE or ALLOCATABLE attribute
  integer :: local[*]
  integer, save :: saved[*] ! ok
  integer :: inited[*] = 1 ! ok
  func = 1
end

function func2()
  type t
    real, allocatable :: comp[:]
  end type
  type t2
    !ERROR: Allocatable or array component 'allo' may not have a coarray ultimate component '%comp'
    type(t), allocatable :: allo
    !ERROR: Allocatable or array component 'arr' may not have a coarray ultimate component '%comp'
    type(t) :: arr(1)
  end type
  !ERROR: Function result 'func2' may not have a coarray potential component '%comp'
  type(t) func2
  !ERROR: Pointer 'ptr' may not have a coarray potential component '%comp'
  type(t), pointer :: ptr
  !ERROR: Coarray 'coarr' may not have a coarray potential component '%comp'
  type(t), save :: coarr[*]
  !ERROR: Local variable 'local' without the SAVE or ALLOCATABLE attribute may not have a coarray potential subobject component '%comp'
  type(t) :: local
end

module m2
  type t0
    integer n
  end type
  type t1
    class(t0), allocatable :: a
  end type
  type t2
    type(t1) c
  end type
 contains
  subroutine test(x)
    type(t2), intent(in) :: x[*]
    !ERROR: The base of a polymorphic object may not be coindexed
    call sub1(x[1]%c%a)
    !ERROR: A coindexed designator may not have a type with the polymorphic potential subobject component '%a'
    call sub2(x[1]%c)
  end
  subroutine sub1(x)
    type(t0), intent(in) :: x
  end
  subroutine sub2(x)
    type(t1), intent(in) :: x
  end
end

module m3
  type t
    real, allocatable :: a(:)
    real, pointer :: p(:)
    real arr(2)
  end type
 contains
  subroutine sub(ca)
    real, intent(in) :: ca(:)[*]
  end
  subroutine test(cat)
    type(t), intent(in) :: cat[*]
    call sub(cat%arr(1:2)) ! ok
    !ERROR: Actual argument associated with coarray dummy argument 'ca=' must be a coarray
    call sub(cat%arr([1]))
    !ERROR: Actual argument associated with coarray dummy argument 'ca=' must be a coarray
    call sub(cat%a)
    !ERROR: Actual argument associated with coarray dummy argument 'ca=' must be a coarray
    call sub(cat%p)
  end
end
