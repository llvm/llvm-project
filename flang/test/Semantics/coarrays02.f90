! RUN: %python %S/test_errors.py %s %flang_fc1
! More coarray error tests.
module m
  integer :: local[*] ! ok in module
end
program main
  use iso_fortran_env
  !ERROR: Coarray 'namedconst' may not be a named constant
  !ERROR: Local coarray must have the SAVE attribute
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
  !ERROR: Local coarray must have the SAVE attribute
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
  !ERROR: Local variable 'local' without the SAVE attribute may not have a coarray potential subobject component '%comp'
  type(t) :: local
end
