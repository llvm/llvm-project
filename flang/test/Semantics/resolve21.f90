! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
subroutine s1
  type :: t
    integer :: i
    integer :: s1
    integer :: t
  end type
  !ERROR: 't' is already declared in this scoping unit
  integer :: t
  integer :: i, j
  type(t) :: x
  !ERROR: Derived type 't2' not found
  type(t2) :: y
  external :: v
  type(t) :: v, w
  external :: w
  !ERROR: 'z' is not an object of derived type; it is implicitly typed
  i = z%i
  !ERROR: 's1' is an invalid base for a component reference
  i = s1%i
  !ERROR: 'j' is not an object of derived type
  i = j%i
  !ERROR: Component 'j' not found in derived type 't'
  i = x%j
  !ERROR: 'v' is an invalid base for a component reference
  i = v%i
  !ERROR: 'w' is an invalid base for a component reference
  i = w%i
  i = x%i  !OK
end subroutine

subroutine s2
  type :: t1
    integer :: i
  end type
  type :: t2
    type(t1) :: x
  end type
  type(t2) :: y
  integer :: i
  !ERROR: Component 'j' not found in derived type 't1'
  k = y%x%j
  k = y%x%i !OK
end subroutine
