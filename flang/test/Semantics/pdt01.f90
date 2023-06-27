! RUN: %python %S/test_errors.py %s %flang_fc1
! Catch error instead of crashing with infinite recursion
! when a LEN PDT from one type is being used to define a
! LEN PDT in another type's instantiation.
program main
  type t1(lp)
    integer, len :: lp
  end type
  type t2(lp)
    integer, len :: lp
    type(t1(lp)) :: c
  end type
  integer local
  !ERROR: Invalid specification expression: reference to local entity 'local'
  type(t2(local)) :: x
end
