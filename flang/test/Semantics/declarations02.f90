! RUN: %python %S/test_errors.py %s %flang_fc1

module m
  !ERROR: 'x1' may not have both the BIND(C) and PARAMETER attributes
  integer, parameter, bind(c, name="a") :: x1 = 1
  !ERROR: 'x2' may not have both the BIND(C) and PARAMETER attributes
  integer, bind(c), parameter :: x2 = 1

  !ERROR: 'x3' may not have both the BIND(C) and PARAMETER attributes
  integer, parameter :: x3 = 1
  bind(c) :: x3

  type :: my_type1
    integer :: x4
  end type
  type, bind(c) :: my_type2
    integer :: x5
  end type

  !ERROR: 't1' may not have both the BIND(C) and PARAMETER attributes
  !ERROR: The derived type of a BIND(C) object must also be BIND(C)
  type(my_type1), bind(c), parameter :: t1 = my_type1(1)
  !ERROR: 't2' may not have both the BIND(C) and PARAMETER attributes
  type(my_type2), bind(c), parameter :: t2 = my_type2(1)

  type(my_type2), parameter :: t3 = my_type2(1) ! no error
  !ERROR: 't4' may not have both the BIND(C) and PARAMETER attributes
  !ERROR: The derived type of a BIND(C) object must also be BIND(C)
  type(my_type1), parameter :: t4 = my_type1(1)
  !ERROR: 't5' may not have both the BIND(C) and PARAMETER attributes
  type(my_type2), parameter :: t5 = my_type2(1)
  bind(c) :: t4, t5

end
