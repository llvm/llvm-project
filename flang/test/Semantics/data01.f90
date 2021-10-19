! RUN: %python %S/test_errors.py %s %flang_fc1
!Test for checking data constraints, C882-C887
module m1
  type person
    integer :: age
    character(len=25) :: name
  end type
  integer, parameter::digits(5) = ( /-11,-22,-33,44,55/ )
  integer ::notConstDigits(5)
  real, parameter::numbers(5) = ( /-11.11,-22.22,-33.33,44.44,55.55/ )
  integer, parameter :: repeat = -1
  integer :: myAge = 2
  type(person) associated
end

subroutine CheckRepeat
  use m1
  type(person) myName(6)
  !C882
  !ERROR: Missing initialization for parameter 'uninitialized'
  integer, parameter :: uninitialized
  !C882
  !ERROR: Repeat count (-1) for data value must not be negative
  DATA myName(1)%age / repeat * 35 /
  !C882
  !ERROR: Repeat count (-11) for data value must not be negative
  DATA myName(2)%age / digits(1) * 35 /
  !C882
  !ERROR: Must be a constant value
  DATA myName(3)%age / repet * 35 /
  !C885
  !ERROR: Must have INTEGER type, but is REAL(4)
  DATA myName(4)%age / numbers(1) * 35 /
  !C886
  !ERROR: Must be a constant value
  DATA myName(5)%age / notConstDigits(1) * 35 /
  !C887
  !ERROR: Must be a constant value
  DATA myName(6)%age / digits(myAge) * 35 /
end

subroutine CheckValue
  use m1
  !ERROR: USE-associated object 'associated' must not be initialized in a DATA statement
  data associated / person(1, 'Abcd Ijkl') /
  type(person) myName(3)
  !OK: constant structure constructor
  data myname(1) / person(1, 'Abcd Ijkl') /
  !C883
  !ERROR: 'persn' is not an array
  data myname(2) / persn(2, 'Abcd Efgh') /
  !C884
  !ERROR: DATA statement value 'person(age=myage,name="Abcd Ijkl                ")' for 'myname(3_8)%age' is not a constant
  data myname(3) / person(myAge, 'Abcd Ijkl') /
  integer, parameter :: a(5) =(/11, 22, 33, 44, 55/)
  integer :: b(5) =(/11, 22, 33, 44, 55/)
  integer :: i
  integer :: x, y, z
  !OK: constant array element
  data x / a(1) /
  !C886, C887
  !ERROR: DATA statement value 'a(int(i,kind=8))' for 'y' is not a constant
  data y / a(i) /
  !ERROR: DATA statement value 'b(1_8)' for 'z' is not a constant
  data z / b(1) /
end
