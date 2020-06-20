! RUN: %S/test_errors.sh %s %t %f18
!Test for checking data constraints, C882-C887
subroutine CheckRepeat
  type person
    integer :: age
    character(len=25) :: name
  end type
  integer, parameter::digits(5) = ( /-11,-22,-33,44,55/ )
  integer ::notConstDigits(5)
  real, parameter::numbers(5) = ( /-11.11,-22.22,-33.33,44.44,55.55/ )
  integer, parameter :: repeat = -1
  integer :: myAge = 2 
  type(person) myName
  !C882
  !ERROR: Missing initialization for parameter 'uninitialized'
  integer, parameter :: uninitialized
  !C882
  !ERROR: Repeat count (-1) for data value must not be negative
  DATA myName%age / repeat * 35 /
  !C882
  !ERROR: Repeat count (-11) for data value must not be negative
  DATA myName%age / digits(1) * 35 /
  !C882
  !ERROR: Must be a constant value
  DATA myName%age / repet * 35 /
  !C885
  !ERROR: Must have INTEGER type, but is REAL(4)
  DATA myName%age / numbers(1) * 35 /
  !C886
  !ERROR: Must be a constant value
  DATA myName%age / notConstDigits(1) * 35 /
  !C887
  !ERROR: Must be a constant value
  DATA myName%age / digits(myAge) * 35 /
end

subroutine CheckValue
  type person
    integer :: age
    character(len=25) :: name
  end type
  integer :: myAge = 2
  type(person) myName
  !OK: constant structure constructor
  data myname / person(1, 'Abcd Ijkl') /
  !C883
  !ERROR: 'persn' is not an array
  data myname / persn(2, 'Abcd Efgh') /
  !C884
  !ERROR: Must be a constant value
  data myname / person(myAge, 'Abcd Ijkl') /
  integer, parameter :: a(5) =(/11, 22, 33, 44, 55/)
  integer :: b(5) =(/11, 22, 33, 44, 55/)
  integer :: i
  integer :: x
  !OK: constant array element
  data x / a(1) /
  !C886, C887
  !ERROR: Must be a constant value
  data x / a(i) /
  !ERROR: Must be a constant value
  data x / b(1) /
end
