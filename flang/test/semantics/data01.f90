!Test for checking data constraints, C882-C887
module m1
  type person
    integer :: age
    character(len=25) :: name
  end type
  integer, parameter::digits(5) = ( /-11,-22,-33,44,55/ )
  integer ::notConstDigits(5) = ( /-11,-22,-33,44,55/ )
  real, parameter::numbers(5) = ( /-11.11,-22.22,-33.33,44.44,55.55/ )
  integer, parameter :: repeat = -1
  integer :: myAge = 2 
  type(person) myName
end

subroutine CheckRepeat
  use m1
  !C882
  !ERROR: Missing initialization for parameter 'uninitialized'
  integer, parameter :: uninitialized
  !C882
  !ERROR: Repeat count for data value must not be negative
  DATA myName%age / repeat * 35 /
  !C882
  !ERROR: Repeat count for data value must not be negative
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
  use m1
  !C883
  !ERROR: Derived type 'persn' not found
  DATA myname / persn(2, 'Abcd Efgh') /
  !C884
  !ERROR: Structure constructor in data value must be a constant expression
  DATA myname / person(myAge, 'Abcd Ijkl') /
end
