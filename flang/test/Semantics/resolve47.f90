! RUN: %S/test_errors.sh %s %t %f18
module m1
  !ERROR: Logical constant '.true.' may not be used as a defined operator
  interface operator(.TRUE.)
  end interface
  !ERROR: Logical constant '.false.' may not be used as a defined operator
  generic :: operator(.false.) => bar
end

module m2
  interface operator(+)
    module procedure foo
  end interface
  interface operator(.foo.)
    module procedure foo
  end interface
  interface operator(.ge.)
    module procedure bar
  end interface
contains
  integer function foo(x, y)
    logical, intent(in) :: x, y
    foo = 0
  end
  logical function bar(x, y)
    complex, intent(in) :: x, y
    bar = .false.
  end
end

!ERROR: Intrinsic operator '.le.' may not be used as a defined operator
use m2, only: operator(.le.) => operator(.ge.)
!ERROR: Intrinsic operator '.not.' may not be used as a defined operator
use m2, only: operator(.not.) => operator(.foo.)
!ERROR: Logical constant '.true.' may not be used as a defined operator
use m2, only: operator(.true.) => operator(.foo.)
end
