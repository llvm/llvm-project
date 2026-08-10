!RUN: %python %S/test_errors.py %s %flang_fc1
module m1
  type pair
  end type
  interface pair
    module procedure f
  end interface
 contains
  type(pair) function f(n)
    integer, intent(in) :: n
    f = pair()
  end
end
module m2
  type pair
  end type
end
module m3
  type pair
  end type
end
program main
  use m1
  use m2
  use m3
  !ERROR: Reference to 'pair' is ambiguous
  type(pair) error
end
