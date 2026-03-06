! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  type t
   contains
    procedure :: tweedledee
    generic :: operator(.ga.) => tweedledee
    generic, private :: operator(.gb.) => tweedledee
  end type
  interface operator(.gc.)
    module procedure tweedledum
  end interface
 contains
  integer function tweedledee(x,y)
    class(t), intent(in) :: x, y
    tweedledee = 1
  end
  integer function tweedledum(x,y)
    class(t), intent(in) :: x, y
    tweedledum = 2
  end
end

module badDueToAccessibility
  !ERROR: Generic 'OPERATOR(.ga.)' may not have specific procedures 'tweedledum' and 't%tweedledee' as their interfaces are not distinguishable
  use m, operator(.ga.) => operator(.gc.)
end

module goodDueToInaccessibility
  use m, operator(.gb.) => operator(.gc.)
end
