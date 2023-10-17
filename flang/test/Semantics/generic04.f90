! RUN: %python %S/test_errors.py %s %flang_fc1
! Warn about inaccessible specific procedures in a generic defined operator
module m
  interface operator (.foo.)
    !WARN: OPERATOR(.foo.) function 'noargs' must have 1 or 2 dummy arguments
    module procedure noargs
    !WARN: OPERATOR(.foo.) function 'noargs' must have 1 or 2 dummy arguments
    module procedure threeargs
  end interface
  type t
   contains
    procedure :: bad
    !WARN: OPERATOR(.bar.) function 'bad' should have 1 or 2 dummy arguments
    generic :: operator (.bar.) => bad
  end type
 contains
  real function noargs()
    noargs = 0.
  end
  real function threeargs(fee,fie,foe)
    real, intent(in) :: fee, fie, foe
  end
  function bad(this,x,y)
    type(t) :: bad
    class(t), intent(in) :: this, x, y
    bad = x
  end
end
