!RUN: %flang_fc1 -fsyntax-only %s  2>&1 | FileCheck %s --allow-empty
!CHECK-NOT: error:
module m
  type t
   contains
    procedure nonelemental
    generic :: operator(+) => nonelemental
  end type
  interface operator(+)
    procedure elemental
  end interface
 contains
  type(t) elemental function elemental (a, b)
    class(t), intent(in) :: a, b
    elemental = t()
  end
  type(t) function nonelemental (a, b)
    class(t), intent(in) :: a, b(:)
    nonelemental = t()
  end
end
program main
  use m
  type(t) x, y(1)
  x = x + y ! ok
end
