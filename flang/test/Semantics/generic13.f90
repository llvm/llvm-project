! RUN: %python %S/test_errors.py %s %flang_fc1
module m1
  type, abstract :: ta1
   contains
    procedure(ta1p1), deferred :: ta1p1
    generic :: gen => ta1p1
  end type
  abstract interface
    subroutine ta1p1(x)
      import ta1
      class(ta1), intent(in) :: x
    end
  end interface
  type :: tb1
   contains
    procedure tb1p1
    generic :: gen => tb1p1
  end type
  type :: tc1
   contains
    procedure tc1p1
    generic, private :: gen => tc1p1
  end type
  type :: td1
   contains
    procedure, nopass :: td1p1
    generic :: gen => td1p1
  end type
 contains
  subroutine tb1p1(x)
    class(tb1), intent(in) :: x
  end
  subroutine tb1p2(x)
    class(tb1), intent(in) :: x
  end
  subroutine tc1p1(x)
    class(tc1), intent(in) :: x
  end
  subroutine td1p1
  end
end

module m2
  use m1
  type, extends(ta1) :: ta2a
   contains
    procedure :: ta1p1 => ta2ap1 ! ok
  end type
  type, extends(ta1) :: ta2b
   contains
    procedure :: ta1p1 => ta2bp1
    generic :: gen => ta1p1 ! ok, overidden deferred
  end type
  type, extends(tb1) :: tb2a
   contains
    generic :: gen => tb1p1 ! ok, same binding
  end type
  type, extends(tb1) :: tb2b
   contains
    procedure :: tb1p1 => tb2bp2
    generic :: gen => tb1p1 ! ok, overridden
  end type
  type, extends(tb1) :: tb2c
   contains
    procedure tb2cp1
    !ERROR: Generic 'gen' may not have specific procedures 'tb1p1' and 'tb2cp1' as their interfaces are not distinguishable
    generic :: gen => tb2cp1
  end type
  type, extends(tc1) :: tc2
   contains
    procedure tc2p1
    !ERROR: 'gen' does not have the same accessibility as its previous declaration
    generic :: gen => tc2p1
  end type
  type, extends(td1) :: td2
   contains
    procedure, nopass :: td2p1 => td1p1
    generic :: gen => td2p1 ! ok, same procedure
  end type
 contains
  subroutine ta2ap1(x)
    class(ta2a), intent(in) :: x
  end
  subroutine ta2bp1(x)
    class(ta2b), intent(in) :: x
  end
  subroutine tb2bp2(x)
    class(tb2b), intent(in) :: x
  end
  subroutine tb2cp1(x)
    class(tb2c), intent(in) :: x
  end
  subroutine tc2p1(x)
    class(tc2), intent(in) :: x
  end
end
