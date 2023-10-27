! RUN: %python %S/test_errors.py %s %flang_fc1
! Test name conflicts with type-bound generics
module m
  type base1(k)
    integer, kind :: k = 4
    real x
   contains
    procedure, nopass :: tbp => sub
    generic :: gen => tbp
  end type
  type, extends(base1) :: ext1
   contains
    procedure, nopass :: sub
    !ERROR: Type parameter, component, or procedure binding 'base1' already defined in this type
    generic :: base1 => sub
    !ERROR: Type bound generic procedure 'k' may not have the same name as a non-generic symbol inherited from an ancestor type
    generic :: k => sub
    !ERROR: Type bound generic procedure 'x' may not have the same name as a non-generic symbol inherited from an ancestor type
    generic :: x => sub
    !ERROR: Type bound generic procedure 'tbp' may not have the same name as a non-generic symbol inherited from an ancestor type
    generic :: tbp => sub
    generic :: gen => sub ! ok
  end type
 contains
  subroutine sub
  end
end
