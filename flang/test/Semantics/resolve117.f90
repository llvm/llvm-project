! RUN: %python %S/test_errors.py %s %flang_fc1
! Test name conflicts with type-bound generics
module m
  type base1(k)
    integer, kind :: k = 4
    real x
   contains
    procedure, nopass :: tbp => sub1
    generic :: gen1 => tbp
    generic :: gen2 => tbp
  end type
  type, extends(base1) :: ext1
   contains
    procedure, nopass :: sub1, sub2
    !ERROR: Type parameter, component, or procedure binding 'base1' already defined in this type
    generic :: base1 => sub1
    !ERROR: Type bound generic procedure 'k' may not have the same name as a non-generic symbol inherited from an ancestor type
    generic :: k => sub1
    !ERROR: Type bound generic procedure 'x' may not have the same name as a non-generic symbol inherited from an ancestor type
    generic :: x => sub1
    !ERROR: Type bound generic procedure 'tbp' may not have the same name as a non-generic symbol inherited from an ancestor type
    generic :: tbp => sub1
    generic :: gen1 => sub1 ! ok
    !ERROR: Generic 'gen2' may not have specific procedures 'tbp' and 'sub2' as their interfaces are not distinguishable
    generic :: gen2 => sub2
  end type
 contains
  subroutine sub1
  end
  subroutine sub2
  end
end
