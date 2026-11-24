! RUN: rm -rf %t && mkdir -p %t
! RUN: %flang_fc1 -fsyntax-only -J%t %S/Inputs/type-parameter-inheritance-modfile.f90
! RUN: %python %S/test_errors.py %s %flang_fc1 -J%t
! Reusing an inaccessible inherited name remains valid after module-file readback.

program p
  use m2
  type, extends(base) :: new_private_generics(g, h)
    integer, len :: g, h
  end type

  !ERROR: Type parameter 'public_g' conflicts with an accessible name inherited from a parent type
  type, extends(base) :: public_generic(public_g)
    integer, len :: public_g
  end type
end program
