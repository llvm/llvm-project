! RUN: %python %S/test_errors.py %s %flang_fc1
module explicit
  intrinsic cos
end
subroutine testExplicit
  use explicit
  !ERROR: 'cos' is use-associated from module 'explicit' and cannot be re-declared
  real :: cos = 2.
end
subroutine extendsUsedIntrinsic
  use explicit
  interface cos
    pure real function mycos(x)
      real, intent(in) :: x
    end
  end interface
end
subroutine sameIntrinsic1
  use explicit
  !WARNING: Use-associated 'cos' already has 'INTRINSIC' attribute
  intrinsic cos
  real :: one = cos(0.)
end
module renamer
  use explicit, renamedCos => cos
end
subroutine sameIntrinsic2
  use explicit
  use renamer, cos => renamedCos
  real :: one = cos(0.)
end
module implicit
  real :: one = cos(0.)
end
subroutine testImplicit
  use implicit
  real :: cos = 2.
end
