! RUN: %flang -fsyntax-only -pedantic 2>&1 %s | FileCheck %s
! Verifies that warnings issue when actual arguments with implicit
! interfaces are associated with dummy procedures and dummy procedure
! pointers whose interfaces are explicit.
module m
 contains
  real function realfunc(x)
    real, intent(in) :: x
    realfunc = x
  end function
  subroutine s00(p0)
    procedure(realfunc) :: p0
  end subroutine
  subroutine s01(p1)
    procedure(realfunc), pointer, intent(in) :: p1
  end subroutine
  subroutine s02(p2)
    procedure(realfunc), pointer :: p2
  end subroutine
  subroutine test
    external :: extfunc
    external :: extfuncPtr
    pointer :: extfuncPtr
    !CHECK: Actual procedure argument has an implicit interface which is not known to be compatible with dummy argument 'p0=' which has an explicit interface
    call s00(extfunc)
    !CHECK: Actual procedure argument has an implicit interface which is not known to be compatible with dummy argument 'p1=' which has an explicit interface
    call s01(extfunc)
    !CHECK: Actual procedure argument has an implicit interface which is not known to be compatible with dummy argument 'p2=' which has an explicit interface
    call s02(extfuncPtr)
  end subroutine
end module
