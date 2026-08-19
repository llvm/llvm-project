! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Check that an ATTACH/DETACH argument whose name could not be resolved is
! diagnosed instead of crashing the compiler.

subroutine test_attach_unresolved
  type :: ty
    integer :: i
  end type ty
  type(ty) :: x

  !ERROR: Component 'bad' not found in derived type 'ty'
  !$acc enter data attach(x%bad)

  !ERROR: Component 'bad' not found in derived type 'ty'
  !$acc exit data detach(x%bad)
end subroutine test_attach_unresolved
