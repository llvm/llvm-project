! RUN: not %flang_fc1 -fsyntax-only -fopenacc %s 2>&1 | FileCheck %s

! An associate name does not have the POINTER or ALLOCATABLE attribute of its
! selector, so it cannot appear on an ATTACH or DETACH clause. Check that the
! error explains why, and suggests the selector when the selector would be
! accepted.

module attach_associate_m
  type :: t
    real, pointer :: p(:) => null()
    real, allocatable :: a(:)
    real :: r(10)
  end type
end module

subroutine test_attach_associate(self)
  use attach_associate_m
  type(t) :: self

  associate (ptr => self%p, alc => self%a, arr => self%r)
    ! CHECK: error: Argument `ptr` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
    ! CHECK: `ptr` is an associate name and does not have the POINTER or ALLOCATABLE attribute of its selector; use `self%p` in the ATTACH clause instead
    !$acc enter data attach(ptr)

    ! CHECK: error: Argument `alc` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
    ! CHECK: `alc` is an associate name and does not have the POINTER or ALLOCATABLE attribute of its selector; use `self%a` in the ATTACH clause instead
    !$acc enter data attach(alc)

    ! CHECK: error: Argument `ptr` on the DETACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
    ! CHECK: `ptr` is an associate name and does not have the POINTER or ALLOCATABLE attribute of its selector; use `self%p` in the DETACH clause instead
    !$acc exit data detach(ptr)

    ! The selector is not a pointer either, so there is nothing to suggest.
    ! CHECK: error: Argument `arr` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
    ! CHECK: `arr` is an associate name and does not have the POINTER or ALLOCATABLE attribute{{$}}
    !$acc enter data attach(arr)
  end associate

  ! The selector is an expression, not a variable.
  associate (e => self%p + 1.0)
    ! CHECK: error: Argument `e` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
    ! CHECK: `e` is an associate name and does not have the POINTER or ALLOCATABLE attribute{{$}}
    !$acc enter data attach(e)
  end associate

  ! Naming the pointer component directly is accepted.
  ! CHECK-NOT: error:
  !$acc enter data attach(self%p)
  !$acc exit data detach(self%p)
end subroutine

! A SELECT RANK associate name keeps the POINTER attribute of its selector.
subroutine test_attach_select_rank(x)
  real, pointer :: x(..)
  select rank (y => x)
  rank (1)
    !$acc enter data attach(y)
  end select
end subroutine
