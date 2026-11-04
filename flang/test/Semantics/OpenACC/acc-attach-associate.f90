! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! An ASSOCIATE or SELECT TYPE associate name does not have the POINTER or
! ALLOCATABLE attribute of its selector, so it cannot appear on an ATTACH or
! DETACH clause. When the selector itself would be accepted, the error explains
! that association is the cause. Otherwise only the error is reported.

module attach_associate_m
  type :: t
    real, pointer :: p(:) => null()
    real, allocatable :: a(:)
    real :: r(10)
    complex, pointer :: z => null()
  end type
end module

subroutine test_attach_associate(self, other)
  use attach_associate_m
  type(t) :: self, other

  !BECAUSE: `ptr` is an associate name and does not have the POINTER or ALLOCATABLE attribute of its selector
  !BECAUSE: `ptr` is an associate name and does not have the POINTER or ALLOCATABLE attribute of its selector
  !BECAUSE: `alc` is an associate name and does not have the POINTER or ALLOCATABLE attribute of its selector
  associate (ptr => self%p, alc => self%a, arr => self%r)
    !ERROR: Argument `ptr` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
    !$acc enter data attach(ptr)
    !ERROR: Argument `alc` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
    !$acc enter data attach(alc)
    !ERROR: Argument `ptr` on the DETACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
    !$acc exit data detach(ptr)
    ! The selector would be rejected too: no note.
    !ERROR: Argument `arr` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
    !$acc enter data attach(arr)
  end associate

  ! The selector would be rejected too: no note.
  associate (re => self%z%re, e => self%p + 1.0)
    !ERROR: Argument `re` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
    !$acc enter data attach(re)
    !ERROR: Argument `e` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
    !$acc enter data attach(e)
  end associate

  ! A nested associate name's selector is itself an associate name, which
  ! would be rejected too: no note.
  !BECAUSE: `a1` is an associate name and does not have the POINTER or ALLOCATABLE attribute of its selector
  associate (a1 => self%p)
    associate (a2 => a1)
      !ERROR: Argument `a2` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
      !$acc enter data attach(a2)
      !ERROR: Argument `a1` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
      !$acc enter data attach(a1)
    end associate
  end associate

  ! The note does not name the selector, whose names can mean something else
  ! at the clause: here `self%p` would name `other%p`.
  !BECAUSE: `q` is an associate name and does not have the POINTER or ALLOCATABLE attribute of its selector
  associate (q => self%p)
    associate (self => other)
      !ERROR: Argument `q` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
      !$acc enter data attach(q)
    end associate
  end associate

  ! Naming the pointer or allocatable directly is accepted.
  !$acc enter data attach(self%p, self%a)
  !$acc exit data detach(self%p, self%a)
end subroutine

subroutine test_attach_select_type(c, x)
  class(*), pointer :: c, x(:)

  !BECAUSE: `d` is an associate name and does not have the POINTER or ALLOCATABLE attribute of its selector
  select type (d => c)
  type is (real)
    !ERROR: Argument `d` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
    !$acc enter data attach(d)
  end select

  ! Without an associate-name, the note is at the clause, not at the
  ! declaration of x.
  select type (x)
  type is (real)
    !ERROR: Argument `x` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
    !BECAUSE: `x` is an associate name and does not have the POINTER or ALLOCATABLE attribute of its selector
    !$acc enter data attach(x)
  end select
end subroutine

subroutine test_attach_select_type_host(x)
  class(*), pointer :: x(:)
contains
  subroutine inner()
    select type (x)
    type is (real)
      !ERROR: Argument `x` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
      !BECAUSE: `x` is an associate name and does not have the POINTER or ALLOCATABLE attribute of its selector
      !$acc enter data attach(x)
    end select
  end subroutine
end subroutine

! The associate name has its selector's spelling, so the note must stay at the
! ASSOCIATE statement; and a section of a pointer selector gets the note too.
subroutine test_attach_same_name(self, p)
  use attach_associate_m
  type(t) :: self
  real, pointer :: p(:)

  !BECAUSE: `p` is an associate name and does not have the POINTER or ALLOCATABLE attribute of its selector
  associate (p => p)
    !ERROR: Argument `p` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
    !$acc enter data attach(p)
  end associate

  !BECAUSE: `s` is an associate name and does not have the POINTER or ALLOCATABLE attribute of its selector
  associate (s => self%p(1:2))
    !ERROR: Argument `s` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
    !$acc enter data attach(s)
  end associate
  !$acc enter data attach(self%p(1:2))
end subroutine

! A SELECT RANK associate name keeps the POINTER attribute of its selector.
subroutine test_attach_select_rank(x)
  real, pointer :: x(..)
  select rank (y => x)
  rank (1)
    !$acc enter data attach(y)
  end select
end subroutine
