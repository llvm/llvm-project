! RUN: %python %S/test_errors.py %s %flang_fc1
subroutine forall1
  real :: a(9)
  !ERROR: 'i' is already declared in this scoping unit
  !ERROR: Cannot redefine FORALL variable 'i'
  forall (i=1:8, i=1:9)  a(i) = i
  !ERROR: 'i' is already declared in this scoping unit
  !ERROR: Cannot redefine FORALL variable 'i'
  forall (i=1:8, i=1:9)
    a(i) = i
  end forall
  forall (j=1:8)
    !ERROR: 'j' is already declared in this scoping unit
    !ERROR: Cannot redefine FORALL variable 'j'
    forall (j=1:9)
    end forall
  end forall
end

subroutine forall2
  integer, pointer :: a(:)
  integer, target :: b(10,10)
  forall (i=1:10)
    !ERROR: Impure procedure 'f_impure' may not be referenced in a FORALL
    a(f_impure(i):) => b(i,:)
  end forall
  !ERROR: FORALL mask expression may not reference impure procedure 'f_impure'
  forall (j=1:10, f_impure(1)>2)
  end forall
contains
  impure integer function f_impure(i)
    f_impure = i
  end
end

subroutine forall3
  real :: x
  forall(i=1:10)
    !ERROR: Cannot redefine FORALL variable 'i'
    i = 1
  end forall
  forall(i=1:10)
    forall(j=1:10)
      !ERROR: Cannot redefine FORALL variable 'i'
      !WARNING: FORALL index variable 'j' not used on left-hand side of assignment
      i = 1
    end forall
  end forall
  !ERROR: Cannot redefine FORALL variable 'i'
  forall(i=1:10) i = 1
end

subroutine forall4
  integer, parameter :: zero = 0
  integer :: a(10)

  !ERROR: FORALL limit expression may not reference index variable 'i'
  forall(i=1:i)
    a(i) = i
  end forall
  !ERROR: FORALL step expression may not reference index variable 'i'
  forall(i=1:10:i)
    a(i) = i
  end forall
  !ERROR: FORALL step expression may not be zero
  forall(i=1:10:zero)
    a(i) = i
  end forall

  !ERROR: FORALL limit expression may not reference index variable 'i'
  forall(i=1:i) a(i) = i
  !ERROR: FORALL step expression may not reference index variable 'i'
  forall(i=1:10:i) a(i) = i
  !ERROR: FORALL step expression may not be zero
  forall(i=1:10:zero) a(i) = i
end

! Note: this gets warnings but not errors
subroutine forall5
  real, target :: x(10), y(10)
  forall(i=1:10)
    x(i) = y(i)
  end forall
  forall(i=1:10)
    !WARNING: FORALL index variable 'i' not used on left-hand side of assignment
    x = y
    forall(j=1:10)
      !WARNING: FORALL index variable 'j' not used on left-hand side of assignment
      x(i) = y(i)
      !WARNING: FORALL index variable 'i' not used on left-hand side of assignment
      x(j) = y(j)
    endforall
  endforall
  do concurrent(i=1:10)
    x = y
    !WARNING: FORALL index variable 'i' not used on left-hand side of assignment
    forall(i=1:10) x = y
  end do
end

subroutine forall6
  type t
    real, pointer :: p
  end type
  type(t) :: a(10)
  real, target :: b(10)
  forall(i=1:10)
    a(i)%p => b(i)
    !WARNING: FORALL index variable 'i' not used on left-hand side of assignment
    a(1)%p => b(i)
  end forall
end

subroutine forall7(x)
  integer :: iarr(1)
  real :: a(10)
  class(*) :: x
  associate (j => iarr(1))
    forall (j=1:size(a))
      a(j) = a(j) + 1
    end forall
  end associate
  associate (j => iarr(1) + 1)
    forall (j=1:size(a))
      a(j) = a(j) + 1
    end forall
  end associate
  select type (j => x)
  type is (integer)
    forall (j=1:size(a))
      a(j) = a(j) + 1
    end forall
  end select
end subroutine
