! RUN: %python %S/test_errors.py %s %flang_fc1
! Test DATA statement pointer initialization with array pointer components
! in implied DO loops.

subroutine array_ptr
  type ty
    real, pointer :: ptr(:)
  end type
  real, target :: targ(10) = [(i, i=1,10)]
  type(ty) :: x(5)
  ! Array pointer component in implied DO - valid
  data (x(i)%ptr, i=1,5) / 5*targ /
end subroutine

subroutine array_ptr_2d
  type ty
    real, pointer :: ptr(:)
  end type
  real, target :: targ(10) = [(i, i=1,10)]
  type(ty) :: x(5,5)
  ! Array pointer component in implied DO - valid
  data ((x(i,j)%ptr, i=1,5),j=1,5) / 25*targ /
end subroutine

subroutine scalar_ptr
  type ty
    real, pointer :: ptr
  end type
  real, target :: targ = 1.0
  type(ty) :: x(5)
  ! Scalar pointer component in implied DO - valid
  data (x(i)%ptr, i=1,5) / 5*targ /
end subroutine

subroutine scalar_ptr_2d
  type ty
    real, pointer :: ptr
  end type
  real, target :: targ = 1.0
  type(ty) :: x(5,5)
  ! Scalar pointer component in implied DO - valid
  data ((x(i,j)%ptr, i=1,5),j=1,5) / 25*targ /
end subroutine

subroutine array_object
  type ty
    real :: arr(10)
  end type
  type(ty) :: x(5)
  !ERROR: Data implied do object must be scalar or a pointer
  data (x(i)%arr, i=1,5) / 50*0.0 /
end subroutine

subroutine array_object_2d
  type ty
    real :: arr(10)
  end type
  type(ty) :: x(5,5)
  !ERROR: Data implied do object must be scalar or a pointer
  data ((x(i,j)%arr, i=1,5),j=1,5) / 250*0.0 /
end subroutine
