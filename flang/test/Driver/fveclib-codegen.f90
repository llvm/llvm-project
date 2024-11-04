! test that -fveclib= is passed to the backend
! -target aarch64 so that ArmPL is available
! RUN: %flang -S -Ofast -fveclib=LIBMVEC -o - %s | FileCheck %s
! RUN: %flang -S -Ofast -fveclib=NoLibrary -o - %s | FileCheck %s --check-prefix=NOLIB

subroutine sb(a, b)
  real :: a(:), b(:)
  integer :: i
  do i=1,100
! check that we used a vectorized call to powf()
! CHECK: _ZGVbN4vv_powf
! NOLIB: powf
    a(i) = a(i) ** b(i)
  end do
end subroutine
