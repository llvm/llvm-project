! test that -fveclib= is passed to the backend
! RUN: %if aarch64-registered-target %{ %flang -S -Ofast -target aarch64-unknown-linux-gnu -fveclib=LIBMVEC -o - %s | FileCheck %s %}
! RUN: %if x86-registered-target %{ %flang -S -Ofast -target x86_64-unknown-linux-gnu -fveclib=LIBMVEC -o - %s | FileCheck %s %}
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
