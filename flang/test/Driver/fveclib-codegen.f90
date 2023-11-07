! test that -fveclib= is passed to the backend
! -target aarch64 so that ArmPL is available
! RUN: %flang -S -target aarch64-unknown-linux-gnu -mcpu=neoverse-v1 -Ofast -fveclib=ArmPL -o - %s | FileCheck %s

subroutine sb(a, b)
  real :: a(:), b(:)
  integer :: i
  do i=1,100
! check that we used a vectorized call to powf()
! CHECK: armpl_svpow_f32_x
    a(i) = a(i) ** b(i)
  end do
end subroutine
