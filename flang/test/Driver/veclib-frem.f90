! RUN: %flang %s -O3 -ffast-math -fveclib=ArmPL --target=aarch64 -mcpu=neoverse-v1 -S -o - | FileCheck %s
! REQUIRES: aarch64-registered-target

! This test checks that veclib works from flang through the pipeline:
!  The modulo is turned into a loop with double frem.
!  The frem is vectorized in the loop vectorizer using fveclib=ArmPL to a
!    <vscale x 2 x double> frem.
!  The <vscale x 2 x double> frem is converted to a call to armpl_svfmod_f64_x
!    in the backend.

! CHECK-LABEL: frem_kernel_
! CHECK: bl armpl_svfmod_f64_x

  subroutine frem_kernel(a, b, c, n)
    integer, intent(in) :: n
    real(8), intent(in)  :: a(n), b(n)
    real(8), intent(out) :: c(n)
    integer :: i

    do i = 1, n
       c(i) = modulo(a(i), b(i))
    end do
  end subroutine frem_kernel