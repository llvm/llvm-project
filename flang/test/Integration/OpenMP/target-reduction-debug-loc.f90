!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! The device reduction helpers emitted by OpenMPIRBuilder have no debug info of
! their own. If the builder's current debug location is left set while they are
! emitted, their instructions carry locations scoped to the enclosing kernel's
! subprogram. That is latent until the inliner folds a helper into a function
! that does have a subprogram, at which point the module fails verification with
! "!dbg attachment points at wrong subprogram for function".

! REQUIRES: amdgpu-registered-target

! RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-is-target-device \
! RUN:   -triple amdgcn-amd-amdhsa -debug-info-kind=standalone -O0 \
! RUN:   -o - %s | FileCheck %s

subroutine k(a, n, s)
  real(8), intent(in)  :: a(*)
  integer, intent(in)  :: n
  real(8), intent(out) :: s
  integer :: i
  s = 0
  !$omp target teams distribute parallel do reduction(+:s)
  do i = 1, n
     s = s + a(i)
  end do
end subroutine

! The helpers must be emitted without debug locations, so no !dbg appears
! between their definition and the closing brace.

! CHECK-LABEL: define internal void @_omp_reduction_shuffle_and_reduce_func
! CHECK-NOT:     !dbg
! CHECK:       }

! CHECK-LABEL: define internal void @_omp_reduction_inter_warp_copy_func
! CHECK-NOT:     !dbg
! CHECK:       }
