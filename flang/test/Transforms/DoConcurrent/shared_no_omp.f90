! Tests that the fir.local {type = shared} op is created during lowering
! even when -fdo-concurrent-to-openmp is NOT specified, and is properly
! removed by the time we reach LLVM IR.
!
! When the OpenMP conversion path is disabled, the do concurrent loop goes
! through SimplifyFIROperations (converted to fir.do_loop). The SHARED
! variable is not in local_vars, so SimplifyFIROperations does not encounter
! it. The fir.local {type = shared} op remains unreferenced at module level
! and is erased during FIR-to-LLVM CodeGen by DoConcurrentSpecifierOpConversion.

! Check 1: At the FIR level, fir.local {type = shared} should exist.
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s --check-prefix=FIR

! Check 2: At the LLVM IR level, fir.local should be gone.
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

subroutine test_shared_no_omp(a, n)
  implicit none
  integer, intent(in) :: n
  real, intent(inout) :: a(n)
  real :: shared_val
  integer :: i

  shared_val = 42.0

  do concurrent (i = 1:n) shared(shared_val)
    a(i) = shared_val + real(i)
  end do
end subroutine

! FIR: fir.local {type = shared} @{{.*}}shared_val{{.*}} : f32

! LLVM-NOT: fir.local
! LLVM-NOT: fir.local {type = shared}
