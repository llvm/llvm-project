!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!REQUIRES: amdgpu-registered-target
!RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -fopenmp -nogpulib -fopenmp-is-target-device -mmlir --mlir-print-ir-before-all -S %s -o /dev/null 2>&1 | FileCheck %s


! This tests the fix for https://github.com/llvm/llvm-project/issues/209123
! We are only interested in ensuring that functions used in target regions should have
! the omp.declare_target attribute so the -omp-host-op-filter pass doesn't crash.

! CHECK-LABEL: IR Dump Before HostOpFilteringPass: omp-host-op-filter
! CHECK: llvm.func{{.*}}@__mlir_math_ipowi_i32
! CHECK-SAME: attributes{{.*}}omp.declare_target{{.*}}device_type =
! CHECK-NOT: (host)

module m
contains
    subroutine s()
      integer :: n1
      real :: tmp1
      !$omp declare target
      tmp1 = 2**n1
    end subroutine s
end module
