!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-llvm-bc -fopenmp -o %t.bc %s 2>&1
!RUN: %flang_fc1 -emit-mlir -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.bc -o - %s 2>&1 | FileCheck %s

!CHECK: module attributes {{{.*}}, omp.host_ir_filepath = "{{.*}}.bc", omp.is_gpu = false, omp.is_target_device = true{{.*}}}
subroutine omp_subroutine()
end subroutine omp_subroutine
