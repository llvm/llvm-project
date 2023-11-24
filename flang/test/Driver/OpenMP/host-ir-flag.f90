!RUN: %flang_fc1 -emit-llvm-bc -fopenmp -o %t.bc %s 2>&1
!RUN: %flang_fc1 -emit-mlir -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.bc -o - %s 2>&1 | FileCheck %s

!CHECK: module attributes {{{.*}}, omp.host_ir_filepath = "{{.*}}.bc", omp.is_gpu = false, omp.is_target_device = true{{.*}}}
subroutine omp_subroutine()
end subroutine omp_subroutine
