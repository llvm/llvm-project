!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s
!RUN: bbc -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine sample()
! CHECK: omp.barrier
!$omp barrier
end subroutine sample
