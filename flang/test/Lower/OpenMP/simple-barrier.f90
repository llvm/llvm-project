!RUN: %flang_fc1 -flang-experimental-hlfir -emit-hlfir -fopenmp %s -o - | FileCheck %s
!RUN: bbc -hlfir -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine sample()
! CHECK: omp.barrier
!$omp barrier
end subroutine sample
