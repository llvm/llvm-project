! RUN: bbc -fopenmp -emit-fir -o - %s | FileCheck %s

subroutine sample()
! CHECK: omp.barrier
!$omp barrier
end subroutine sample
