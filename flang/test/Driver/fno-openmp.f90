! RUN: %flang_fc1 -fopenmp -fno-openmp %s -emit-hlfir -o - | FileCheck --check-prefix=CHECK-NO-OMP %s
! RUN: %flang_fc1 -fno-openmp %s -emit-hlfir -o - | FileCheck --check-prefix=CHECK-NO-OMP %s
! RUN: %flang_fc1 -fno-openmp -fopenmp %s -emit-hlfir -o - | FileCheck --check-prefix=CHECK-OMP %s
! RUN: %flang_fc1 -fopenmp %s -emit-hlfir -o - | FileCheck --check-prefix=CHECK-OMP %s

subroutine main
  ! CHECK-NO-OMP-NOT: omp.parallel
  ! CHECK-OMP: omp.parallel
  !$omp parallel
  print *,"test"
  !$omp end parallel
end subroutine
