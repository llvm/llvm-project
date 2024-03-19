! Check that for parallel do, reduction is only processed for the loop

! RUN: bbc -fopenmp --force-byref-reduction -emit-hlfir %s -o - | FileCheck %s
! RUN: flang-new -fc1 -fopenmp -mmlir --force-byref-reduction -emit-hlfir %s -o - | FileCheck %s

! CHECK: omp.parallel {
! CHECK: omp.wsloop byref reduction(@add_reduction_i_32
subroutine sb
  integer :: x
  x = 0
  !$omp parallel do reduction(+:x)
  do i=1,100
    x = x + 1
  end do
  !$omp end parallel do
end subroutine
