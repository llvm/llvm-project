! RUN: %flang_fc1 -emit-fir -fopenmp -o - %s | FileCheck %s
! RUN: bbc -emit-fir -fopenmp -o - %s | FileCheck %s

! CHECK:       omp.teams
! CHECK-SAME:  reduction(@add_reduction_i32 %{{.*}} -> %{{.*}} : !fir.ref<i32>)
subroutine myfun()
  integer :: i, j
  i = 0
  j = 0
  !$omp target teams distribute parallel do reduction(+:i)
  do j = 1,5
     i = i + j
  end do
  !$omp end target teams distribute parallel do
end subroutine myfun
