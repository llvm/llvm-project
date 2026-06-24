! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! Check that the `task` reduction modifier is lowered to the `task`
! reduction modifier attribute on the parallel and worksharing constructs.

! CHECK-LABEL: func.func @_QPreduction_task_parallel
subroutine reduction_task_parallel()
  integer :: i
  i = 0
  ! CHECK: omp.parallel reduction(mod: task, @{{.*}} %{{.*}} -> %{{.*}} : !fir.ref<i32>) {
  !$omp parallel reduction(task, +:i)
  i = i + 1
  !$omp end parallel
end subroutine reduction_task_parallel

! CHECK-LABEL: func.func @_QPreduction_task_do
subroutine reduction_task_do()
  integer :: i, j
  i = 0
  ! CHECK: omp.wsloop {{.*}}reduction(mod: task, @{{.*}} %{{.*}} -> %{{.*}} : !fir.ref<i32>) {
  !$omp do reduction(task, +:i)
  do j = 1, 10
    i = i + 1
  end do
  !$omp end do
end subroutine reduction_task_do

! CHECK-LABEL: func.func @_QPreduction_task_sections
subroutine reduction_task_sections()
  integer :: i
  i = 0
  ! CHECK: omp.sections {{.*}}reduction(mod: task, @{{.*}} %{{.*}} -> %{{.*}} : !fir.ref<i32>) {
  !$omp sections reduction(task, +:i)
  i = i + 1
  !$omp end sections
end subroutine reduction_task_sections
