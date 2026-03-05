! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s | FileCheck %s

! scalar element locator
subroutine omp_task_affinity_elem()
  implicit none
  integer, parameter :: n = 100
  integer :: a(n)

  !$omp parallel
  !$omp single
  !$omp task affinity(a(1))
    a(1) = 1
  !$omp end task
  !$omp end single
  !$omp end parallel
end subroutine omp_task_affinity_elem

! CHECK-LABEL: func.func @_QPomp_task_affinity_elem()
! CHECK: %[[A1:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFomp_task_affinity_elemEa"}
! CHECK: omp.parallel {
! CHECK:   omp.single {
! CHECK:     omp.task affinity(%[[A1]]#0 : !fir.ref<!fir.array<100xi32>>) {
! CHECK:       omp.terminator
! CHECK:     }
! CHECK:   omp.terminator
! CHECK: }
! CHECK: return

! array section locator
subroutine omp_task_affinity_array_section()
  implicit none
  integer, parameter :: n = 100
  integer :: a(n)
  integer :: i

  !$omp parallel
  !$omp single
  !$omp task affinity(a(2:50)) private(i)
    do i = 2, 50
      a(i) = i
    end do
  !$omp end task
  !$omp end single
  !$omp end parallel
end subroutine omp_task_affinity_array_section

! CHECK-LABEL: func.func @_QPomp_task_affinity_array_section()
! CHECK: %[[A2:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFomp_task_affinity_array_sectionEa"}
! CHECK: %[[I2:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFomp_task_affinity_array_sectionEi"}
! CHECK: omp.parallel {
! CHECK:   omp.single {
! CHECK:     omp.task affinity(%[[A2]]#0 : !fir.ref<!fir.array<100xi32>>) private({{.*}} %[[I2]]#0 -> %{{.*}} : !fir.ref<i32>) {
! CHECK:       omp.terminator
! CHECK:     }
! CHECK:   omp.terminator
! CHECK: }
! CHECK: return

! scalar variable locator
subroutine omp_task_affinity_scalar()
  implicit none
  integer :: s
  s = 7

  !$omp parallel
  !$omp single
  !$omp task affinity(s)
    s = s + 1
  !$omp end task
  !$omp end single
  !$omp end parallel
end subroutine omp_task_affinity_scalar

! CHECK-LABEL: func.func @_QPomp_task_affinity_scalar()
! CHECK: %[[S3:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFomp_task_affinity_scalarEs"}
! CHECK: omp.parallel {
! CHECK:   omp.single {
! CHECK:     omp.task affinity(%[[S3]]#0 : !fir.ref<i32>) {
! CHECK:       omp.terminator
! CHECK:     }
! CHECK:   omp.terminator
! CHECK: }
! CHECK: return

! multiple locators
subroutine omp_task_affinity_multi()
  implicit none
  integer, parameter :: n = 100
  integer :: a(n), b(n)

  !$omp parallel
  !$omp single
  !$omp task affinity(a(1), b(1))
    a(2) = 2
    b(2) = 2
  !$omp end task
  !$omp end single
  !$omp end parallel
end subroutine omp_task_affinity_multi

! CHECK-LABEL: func.func @_QPomp_task_affinity_multi()
! CHECK: %[[A4:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFomp_task_affinity_multiEa"}
! CHECK: %[[B4:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFomp_task_affinity_multiEb"}
! CHECK: omp.parallel {
! CHECK:   omp.single {
! CHECK:     omp.task affinity(%[[A4]]#0 : !fir.ref<!fir.array<100xi32>>, %[[B4]]#0 : !fir.ref<!fir.array<100xi32>>) {
! CHECK:       omp.terminator
! CHECK:     }
! CHECK:   omp.terminator
! CHECK: }
! CHECK: return
