! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s --implicit-check-not=add_reduction_byref_box

subroutine reduction_literal(a, n)
  integer :: a(4), n
!$omp parallel do reduction(+: a(2))
  do i = 1, n
    a(2) = a(2) + i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPreduction_literal
! CHECK: omp.wsloop {{.*}} reduction(@add_reduction_i32 {{.*}} : !fir.ref<i32>) {
! CHECK: hlfir.declare %arg{{[0-9]+}} {uniq_name = "omp.reduction.element"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: fir.load %{{[0-9]+}}#0 : !fir.ref<i32>
! CHECK: hlfir.assign {{.*}} to %{{[0-9]+}}#0 : i32, !fir.ref<i32>

subroutine reduction_multiple(a, n)
  integer :: a(4), n
!$omp parallel do reduction(+: a(2), a(3))
  do i = 1, n
    a(2) = a(2) + i
    a(3) = a(3) + i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPreduction_multiple
! CHECK: omp.wsloop {{.*}} reduction(@add_reduction_i32 {{.*}}, @add_reduction_i32 {{.*}} : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK: hlfir.declare %arg{{[0-9]+}} {uniq_name = "omp.reduction.element"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: hlfir.declare %arg{{[0-9]+}} {uniq_name = "omp.reduction.element"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: hlfir.assign {{.*}} to %{{[0-9]+}}#0 : i32, !fir.ref<i32>
! CHECK: hlfir.assign {{.*}} to %{{[0-9]+}}#0 : i32, !fir.ref<i32>

subroutine reduction_arrays(a, b, n)
  integer :: a(4), b(4), n
!$omp parallel do reduction(+: a(2), b(2))
  do i = 1, n
    a(2) = a(2) + b(2) + i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPreduction_arrays
! CHECK: omp.wsloop {{.*}} reduction(@add_reduction_i32 {{.*}}, @add_reduction_i32 {{.*}} : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK: hlfir.declare %arg{{[0-9]+}} {uniq_name = "omp.reduction.element"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: hlfir.declare %arg{{[0-9]+}} {uniq_name = "omp.reduction.element"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

subroutine reduction_variable(a, n, j)
  integer :: a(4), n, j
!$omp parallel do reduction(+: a(j))
  do i = 1, n
    a(j) = a(j) + i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPreduction_variable
! CHECK: omp.wsloop {{.*}} reduction(@add_reduction_i32 {{.*}} : !fir.ref<i32>) {
! CHECK: hlfir.declare %arg{{[0-9]+}} {uniq_name = "omp.reduction.element"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: hlfir.assign {{.*}} to %{{[0-9]+}}#0 : i32, !fir.ref<i32>

subroutine reduction_do_simd(a, n)
  integer :: a(4), n
!$omp parallel do simd reduction(+: a(2))
  do i = 1, n
    a(2) = a(2) + i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPreduction_do_simd
! CHECK: omp.wsloop reduction(@add_reduction_i32 {{.*}} -> [[WSARG:%arg[0-9]+]] : !fir.ref<i32>) {
! CHECK: omp.simd {{.*}} reduction(@add_reduction_i32 [[WSARG]] -> [[SIMDARG:%arg[0-9]+]] : !fir.ref<i32>) {
! CHECK: hlfir.declare [[SIMDARG]] {uniq_name = "omp.reduction.element"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: hlfir.assign {{.*}} to %{{[0-9]+}}#0 : i32, !fir.ref<i32>

subroutine task_reduction_element(a)
  integer :: a(4)
!$omp taskgroup task_reduction(+: a(2))
!$omp task in_reduction(+: a(2))
  a(2) = a(2) + 1
!$omp end task
!$omp end taskgroup
end subroutine

! CHECK-LABEL: func.func @_QPtask_reduction_element
! CHECK-NOT: _QFtask_reduction_elementEa_firstprivate_box_4xi32
! CHECK: omp.taskgroup task_reduction(@add_reduction_i32 {{.*}} -> [[TGARG:%arg[0-9]+]] : !fir.ref<i32>) {
! CHECK: [[TGDECL:%[0-9]+]]:2 = hlfir.declare [[TGARG]] {uniq_name = "omp.reduction.element"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: omp.task in_reduction(@add_reduction_i32 [[TGDECL]]#0 -> [[TASKARG:%arg[0-9]+]] : !fir.ref<i32>)
! CHECK: [[TASKDECL:%[0-9]+]]:2 = hlfir.declare [[TASKARG]] {uniq_name = "omp.reduction.element"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: fir.load [[TASKDECL]]#0 : !fir.ref<i32>
! CHECK: hlfir.assign {{.*}} to [[TASKDECL]]#0 : i32, !fir.ref<i32>

subroutine taskloop_in_reduction_element(a, n)
  integer :: a(4), n
!$omp taskloop in_reduction(+: a(2))
  do i = 1, n
    a(2) = a(2) + i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtaskloop_in_reduction_element
! CHECK-NOT: _QFtaskloop_in_reduction_elementEa_firstprivate_box_4xi32
! CHECK: omp.taskloop.context in_reduction(@add_reduction_i32 {{.*}} -> [[TLARG:%arg[0-9]+]] : !fir.ref<i32>)
! CHECK: hlfir.declare [[TLARG]] {uniq_name = "omp.reduction.element"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: hlfir.assign {{.*}} to %{{[0-9]+}}#0 : i32, !fir.ref<i32>

subroutine taskloop_reduction_element(a, n)
  integer :: a(4), n
!$omp taskloop reduction(+: a(2))
  do i = 1, n
    a(2) = a(2) + i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtaskloop_reduction_element
! CHECK-NOT: _QFtaskloop_reduction_elementEa_firstprivate_box_4xi32
! CHECK: omp.taskloop.context {{.*}} reduction(@add_reduction_i32 {{.*}} -> [[TLRARG:%arg[0-9]+]] : !fir.ref<i32>)
! CHECK: hlfir.declare [[TLRARG]] {uniq_name = "omp.reduction.element"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: hlfir.assign {{.*}} to %{{[0-9]+}}#0 : i32, !fir.ref<i32>

subroutine taskloop_reduction_mixed_use(a, n)
  integer :: a(4), n
!$omp taskloop reduction(+: a(2))
  do i = 1, n
    a(2) = a(2) + i
    a(1) = a(1) + 1
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtaskloop_reduction_mixed_use
! CHECK: omp.taskloop.context private({{.*}}@_QFtaskloop_reduction_mixed_useEa_firstprivate_box_4xi32{{.*}}) reduction(@add_reduction_i32 {{.*}} -> [[TLMARG:%arg[0-9]+]] : !fir.ref<i32>)
! CHECK: hlfir.declare [[TLMARG]] {uniq_name = "omp.reduction.element"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

subroutine taskloop_reduction_nested_index_use(a, b, n)
  integer :: a(4), b(4), n
!$omp taskloop reduction(+: a(2))
  do i = 1, n
    a(2) = a(2) + i
    b(a(1)) = b(a(1)) + 1
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtaskloop_reduction_nested_index_use
! CHECK: omp.taskloop.context private({{.*}}@_QFtaskloop_reduction_nested_index_useEa_firstprivate_box_4xi32{{.*}}) reduction(@add_reduction_i32 {{.*}} -> [[TLNARG:%arg[0-9]+]] : !fir.ref<i32>)
! CHECK: hlfir.declare [[TLNARG]] {uniq_name = "omp.reduction.element"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
