! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s | FileCheck %s

! scalar element locator
subroutine omp_task_affinity_elem()
  implicit none
  integer, parameter :: n = 100
  integer :: a(n)

  !$omp parallel
  !$omp task affinity(a(1))
    a(1) = 1
  !$omp end task
  !$omp end parallel
end subroutine omp_task_affinity_elem

! CHECK-LABEL: func.func @_QPomp_task_affinity_elem()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFomp_task_affinity_elemEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK: omp.parallel {
! CHECK:     %[[C1:.*]] = arith.constant 1 : index
! CHECK:     %[[ELEM:.*]] = hlfir.designate %[[A]]#0 (%[[C1]]) : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:     %[[C1_0:.*]] = arith.constant 1 : index
! CHECK:     %[[C0:.*]] = arith.constant 0 : index
! CHECK:     %[[SUB:.*]] = arith.subi %[[C0]], %[[C0]] : index
! CHECK:     %[[DIV:.*]] = arith.divui %[[SUB]], %[[C1_0]] : index
! CHECK:     %[[C1_1:.*]] = arith.constant 1 : index
! CHECK:     %[[ADD:.*]] = arith.addi %[[DIV]], %[[C1_1]] : index
! CHECK:     %[[CAST:.*]] = arith.index_cast %[[ADD]] : index to i64
! CHECK:     %[[C4:.*]] = arith.constant 4 : i64
! CHECK:     %[[LEN:.*]] = arith.muli %[[CAST]], %[[C4]] : i64
! CHECK:     %[[ADDRI8:.*]] = fir.convert %[[ELEM]] : (!fir.ref<i32>) -> !fir.ref<i8>
! CHECK:     %[[ENTRY:.*]] = omp.affinity_entry %[[ADDRI8]], %[[LEN]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK:     omp.task affinity(%[[ENTRY]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>) {

subroutine omp_task_affinity_array_section()
  implicit none
  integer, parameter :: n = 100
  integer :: a(n)
  integer :: i

  !$omp parallel
  !$omp task affinity(a(2:50)) private(i)
    do i = 2, 50
      a(i) = i
    end do
  !$omp end task
  !$omp end parallel
end subroutine omp_task_affinity_array_section

! CHECK-LABEL: func.func @_QPomp_task_affinity_array_section()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFomp_task_affinity_array_sectionEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK: %[[I:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFomp_task_affinity_array_sectionEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: omp.parallel {
! CHECK:     %[[C2:.*]] = arith.constant 2 : index
! CHECK:     %[[C50:.*]] = arith.constant 50 : index
! CHECK:     %[[C1:.*]] = arith.constant 1 : index
! CHECK:     %[[C49:.*]] = arith.constant 49 : index
! CHECK:     %[[SHAPE:.*]] = fir.shape %[[C49]] : (index) -> !fir.shape<1>
! CHECK:     %[[SLICE:.*]] = hlfir.designate %[[A]]#0 (%[[C2]]:%[[C50]]:%[[C1]])  shape %[[SHAPE]] : (!fir.ref<!fir.array<100xi32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<49xi32>>
! CHECK:     %[[C1_0:.*]] = arith.constant 1 : index
! CHECK:     %[[C1_1:.*]] = arith.constant 1 : index
! CHECK:     %[[C49_2:.*]] = arith.constant 49 : index
! CHECK:     %[[SUB:.*]] = arith.subi %[[C49_2]], %[[C1_1]] : index
! CHECK:     %[[DIV:.*]] = arith.divui %[[SUB]], %[[C1_0]] : index
! CHECK:     %[[C1_3:.*]] = arith.constant 1 : index
! CHECK:     %[[ADD:.*]] = arith.addi %[[DIV]], %[[C1_3]] : index
! CHECK:     %[[CAST:.*]] = arith.index_cast %[[ADD]] : index to i64
! CHECK:     %[[C4:.*]] = arith.constant 4 : i64
! CHECK:     %[[LEN:.*]] = arith.muli %[[CAST]], %[[C4]] : i64
! CHECK:     %[[ADDRI8:.*]] = fir.convert %[[SLICE]] : (!fir.ref<!fir.array<49xi32>>) -> !fir.ref<i8>
! CHECK:     %[[ENTRY:.*]] = omp.affinity_entry %[[ADDRI8]], %[[LEN]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK:     omp.task affinity(%[[ENTRY]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>) private(@_QFomp_task_affinity_array_sectionEi_private_i32 %[[I]]#0 -> %[[P:.*]] : !fir.ref<i32>) {

subroutine omp_task_affinity_scalar()
  implicit none
  integer :: s
  s = 7

  !$omp parallel
  !$omp task affinity(s)
    s = s + 1
  !$omp end task
  !$omp end parallel
end subroutine omp_task_affinity_scalar

! CHECK-LABEL: func.func @_QPomp_task_affinity_scalar()
! CHECK: %[[S:.*]] = fir.alloca i32 {bindc_name = "s", uniq_name = "_QFomp_task_affinity_scalarEs"}
! CHECK: %[[SDECL:.*]]:2 = hlfir.declare %[[S]] {uniq_name = "_QFomp_task_affinity_scalarEs"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: hlfir.assign %{{.*}} to %[[SDECL]]#0 : i32, !fir.ref<i32>
! CHECK: omp.parallel {
! CHECK:     %[[LEN:.*]] = arith.constant 4 : i64
! CHECK:     %[[ADDRI8:.*]] = fir.convert %[[SDECL]]#0 : (!fir.ref<i32>) -> !fir.ref<i8>
! CHECK:     %[[ENTRY:.*]] = omp.affinity_entry %[[ADDRI8]], %[[LEN]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK:     omp.task affinity(%[[ENTRY]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>) {

subroutine omp_task_affinity_multi()
  implicit none
  integer, parameter :: n = 100
  integer :: a(n), b(n)

  !$omp parallel
  !$omp task affinity(a(1), b(1))
    a(2) = 2
    b(2) = 2
  !$omp end task
  !$omp end parallel
end subroutine omp_task_affinity_multi

! CHECK-LABEL: func.func @_QPomp_task_affinity_multi()
! CHECK: omp.parallel {
! CHECK:     %[[AADDR:.*]] = fir.convert %{{.*}} : (!fir.ref<i32>) -> !fir.ref<i8>
! CHECK:     %[[AENT:.*]] = omp.affinity_entry %[[AADDR]], %{{.*}} : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK:     %[[BADDR:.*]] = fir.convert %{{.*}} : (!fir.ref<i32>) -> !fir.ref<i8>
! CHECK:     %[[BENT:.*]] = omp.affinity_entry %[[BADDR]], %{{.*}} : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK:     omp.task affinity(%[[AENT]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>, %[[BENT]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>) {

subroutine task_affinity_iterator_simple()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp parallel
  !$omp single
  !$omp task affinity(iterator(i = 1:n) : a(i))
    a(i) = i
  !$omp end task
  !$omp end single
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtask_affinity_iterator_simple()
! CHECK:     %[[ITERATED:.*]] = omp.iterators(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:       {{.*}} = arith.index_cast %[[IV]] : index to i32
! CHECK:       {{.*}} = fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<16xi32>>, i32) -> !fir.ref<i32>
! CHECK:       {{.*}} = fir.convert {{.*}} : (!fir.ref<i32>) -> !fir.ref<i8>
! CHECK:       %[[ENTRY:.*]] = omp.affinity_entry {{.*}}, {{.*}} : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK:       omp.yield(%[[ENTRY]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>)
! CHECK:     } -> !omp.iterated<!omp.affinity_entry_ty<!fir.ref<i8>, i64>>
! CHECK:     omp.task affinity(%{{.*}} : !omp.iterated<!omp.affinity_entry_ty<!fir.ref<i8>, i64>>) {

subroutine task_affinity_iterator_multi_dimension()
  integer, parameter :: n = 4, m = 6
  integer :: a(n, m)
  integer :: i, j

  !$omp parallel
  !$omp single
  !$omp task affinity(iterator(i = 3:n, j = 1:m) : a(i, j)) shared(a)
    do i = 1, n
      do j = 1, m
        a(i, j) = 100*i + j
      end do
    end do
  !$omp end task
  !$omp end single
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtask_affinity_iterator_multi_dimension()
! CHECK: %[[ITER:.*]] = omp.iterators(%[[IV0:.*]]: index, %[[IV1:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}, {{.*}} to {{.*}} step {{.*}}) {
! CHECK:   {{.*}} = arith.index_cast %[[IV0]] : index to i32
! CHECK:   {{.*}} = arith.index_cast %[[IV1]] : index to i32
! CHECK:   {{.*}} = fir.coordinate_of {{.*}}, {{.*}}, {{.*}} : (!fir.ref<!fir.array<4x6xi32>>, i32, i32) -> !fir.ref<i32>
! CHECK:   {{.*}} = fir.convert {{.*}} : (!fir.ref<i32>) -> !fir.ref<i8>
! CHECK:   %[[ENTRY:.*]] = omp.affinity_entry {{.*}}, {{.*}} : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK:   omp.yield(%[[ENTRY]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>)
! CHECK: } -> !omp.iterated<!omp.affinity_entry_ty<!fir.ref<i8>, i64>>
! CHECK: omp.task affinity(%[[ITER]] : !omp.iterated<!omp.affinity_entry_ty<!fir.ref<i8>, i64>>)
