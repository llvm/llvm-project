! Test COPYPRIVATE.
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK-DAG: func private @_copy_f32(%{{.*}}: !fir.ref<f32>, %{{.*}}: !fir.ref<f32>)
!CHECK-DAG: func private @_copy_10xi32(%{{.*}}: !fir.ref<!fir.array<10xi32>>, %{{.*}}: !fir.ref<!fir.array<10xi32>>)

!CHECK-LABEL: func private @_copy_i32(
!CHECK-SAME:                  %[[ARG0:.*]]: !fir.ref<i32>, %[[ARG1:.*]]: !fir.ref<i32>) {
!CHECK-NEXT:    %[[DST:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_copy_i32_dst"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:    %[[SRC:.*]]:2 = hlfir.declare %[[ARG1]] {uniq_name = "_copy_i32_src"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:    %[[SRC_VAL:.*]] = fir.load %[[SRC]]#0 : !fir.ref<i32>
!CHECK-NEXT:    hlfir.assign %[[SRC_VAL]] to %[[DST]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK-NEXT:    return
!CHECK-NEXT:  }

!CHECK-LABEL: func @_QPtest_scalar
!CHECK:         omp.parallel
!CHECK:           %[[I:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_scalarEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[J:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_scalarEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[K:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_scalarEk"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:           omp.single copyprivate(%[[I]]#0 -> @_copy_i32 : !fir.ref<i32>, %[[J]]#0 -> @_copy_i32 : !fir.ref<i32>, %[[K]]#0 -> @_copy_f32 : !fir.ref<f32>)
subroutine test_scalar()
  integer, save :: i, j
  !$omp threadprivate(i, j)
  real :: k

  k = 33.3
  !$omp parallel firstprivate(k)
  !$omp single
  i = 11
  j = 22
  !$omp end single copyprivate(i, j, k)
  !$omp end parallel
end subroutine

!CHECK-LABEL: func @_QPtest_array
!CHECK:         omp.parallel
!CHECK:           %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtest_arrayEa"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
!CHECK:           omp.single copyprivate(%[[A]]#0 -> @_copy_10xi32 : !fir.ref<!fir.array<10xi32>>)
subroutine test_array()
  integer :: a(10)

  !$omp parallel private(a)
  !$omp single
  a = 100
  !$omp end single copyprivate(a)
  !$omp end parallel
end subroutine
