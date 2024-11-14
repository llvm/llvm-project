! RUN: %flang_fc1 -emit-hlfir -o - -fopenmp %s | FileCheck %s
! RUN: bbc -emit-hlfir -o - -fopenmp %s | FileCheck %s

program lastprivate_allocatable
  integer, allocatable :: a
  integer :: i
  ! a is unallocated here
  !$omp parallel do lastprivate(a)
  do i=1,1
    a = 42
  enddo
  !$omp end parallel do
  ! a should be allocated here
end program

! CHECK-LABEL:   func.func @_QQmain()
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "a", uniq_name = "_QFEa"}
! CHECK:           %[[VAL_1:.*]] = fir.zero_bits !fir.heap<i32>
! CHECK:           %[[VAL_2:.*]] = fir.embox %[[VAL_1]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK:           fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
! CHECK:           omp.parallel {
!                    create original copy of private variable
! CHECK:             %[[VAL_16:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
! CHECK:             %[[VAL_17:.*]] = fir.alloca i32 {bindc_name = "i", pinned, uniq_name = "_QFEi"}
! CHECK:             %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_17]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             omp.wsloop {
! CHECK:               omp.loop_nest
!                        [...]
!                        if this is the last iteration
! CHECK:                 fir.if %{{.*}} {
!                          store loop IV
! CHECK:                   fir.store %{{.*}} to %[[VAL_18]]#1 : !fir.ref<i32>
!                          assign private variable to original copy: realloc
! CHECK:                   hlfir.assign %[[VAL_16]]#0 to %[[VAL_3]]#0 realloc : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK-NEXT:            }
! CHECK-NEXT:            omp.yield
! CHECK-NEXT:          }
