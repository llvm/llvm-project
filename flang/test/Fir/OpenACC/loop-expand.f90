! RUN: bbc -fopenacc -emit-hlfir %s -o - | fir-opt --split-input-file --acc-loop-expand | FileCheck %s

subroutine singleloop(a)
   real :: a(:)
   integer :: i
   a = 0.0

   !$acc loop
   do i = 1, 10
     a(i) = i
  end do
end subroutine
! CHECK-LABEL: func.func @_QPsingleloop
! CHECK: %[[I:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFsingleloopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: acc.loop private
! CHECK:   %[[LB0:.*]] = fir.convert %c1_i32 : (i32) -> index
! CHECK:   %[[UB0:.*]] = fir.convert %c10_i32 : (i32) -> index
! CHECK:   %[[STEP0:.*]] = fir.convert %c1_i32_0 : (i32) -> index
! CHECK:   %{{.*}} = fir.do_loop %[[ARG1:.*]] = %[[LB0]] to %[[UB0]] step %[[STEP0]] iter_args(%[[ARG2:.*]] = %{{.*}}) -> (index, i32) {
! CHECK:     fir.store %[[ARG2]] to %2#1 : !fir.ref<i32>
! CHECK:     %[[INCR1:.*]] = arith.addi %[[ARG1]], %[[STEP0]] : index
! CHECK:     %[[CONV_STEP:.*]] = fir.convert %[[STEP0]] : (index) -> i32
! CHECK:     %[[LOAD_I:.*]] = fir.load %[[I]]#1 : !fir.ref<i32>
! CHECK:     %[[INCR2:.*]] = arith.addi %[[LOAD_I]], %[[CONV_STEP]] : i32
! CHECK:     fir.result %[[INCR1]], %[[INCR2]] : index, i32
! CHECK:   }
! CHECK:   acc.yield
! CHECK: }

subroutine single_loop_with_nest(a)
  real :: a(:,:)
  integer :: i, j
  a = 0.0

  !$acc loop
  do i = 1, 10
    do j = 1, 10
      a(i, j) = i
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPsingle_loop_with_nest
! CHECK: %[[I:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFsingle_loop_with_nestEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: acc.loop private
! CHECK:   %[[LB0:.*]] = fir.convert %c1_i32 : (i32) -> index
! CHECK:   %[[UB0:.*]] = fir.convert %c10_i32 : (i32) -> index
! CHECK:   %[[STEP0:.*]] = fir.convert %c1_i32_0 : (i32) -> index
! CHECK:   %{{.*}} = fir.do_loop %[[ARG1:.*]] = %[[LB0]] to %[[UB0]] step %[[STEP0]] iter_args(%[[ARG2:.*]] = %{{.*}}) -> (index, i32) {
! CHECK:     fir.store %[[ARG2]] to %2#1 : !fir.ref<i32>
! CHECK:     fir.do_loop
! CHECK:     }
! CHECK:     %[[INCR1:.*]] = arith.addi %[[ARG1]], %[[STEP0]] : index
! CHECK:     %[[CONV_STEP:.*]] = fir.convert %[[STEP0]] : (index) -> i32
! CHECK:     %[[LOAD_I:.*]] = fir.load %[[I]]#1 : !fir.ref<i32>
! CHECK:     %[[INCR2:.*]] = arith.addi %[[LOAD_I]], %[[CONV_STEP]] : i32
! CHECK:     fir.result %[[INCR1]], %[[INCR2]] : index, i32
! CHECK:   }
! CHECK:   acc.yield
! CHECK: }

subroutine loop_with_nest(a)
  real :: a(:,:)
  integer :: i, j
  a = 0.0

  !$acc loop collapse(2)
  do i = 1, 10
    do j = 1, 10
      a(i, j) = i
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPloop_with_nest
! CHECK: %[[I:.*]]:2 = hlfir.declare %1 {uniq_name = "_QFloop_with_nestEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[J:.*]]:2 = hlfir.declare %3 {uniq_name = "_QFloop_with_nestEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: acc.loop private(@privatization_ref_i32 -> %{{.*}} : !fir.ref<i32>, @privatization_ref_i32 -> %{{.*}} : !fir.ref<i32>) {
! CHECK:   %[[LB0:.*]] = fir.convert %{{.*}} : (i32) -> index
! CHECK:   %[[UB0:.*]] = fir.convert %{{.*}} : (i32) -> index
! CHECK:   %[[STEP0:.*]] = fir.convert %{{.*}} : (i32) -> index
! CHECK:   %{{.*}}:2 = fir.do_loop %[[ARG1:.*]] = %[[LB0]] to %[[UB0]] step %[[STEP0]] iter_args(%[[ARG2:.*]] = %{{.*}}) -> (index, i32) {
! CHECK:     fir.store %[[ARG2]] to %[[I]]#1 : !fir.ref<i32>
! CHECK:     %[[LB1:.*]] = fir.convert %{{.*}} : (i32) -> index
! CHECK:     %[[UB1:.*]] = fir.convert %{{.*}} : (i32) -> index
! CHECK:     %[[STEP1:.*]] = fir.convert %{{.*}} : (i32) -> index
! CHECK:     %{{.*}}:2 = fir.do_loop %[[ARG3:.*]] = %[[LB1]] to %[[UB1]] step %[[STEP1]] iter_args(%[[ARG4:.*]] = %{{.*}}) -> (index, i32) {
! CHECK:       fir.store %[[ARG4]] to %[[J]]#1 : !fir.ref<i32>

! CHECK:       %[[INCR1:.*]] = arith.addi %[[ARG3]], %[[STEP1]] : index
! CHECK:       %[[CONV_STEP1:.*]] = fir.convert %[[STEP1]] : (index) -> i32
! CHECK:       %[[LOAD_J:.*]] = fir.load %[[J]]#1 : !fir.ref<i32>
! CHECK:       %[[INCR2:.*]] = arith.addi %[[LOAD_J]], %[[CONV_STEP1]] : i32
! CHECK:       fir.result %[[INCR1]], %[[INCR2]] : index, i32
! CHECK:     }
! CHECK:     %[[INCR1:.*]] = arith.addi %[[ARG1]], %[[STEP0]] : index
! CHECK:     %[[CONV_STEP0:.*]] = fir.convert %[[STEP0]] : (index) -> i32
! CHECK:     %[[LOAD_I:.*]] = fir.load %[[I]]#1 : !fir.ref<i32>
! CHECK:     %[[INCR2:.*]] = arith.addi %[[LOAD_I]], %18 : i32
! CHECK:     fir.result %[[INCR1]], %[[INCR2]] : index, i32
! CHECK:   }
! CHECK:   acc.yield
! CHECK: }

subroutine loop_unstructured(a)
  real :: a(:)
  integer :: i
  a = 0.0

  !$acc loop
  do i = 1, 10
    if (a(i) > 0.0) stop 'stop'
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPloop_unstructured
! CHECK: acc.loop private(@privatization_ref_i32 -> %{{.*}} : !fir.ref<i32>)
