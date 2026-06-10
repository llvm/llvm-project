! RUN: %flang_fc1 -emit-hlfir -mmlir --enable-delayed-privatization-staging=true -o - %s | FileCheck %s

subroutine do_concurrent_reduce
  implicit none
  integer :: s, i

  do concurrent (i=1:10) reduce(+:s)
    s = s + 1
  end do
end

! CHECK-LABEL:  fir.declare_reduction @add_reduction_i32 : i32 init {
! CHECK:        ^bb0(%[[ARG0:.*]]: i32):
! CHECK:          %[[VAL_0:.*]] = arith.constant 0 : i32
! CHECK:          fir.yield(%[[VAL_0]] : i32)
! CHECK:        } combiner {
! CHECK:          ^bb0(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32):
! CHECK:          %[[VAL_3:.*]] = arith.addi %[[VAL_1]], %[[VAL_2]] : i32
! CHECK:          fir.yield(%[[VAL_3]] : i32)
! CHECK:        }

! CHECK-LABEL:  func.func @_QPdo_concurrent_reduce() {
! CHECK:           %[[S_ALLOC:.*]] = fir.alloca i32 {bindc_name = "s", uniq_name = "_QFdo_concurrent_reduceEs"}
! CHECK:           %[[S_DECL:.*]]:2 = hlfir.declare %[[S_ALLOC]] {uniq_name = "_QFdo_concurrent_reduceEs"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

! CHECK:           fir.do_concurrent {
! CHECK:             %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i"}
! CHECK:             %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFdo_concurrent_reduceEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             fir.do_concurrent.loop (%{{.*}}) = (%{{.*}}) to (%{{.*}}) step (%{{[^[:space:]]+}})
! CHECK-SAME:          reduce(@add_reduction_i32 #fir.reduce_attr<add> %[[S_DECL]]#0 -> %[[S_ARG:.*]] : !fir.ref<i32>) {

! CHECK:               %[[S_ARG_DECL:.*]]:2 = hlfir.declare %[[S_ARG]] {uniq_name = "_QFdo_concurrent_reduceEs"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:               %[[S_ARG_VAL:.*]] = fir.load %[[S_ARG_DECL]]#0 : !fir.ref<i32>
! CHECK:               %[[C1:.*]] = arith.constant 1 : i32
! CHECK:               %[[RED_UPDATE:.*]] = arith.addi %[[S_ARG_VAL]], %[[C1]] : i32
! CHECK:               hlfir.assign %[[RED_UPDATE]] to %[[S_ARG_DECL]]#0 : i32, !fir.ref<i32>

! CHECK:             }
! CHECK:           }
! CHECK:           return
! CHECK:        }
