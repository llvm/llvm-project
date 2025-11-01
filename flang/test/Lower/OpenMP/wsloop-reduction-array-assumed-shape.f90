! RUN: bbc -emit-hlfir -fopenmp -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s

program reduce_assumed_shape
real(8), dimension(2) :: r
r = 0
call reduce(r)
print *, r

contains
subroutine reduce(r)
  implicit none
  real(8),intent(inout) :: r(:)
  integer :: i = 0

  !$omp parallel do reduction(+:r)
  do i=0,10
    r(1) = i
    r(2) = 1
  enddo
  !$omp end parallel do
end subroutine
end program

! CHECK-LABEL:   omp.declare_reduction @add_reduction_byref_box_Uxf64 : !fir.ref<!fir.box<!fir.array<?xf64>>> alloc {
! CHECK:           %[[VAL_8:.*]] = fir.alloca !fir.box<!fir.array<?xf64>>
! CHECK:           omp.yield(%[[VAL_8]] : !fir.ref<!fir.box<!fir.array<?xf64>>>)
! CHECK-LABEL:   } init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<?xf64>>>, %[[ALLOC:.*]]: !fir.ref<!fir.box<!fir.array<?xf64>>>):
! CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?xf64>>>
! CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_3]] : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]] = fir.allocmem !fir.array<?xf64>, %[[VAL_4]]#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]](%[[VAL_5]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.heap<!fir.array<?xf64>>)
! CHECK:           %[[C0:.*]] = arith.constant 0 : index
! CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[VAL_2]], %[[C0]] : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
! CHECK:           %[[SHIFT:.*]] = fir.shape_shift %[[DIMS]]#0, %[[DIMS]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[REBOX:.*]] = fir.rebox %[[VAL_7]]#0(%[[SHIFT]]) : (!fir.box<!fir.array<?xf64>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xf64>
! CHECK:           hlfir.assign %[[VAL_1]] to %[[REBOX]] : f64, !fir.box<!fir.array<?xf64>>
! CHECK:           fir.store %[[REBOX]] to %[[ALLOC]] : !fir.ref<!fir.box<!fir.array<?xf64>>>
! CHECK:           omp.yield(%[[ALLOC]] : !fir.ref<!fir.box<!fir.array<?xf64>>>)
! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<?xf64>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.array<?xf64>>>):
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?xf64>>>
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.array<?xf64>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_4]] : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
! CHECK:           %[[C1:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_6:.*]] = fir.shape_shift %[[C1]], %[[VAL_5]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_8:.*]] = %[[VAL_7]] to %[[VAL_5]]#1 step %[[VAL_7]] unordered {
! CHECK:             %[[VAL_9:.*]] = fir.array_coor %[[VAL_2]](%[[VAL_6]]) %[[VAL_8]] : (!fir.box<!fir.array<?xf64>>, !fir.shapeshift<1>, index) -> !fir.ref<f64>
! CHECK:             %[[VAL_10:.*]] = fir.array_coor %[[VAL_3]](%[[VAL_6]]) %[[VAL_8]] : (!fir.box<!fir.array<?xf64>>, !fir.shapeshift<1>, index) -> !fir.ref<f64>
! CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_9]] : !fir.ref<f64>
! CHECK:             %[[VAL_12:.*]] = fir.load %[[VAL_10]] : !fir.ref<f64>
! CHECK:             %[[VAL_13:.*]] = arith.addf %[[VAL_11]], %[[VAL_12]] fastmath<contract> : f64
! CHECK:             fir.store %[[VAL_13]] to %[[VAL_9]] : !fir.ref<f64>
! CHECK:           }
! CHECK:           omp.yield(%[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?xf64>>>)
! CHECK:         }  cleanup {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<?xf64>>>):
! CHECK:           %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?xf64>>>
! CHECK:           %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<?xf64>>) -> i64
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:           fir.if %[[VAL_5]] {
! CHECK:             %[[VAL_6:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<?xf64>>) -> !fir.heap<!fir.array<?xf64>>
! CHECK:             fir.freemem %[[VAL_6]] : !fir.heap<!fir.array<?xf64>>
! CHECK:           }
! CHECK:           omp.yield
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPreduce(
! CHECK-SAME:                                  %[[VAL_0:.*]]: !fir.box<!fir.array<?xf64>> {fir.bindc_name = "r"}) attributes {{.*}} {
! CHECK:           %[[VAL_1:.*]] = fir.address_of(@_QFFreduceEi) : !fir.ref<i32>
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFFreduceEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {fortran_attrs = {{.*}}, uniq_name = "_QFFreduceEr"} : (!fir.box<!fir.array<?xf64>>, !fir.dscope) -> (!fir.box<!fir.array<?xf64>>, !fir.box<!fir.array<?xf64>>)
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_4:.*]] = fir.alloca !fir.box<!fir.array<?xf64>>
! CHECK:             fir.store %[[VAL_3]]#0 to %[[VAL_4]] : !fir.ref<!fir.box<!fir.array<?xf64>>>
! CHECK:             %[[VAL_7:.*]] = arith.constant 0 : i32
! CHECK:             %[[VAL_8:.*]] = arith.constant 10 : i32
! CHECK:             %[[VAL_9:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop private(@{{.*}} %{{.*}}#0 -> %[[VAL_5:.*]] : !fir.ref<i32>) reduction(byref @add_reduction_byref_box_Uxf64 %[[VAL_4]] -> %[[VAL_10:.*]] : !fir.ref<!fir.box<!fir.array<?xf64>>>) {
! CHECK-NEXT:          omp.loop_nest (%[[VAL_11:.*]]) : i32 = (%[[VAL_7]]) to (%[[VAL_8]]) inclusive step (%[[VAL_9]]) {
! CHECK:                 %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFFreduceEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:                 %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_10]] {fortran_attrs = {{.*}}, uniq_name = "_QFFreduceEr"} : (!fir.ref<!fir.box<!fir.array<?xf64>>>) -> (!fir.ref<!fir.box<!fir.array<?xf64>>>, !fir.ref<!fir.box<!fir.array<?xf64>>>)
! CHECK:                 hlfir.assign %[[VAL_11]] to %[[VAL_6]]#0 : i32, !fir.ref<i32>
! CHECK:                 %[[VAL_13:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
! CHECK:                 %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> f64
! CHECK:                 %[[VAL_15:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<!fir.box<!fir.array<?xf64>>>
! CHECK:                 %[[VAL_16:.*]] = arith.constant 1 : index
! CHECK:                 %[[VAL_17:.*]] = hlfir.designate %[[VAL_15]] (%[[VAL_16]])  : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
! CHECK:                 hlfir.assign %[[VAL_14]] to %[[VAL_17]] : f64, !fir.ref<f64>
! CHECK:                 %[[VAL_18:.*]] = arith.constant 1.000000e+00 : f64
! CHECK:                 %[[VAL_19:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<!fir.box<!fir.array<?xf64>>>
! CHECK:                 %[[VAL_20:.*]] = arith.constant 2 : index
! CHECK:                 %[[VAL_21:.*]] = hlfir.designate %[[VAL_19]] (%[[VAL_20]])  : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
! CHECK:                 hlfir.assign %[[VAL_18]] to %[[VAL_21]] : f64, !fir.ref<f64>
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }
