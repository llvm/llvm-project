! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

program reduce
integer, dimension(3) :: i = 0

!$omp parallel reduction(+:i)
i(1) = 1
i(2) = 2
i(3) = 3
!$omp end parallel

print *,i
end program

! CHECK-LABEL:   omp.declare_reduction @add_reduction_byref_box_3xi32 : !fir.ref<!fir.box<!fir.array<3xi32>>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<3xi32>>>):
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.array<3xi32> {bindc_name = ".tmp"}
! CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_5]]) {uniq_name = ".tmp"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
! CHECK:           %[[VAL_7:.*]] = fir.embox %[[VAL_6]]#0(%[[VAL_5]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_7]] : i32, !fir.box<!fir.array<3xi32>>
! CHECK:           %[[VAL_8:.*]] = fir.alloca !fir.box<!fir.array<3xi32>>
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_8]] : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CHECK:           omp.yield(%[[VAL_8]] : !fir.ref<!fir.box<!fir.array<3xi32>>>)
! CHECK:         } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<3xi32>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.array<3xi32>>>):
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_4]] : (!fir.box<!fir.array<3xi32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_6:.*]] = fir.shape_shift %[[VAL_5]]#0, %[[VAL_5]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_8:.*]] = %[[VAL_7]] to %[[VAL_5]]#1 step %[[VAL_7]] unordered {
! CHECK:             %[[VAL_9:.*]] = fir.array_coor %[[VAL_2]](%[[VAL_6]]) %[[VAL_8]] : (!fir.box<!fir.array<3xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_10:.*]] = fir.array_coor %[[VAL_3]](%[[VAL_6]]) %[[VAL_8]] : (!fir.box<!fir.array<3xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
! CHECK:             %[[VAL_12:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
! CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_11]], %[[VAL_12]] : i32
! CHECK:             fir.store %[[VAL_13]] to %[[VAL_9]] : !fir.ref<i32>
! CHECK:           }
! CHECK:           omp.yield(%[[VAL_0]] : !fir.ref<!fir.box<!fir.array<3xi32>>>)
! CHECK:         }

! CHECK-LABEL:   func.func @_QQmain()
! CHECK:           %[[VAL_0:.*]] = fir.address_of(@_QFEi) : !fir.ref<!fir.array<3xi32>>
! CHECK:           %[[VAL_1:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_2]]) {uniq_name = "_QFEi"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
! CHECK:           %[[VAL_4:.*]] = fir.embox %[[VAL_3]]#1(%[[VAL_2]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
! CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.box<!fir.array<3xi32>>
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_5]] : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CHECK:           omp.parallel byref reduction(@add_reduction_byref_box_3xi32 %[[VAL_5]] -> %[[VAL_6:.*]] : !fir.ref<!fir.box<!fir.array<3xi32>>>) {
! CHECK:             %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QFEi"} : (!fir.ref<!fir.box<!fir.array<3xi32>>>) -> (!fir.ref<!fir.box<!fir.array<3xi32>>>, !fir.ref<!fir.box<!fir.array<3xi32>>>)
! CHECK:             %[[VAL_8:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_9:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CHECK:             %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_11:.*]] = hlfir.designate %[[VAL_9]] (%[[VAL_10]])  : (!fir.box<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:             hlfir.assign %[[VAL_8]] to %[[VAL_11]] : i32, !fir.ref<i32>
! CHECK:             %[[VAL_12:.*]] = arith.constant 2 : i32
! CHECK:             %[[VAL_13:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CHECK:             %[[VAL_14:.*]] = arith.constant 2 : index
! CHECK:             %[[VAL_15:.*]] = hlfir.designate %[[VAL_13]] (%[[VAL_14]])  : (!fir.box<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:             hlfir.assign %[[VAL_12]] to %[[VAL_15]] : i32, !fir.ref<i32>
! CHECK:             %[[VAL_16:.*]] = arith.constant 3 : i32
! CHECK:             %[[VAL_17:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CHECK:             %[[VAL_18:.*]] = arith.constant 3 : index
! CHECK:             %[[VAL_19:.*]] = hlfir.designate %[[VAL_17]] (%[[VAL_18]])  : (!fir.box<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:             hlfir.assign %[[VAL_16]] to %[[VAL_19]] : i32, !fir.ref<i32>
! CHECK:             omp.terminator
! CHECK:           }
