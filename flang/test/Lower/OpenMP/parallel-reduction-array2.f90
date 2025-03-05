! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

program reduce
integer, dimension(3) :: i = 0

!$omp parallel reduction(+:i)
i(1) = i(1) + 1
i(2) = i(2) + 2
i(3) = i(3) + 3
!$omp end parallel

print *,i
end program

! CHECK-LABEL:   omp.declare_reduction @add_reduction_byref_box_3xi32 : !fir.ref<!fir.box<!fir.array<3xi32>>> alloc {
! CHECK:           %[[VAL_8:.*]] = fir.alloca !fir.box<!fir.array<3xi32>>
! CHECK:           omp.yield(%[[VAL_8]] : !fir.ref<!fir.box<!fir.array<3xi32>>>)
! CHECK-LABEL:   } init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<3xi32>>>, %[[ALLOC:.*]]: !fir.ref<!fir.box<!fir.array<3xi32>>>):
! CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_1:.*]] = fir.allocmem !fir.array<3xi32>
! CHECK:           %[[TRUE:.*]]  = arith.constant true
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_5]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<3xi32>>, !fir.heap<!fir.array<3xi32>>)
! CHECK:           %[[C0:.*]] = arith.constant 0 : index
! CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[VAL_3]], %[[C0]] : (!fir.box<!fir.array<3xi32>>, index) -> (index, index, index)
! CHECK:           %[[SHIFT:.*]] = fir.shape_shift %[[DIMS]]#0, %[[DIMS]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_7:.*]] = fir.embox %[[VAL_6]]#0(%[[SHIFT]]) : (!fir.heap<!fir.array<3xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<3xi32>>
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_7]] : i32, !fir.box<!fir.array<3xi32>>
! CHECK:           fir.store %[[VAL_7]] to %[[ALLOC]] : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CHECK:           omp.yield(%[[ALLOC]] : !fir.ref<!fir.box<!fir.array<3xi32>>>)
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
! CHECK:         }  cleanup {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<3xi32>>>):
! CHECK:           %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CHECK:           %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<3xi32>>) -> !fir.ref<!fir.array<3xi32>>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<3xi32>>) -> i64
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:           fir.if %[[VAL_5]] {
! CHECK:             %[[VAL_6:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<3xi32>>) -> !fir.heap<!fir.array<3xi32>>
! CHECK:             fir.freemem %[[VAL_6]] : !fir.heap<!fir.array<3xi32>>
! CHECK:           }
! CHECK:           omp.yield
! CHECK:         }

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "reduce"} {
! CHECK:           %[[VAL_0:.*]] = fir.address_of(@_QFEi) : !fir.ref<!fir.array<3xi32>>
! CHECK:           %[[VAL_1:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_2]]) {uniq_name = "_QFEi"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
! CHECK:           %[[VAL_4:.*]] = fir.embox %[[VAL_3]]#0(%[[VAL_2]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
! CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.box<!fir.array<3xi32>>
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_5]] : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CHECK:           omp.parallel reduction(byref @add_reduction_byref_box_3xi32 %[[VAL_5]] -> %[[VAL_6:.*]] : !fir.ref<!fir.box<!fir.array<3xi32>>>) {
! CHECK:             %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QFEi"} : (!fir.ref<!fir.box<!fir.array<3xi32>>>) -> (!fir.ref<!fir.box<!fir.array<3xi32>>>, !fir.ref<!fir.box<!fir.array<3xi32>>>)
! CHECK:             %[[VAL_8:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CHECK:             %[[VAL_9:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_10:.*]] = hlfir.designate %[[VAL_8]] (%[[VAL_9]])  : (!fir.box<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
! CHECK:             %[[VAL_12:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_11]], %[[VAL_12]] : i32
! CHECK:             %[[VAL_14:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CHECK:             %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_16:.*]] = hlfir.designate %[[VAL_14]] (%[[VAL_15]])  : (!fir.box<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:             hlfir.assign %[[VAL_13]] to %[[VAL_16]] : i32, !fir.ref<i32>
! CHECK:             %[[VAL_17:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CHECK:             %[[VAL_18:.*]] = arith.constant 2 : index
! CHECK:             %[[VAL_19:.*]] = hlfir.designate %[[VAL_17]] (%[[VAL_18]])  : (!fir.box<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<i32>
! CHECK:             %[[VAL_21:.*]] = arith.constant 2 : i32
! CHECK:             %[[VAL_22:.*]] = arith.addi %[[VAL_20]], %[[VAL_21]] : i32
! CHECK:             %[[VAL_23:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CHECK:             %[[VAL_24:.*]] = arith.constant 2 : index
! CHECK:             %[[VAL_25:.*]] = hlfir.designate %[[VAL_23]] (%[[VAL_24]])  : (!fir.box<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:             hlfir.assign %[[VAL_22]] to %[[VAL_25]] : i32, !fir.ref<i32>
! CHECK:             %[[VAL_26:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CHECK:             %[[VAL_27:.*]] = arith.constant 3 : index
! CHECK:             %[[VAL_28:.*]] = hlfir.designate %[[VAL_26]] (%[[VAL_27]])  : (!fir.box<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_29:.*]] = fir.load %[[VAL_28]] : !fir.ref<i32>
! CHECK:             %[[VAL_30:.*]] = arith.constant 3 : i32
! CHECK:             %[[VAL_31:.*]] = arith.addi %[[VAL_29]], %[[VAL_30]] : i32
! CHECK:             %[[VAL_32:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CHECK:             %[[VAL_33:.*]] = arith.constant 3 : index
! CHECK:             %[[VAL_34:.*]] = hlfir.designate %[[VAL_32]] (%[[VAL_33]])  : (!fir.box<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:             hlfir.assign %[[VAL_31]] to %[[VAL_34]] : i32, !fir.ref<i32>
! CHECK:             omp.terminator
! CHECK:           }
