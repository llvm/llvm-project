! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

program reduce
integer, dimension(2:4, 2) :: i = 0

!$omp parallel reduction(+:i)
i(3, 1) = 3
!$omp end parallel

print *,i

end program

! CHECK-LABEL:   omp.declare_reduction @add_reduction_byref_box_3x2xi32 : !fir.ref<!fir.box<!fir.array<3x2xi32>>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<3x2xi32>>>):
! CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<3x2xi32>>>
! CHECK:           %[[VAL_15:.*]] = fir.alloca !fir.box<!fir.array<3x2xi32>>
! CHECK:           %[[VAL_3:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_4:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_3]], %[[VAL_4]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_6:.*]] = fir.allocmem !fir.array<3x2xi32> {bindc_name = ".tmp", uniq_name = ""}
! CHECK:           %[[VAL_7:.*]] = arith.constant true
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_6]](%[[VAL_5]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<3x2xi32>>, !fir.shape<2>) -> (!fir.heap<!fir.array<3x2xi32>>, !fir.heap<!fir.array<3x2xi32>>)
! CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_9]] : (!fir.box<!fir.array<3x2xi32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_11]] : (!fir.box<!fir.array<3x2xi32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_13:.*]] = fir.shape_shift %[[VAL_10]]#0, %[[VAL_10]]#1, %[[VAL_12]]#0, %[[VAL_12]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:           %[[VAL_14:.*]] = fir.embox %[[VAL_8]]#0(%[[VAL_13]]) : (!fir.heap<!fir.array<3x2xi32>>, !fir.shapeshift<2>) -> !fir.box<!fir.array<3x2xi32>>
! CHECK:           hlfir.assign %[[VAL_1]] to %[[VAL_14]] : i32, !fir.box<!fir.array<3x2xi32>>
! CHECK:           fir.store %[[VAL_14]] to %[[VAL_15]] : !fir.ref<!fir.box<!fir.array<3x2xi32>>>
! CHECK:           omp.yield(%[[VAL_15]] : !fir.ref<!fir.box<!fir.array<3x2xi32>>>)
! CHECK:         } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<3x2xi32>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.array<3x2xi32>>>):
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<3x2xi32>>>
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.array<3x2xi32>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_4]] : (!fir.box<!fir.array<3x2xi32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_6]] : (!fir.box<!fir.array<3x2xi32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_8:.*]] = fir.shape_shift %[[VAL_5]]#0, %[[VAL_5]]#1, %[[VAL_7]]#0, %[[VAL_7]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:           %[[VAL_9:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_10:.*]] = %[[VAL_9]] to %[[VAL_7]]#1 step %[[VAL_9]] unordered {
! CHECK:             fir.do_loop %[[VAL_11:.*]] = %[[VAL_9]] to %[[VAL_5]]#1 step %[[VAL_9]] unordered {
! CHECK:               %[[VAL_12:.*]] = fir.array_coor %[[VAL_2]](%[[VAL_8]]) %[[VAL_11]], %[[VAL_10]] : (!fir.box<!fir.array<3x2xi32>>, !fir.shapeshift<2>, index, index) -> !fir.ref<i32>
! CHECK:               %[[VAL_13:.*]] = fir.array_coor %[[VAL_3]](%[[VAL_8]]) %[[VAL_11]], %[[VAL_10]] : (!fir.box<!fir.array<3x2xi32>>, !fir.shapeshift<2>, index, index) -> !fir.ref<i32>
! CHECK:               %[[VAL_14:.*]] = fir.load %[[VAL_12]] : !fir.ref<i32>
! CHECK:               %[[VAL_15:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
! CHECK:               %[[VAL_16:.*]] = arith.addi %[[VAL_14]], %[[VAL_15]] : i32
! CHECK:               fir.store %[[VAL_16]] to %[[VAL_12]] : !fir.ref<i32>
! CHECK:             }
! CHECK:           }
! CHECK:           omp.yield(%[[VAL_0]] : !fir.ref<!fir.box<!fir.array<3x2xi32>>>)
! CHECK:         }  cleanup {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<3x2xi32>>>):
! CHECK:           %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<3x2xi32>>>
! CHECK:           %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<3x2xi32>>) -> !fir.ref<!fir.array<3x2xi32>>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<3x2xi32>>) -> i64
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:           fir.if %[[VAL_5]] {
! CHECK:             %[[VAL_6:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<3x2xi32>>) -> !fir.heap<!fir.array<3x2xi32>>
! CHECK:             fir.freemem %[[VAL_6]] : !fir.heap<!fir.array<3x2xi32>>
! CHECK:           }
! CHECK:           omp.yield
! CHECK:         }

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "reduce"} {
! CHECK:           %[[VAL_0:.*]] = fir.address_of(@_QFEi) : !fir.ref<!fir.array<3x2xi32>>
! CHECK:           %[[VAL_1:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_2:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_4:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_5:.*]] = fir.shape_shift %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_5]]) {uniq_name = "_QFEi"} : (!fir.ref<!fir.array<3x2xi32>>, !fir.shapeshift<2>) -> (!fir.box<!fir.array<3x2xi32>>, !fir.ref<!fir.array<3x2xi32>>)
! CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.box<!fir.array<3x2xi32>>
! CHECK:           fir.store %[[VAL_6]]#0 to %[[VAL_7]] : !fir.ref<!fir.box<!fir.array<3x2xi32>>>
! CHECK:           omp.parallel reduction(byref @add_reduction_byref_box_3x2xi32 %[[VAL_7]] -> %[[VAL_8:.*]] : !fir.ref<!fir.box<!fir.array<3x2xi32>>>) {
! CHECK:             %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {uniq_name = "_QFEi"} : (!fir.ref<!fir.box<!fir.array<3x2xi32>>>) -> (!fir.ref<!fir.box<!fir.array<3x2xi32>>>, !fir.ref<!fir.box<!fir.array<3x2xi32>>>)
! CHECK:             %[[VAL_10:.*]] = arith.constant 3 : i32
! CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<!fir.box<!fir.array<3x2xi32>>>
! CHECK:             %[[VAL_12:.*]] = arith.constant 3 : index
! CHECK:             %[[VAL_13:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_14:.*]] = hlfir.designate %[[VAL_11]] (%[[VAL_12]], %[[VAL_13]])  : (!fir.box<!fir.array<3x2xi32>>, index, index) -> !fir.ref<i32>
! CHECK:             hlfir.assign %[[VAL_10]] to %[[VAL_14]] : i32, !fir.ref<i32>
! CHECK:             omp.terminator
! CHECK:           }

