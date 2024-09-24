! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK-LABEL:   omp.declare_reduction @add_reduction_byref_box_Uxi32 : !fir.ref<!fir.box<!fir.array<?xi32>>> alloc {
! CHECK:           %[[VAL_8:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
! CHECK:           omp.yield(%[[VAL_8]] : !fir.ref<!fir.box<!fir.array<?xi32>>>)
! CHECK-LABEL:   } init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<?xi32>>>, %[[ALLOC:.*]]: !fir.ref<!fir.box<!fir.array<?xi32>>>):
! CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_3]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_4]]#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:           %[[TRUE:.*]]  = arith.constant true
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]](%[[VAL_5]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:           %[[C0:.*]] = arith.constant 0 : index
! CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[VAL_2]], %[[C0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[SHIFT:.*]] = fir.shape_shift %[[DIMS]]#0, %[[DIMS]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[REBOX:.*]] = fir.rebox %[[VAL_7]]#0(%[[SHIFT]]) : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           hlfir.assign %[[VAL_1]] to %[[REBOX]] : i32, !fir.box<!fir.array<?xi32>>
! CHECK:           fir.store %[[REBOX]] to %[[ALLOC]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:           omp.yield(%[[ALLOC]] : !fir.ref<!fir.box<!fir.array<?xi32>>>)
! CHECK:         } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<?xi32>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.array<?xi32>>>):
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_4]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_6:.*]] = fir.shape_shift %[[VAL_5]]#0, %[[VAL_5]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_8:.*]] = %[[VAL_7]] to %[[VAL_5]]#1 step %[[VAL_7]] unordered {
! CHECK:             %[[VAL_9:.*]] = fir.array_coor %[[VAL_2]](%[[VAL_6]]) %[[VAL_8]] : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_10:.*]] = fir.array_coor %[[VAL_3]](%[[VAL_6]]) %[[VAL_8]] : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
! CHECK:             %[[VAL_12:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
! CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_11]], %[[VAL_12]] : i32
! CHECK:             fir.store %[[VAL_13]] to %[[VAL_9]] : !fir.ref<i32>
! CHECK:           }
! CHECK:           omp.yield(%[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?xi32>>>)
! CHECK:         }  cleanup {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<?xi32>>>):
! CHECK:           %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:           %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<?xi32>>) -> i64
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:           fir.if %[[VAL_5]] {
! CHECK:             %[[VAL_6:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<?xi32>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:             fir.freemem %[[VAL_6]] : !fir.heap<!fir.array<?xi32>>
! CHECK:           }
! CHECK:           omp.yield
! CHECK:         }

! CHECK-LABEL:   func.func @_QPs(
! CHECK-SAME:                    %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFsEx"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsEi"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFsEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> i64
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
! CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_8:.*]] = arith.cmpi sgt, %[[VAL_6]], %[[VAL_7]] : index
! CHECK:           %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_6]], %[[VAL_7]] : index
! CHECK:           %[[VAL_10:.*]] = fir.alloca !fir.array<?xi32>, %[[VAL_9]] {bindc_name = "c", uniq_name = "_QFsEc"}
! CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_10]](%[[VAL_11]]) {uniq_name = "_QFsEc"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>)
! CHECK:           %[[VAL_13:.*]] = arith.constant 0 : i32
! CHECK:           hlfir.assign %[[VAL_13]] to %[[VAL_12]]#0 : i32, !fir.box<!fir.array<?xi32>>
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_14:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
! CHECK:             fir.store %[[VAL_12]]#0 to %[[VAL_14]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:             %[[VAL_15:.*]] = fir.alloca i32 {bindc_name = "i", pinned, {{.*}}}
! CHECK:             %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_15]] {uniq_name = "_QFsEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             %[[VAL_17:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_18:.*]] = arith.constant 100 : i32
! CHECK:             %[[VAL_19:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop reduction(byref @add_reduction_byref_box_Uxi32 %[[VAL_14]] -> %[[VAL_20:.*]] : !fir.ref<!fir.box<!fir.array<?xi32>>>) {
! CHECK-NEXT:          omp.loop_nest (%[[VAL_21:.*]]) : i32 = (%[[VAL_17]]) to (%[[VAL_18]]) inclusive step (%[[VAL_19]]) {
! CHECK:                 %[[VAL_22:.*]]:2 = hlfir.declare %[[VAL_20]] {uniq_name = "_QFsEc"} : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> (!fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.ref<!fir.box<!fir.array<?xi32>>>)
! CHECK:                 fir.store %[[VAL_21]] to %[[VAL_16]]#1 : !fir.ref<i32>
! CHECK:                 %[[VAL_23:.*]] = fir.load %[[VAL_22]]#0 : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:                 %[[VAL_24:.*]] = fir.load %[[VAL_16]]#0 : !fir.ref<i32>
! CHECK:                 %[[VAL_25:.*]] = arith.constant 0 : index
! CHECK:                 %[[VAL_26:.*]]:3 = fir.box_dims %[[VAL_23]], %[[VAL_25]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:                 %[[VAL_27:.*]] = fir.shape %[[VAL_26]]#1 : (index) -> !fir.shape<1>
! CHECK:                 %[[VAL_28:.*]] = hlfir.elemental %[[VAL_27]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
! CHECK:                 ^bb0(%[[VAL_29:.*]]: index):
! CHECK:                   %[[VAL_30:.*]] = arith.constant 0 : index
! CHECK:                   %[[VAL_31:.*]]:3 = fir.box_dims %[[VAL_23]], %[[VAL_30]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:                   %[[VAL_32:.*]] = arith.constant 1 : index
! CHECK:                   %[[VAL_33:.*]] = arith.subi %[[VAL_31]]#0, %[[VAL_32]] : index
! CHECK:                   %[[VAL_34:.*]] = arith.addi %[[VAL_29]], %[[VAL_33]] : index
! CHECK:                   %[[VAL_35:.*]] = hlfir.designate %[[VAL_23]] (%[[VAL_34]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:                   %[[VAL_36:.*]] = fir.load %[[VAL_35]] : !fir.ref<i32>
! CHECK:                   %[[VAL_37:.*]] = arith.addi %[[VAL_36]], %[[VAL_24]] : i32
! CHECK:                   hlfir.yield_element %[[VAL_37]] : i32
! CHECK:                 }
! CHECK:                 %[[VAL_38:.*]] = fir.load %[[VAL_22]]#0 : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:                 hlfir.assign %[[VAL_28]] to %[[VAL_38]] : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:                 hlfir.destroy %[[VAL_28]] : !hlfir.expr<?xi32>
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           %[[VAL_39:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_40:.*]] = hlfir.designate %[[VAL_12]]#0 (%[[VAL_39]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:           %[[VAL_41:.*]] = fir.load %[[VAL_40]] : !fir.ref<i32>
! CHECK:           %[[VAL_42:.*]] = arith.constant 5050 : i32
! CHECK:           %[[VAL_43:.*]] = arith.cmpi ne, %[[VAL_41]], %[[VAL_42]] : i32
! CHECK:           cf.cond_br %[[VAL_43]], ^bb1, ^bb2
! CHECK:         ^bb1:
! CHECK:           %[[VAL_44:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_45:.*]] = arith.constant false
! CHECK:           %[[VAL_46:.*]] = arith.constant false
! CHECK:           %[[VAL_47:.*]] = fir.call @_FortranAStopStatement(%[[VAL_44]], %[[VAL_45]], %[[VAL_46]]) fastmath<contract> : (i32, i1, i1) -> none
! CHECK:           fir.unreachable
! CHECK:         ^bb2:
! CHECK:           return
! CHECK:         }
! CHECK:         func.func private @_FortranAStopStatement(i32, i1, i1) -> none attributes {fir.runtime}

subroutine s(x)
    integer :: x
    integer :: c(x)
    c = 0
    !$omp parallel do reduction(+:c)
    do i = 1, 100
        c = c + i
    end do
    !$omp end parallel do

    if (c(1) /= 5050) stop 1
end subroutine s
