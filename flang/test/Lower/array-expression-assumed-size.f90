! RUN: bbc --emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: bbc -hlfir=false %s -o - | FileCheck --check-prefix=PostOpt %s


subroutine assumed_size_test(a)
  integer :: a(10,*)
  a(:, 1:2) = a(:, 3:4)
end subroutine assumed_size_test

subroutine assumed_size_forall_test(b)
  integer :: b(10,*)
  forall (i=2:6)
     b(i, 1:2) = b(i, 3:4)
  end forall
end subroutine assumed_size_forall_test

! CHECK-LABEL: func @_QPassumed_size_test(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10x?xi32>>{{.*}}) {
! CHECK:         %[[VAL_1A:.*]] = fir.convert %c10{{.*}} : (i64) -> index 
! CHECK:         %[[VAL_1B:.*]] = arith.cmpi sgt, %[[VAL_1A]], %c0{{.*}} : index 
! CHECK:         %[[VAL_1:.*]] = arith.select %[[VAL_1B]], %[[VAL_1A]], %c0{{.*}} : index
! CHECK:         %[[VAL_2:.*]] = fir.undefined index
! CHECK:         %[[VAL_3:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
! CHECK:         %[[VAL_6:.*]] = arith.addi %[[VAL_3]], %[[VAL_1]] : index
! CHECK:         %[[VAL_7:.*]] = arith.subi %[[VAL_6]], %[[VAL_3]] : index
! CHECK:         %[[VAL_8:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_9:.*]] = arith.subi %[[VAL_7]], %[[VAL_3]] : index
! CHECK:         %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_5]] : index
! CHECK:         %[[VAL_11:.*]] = arith.divsi %[[VAL_10]], %[[VAL_5]] : index
! CHECK:         %[[VAL_12:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[VAL_8]] : index
! CHECK:         %[[VAL_13:.*]] = arith.select %[[VAL_12]], %[[VAL_11]], %[[VAL_8]] : index
! CHECK:         %[[VAL_14:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i64) -> index
! CHECK:         %[[VAL_16:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i64) -> index
! CHECK:         %[[VAL_18:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i64) -> index
! CHECK:         %[[VAL_20:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_21:.*]] = arith.subi %[[VAL_19]], %[[VAL_15]] : index
! CHECK:         %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_17]] : index
! CHECK:         %[[VAL_23:.*]] = arith.divsi %[[VAL_22]], %[[VAL_17]] : index
! CHECK:         %[[VAL_24:.*]] = arith.cmpi sgt, %[[VAL_23]], %[[VAL_20]] : index
! CHECK:         %[[VAL_25:.*]] = arith.select %[[VAL_24]], %[[VAL_23]], %[[VAL_20]] : index
! CHECK:         %[[VAL_26:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_27:.*]] = fir.slice %[[VAL_3]], %[[VAL_7]], %[[VAL_5]], %[[VAL_15]], %[[VAL_19]], %[[VAL_17]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:         %[[VAL_28:.*]] = fir.array_load %[[VAL_0]](%[[VAL_26]]) {{\[}}%[[VAL_27]]] : (!fir.ref<!fir.array<10x?xi32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.array<10x?xi32>
! CHECK:         %[[VAL_29:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_30:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i64) -> index
! CHECK:         %[[VAL_32:.*]] = arith.addi %[[VAL_29]], %[[VAL_1]] : index
! CHECK:         %[[VAL_33:.*]] = arith.subi %[[VAL_32]], %[[VAL_29]] : index
! CHECK:         %[[VAL_34:.*]] = arith.constant 3 : i64
! CHECK:         %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i64) -> index
! CHECK:         %[[VAL_36:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i64) -> index
! CHECK:         %[[VAL_38:.*]] = arith.constant 4 : i64
! CHECK:         %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (i64) -> index
! CHECK:         %[[VAL_40:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_41:.*]] = fir.slice %[[VAL_29]], %[[VAL_33]], %[[VAL_31]], %[[VAL_35]], %[[VAL_39]], %[[VAL_37]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:         %[[VAL_42:.*]] = fir.array_load %[[VAL_0]](%[[VAL_40]]) {{\[}}%[[VAL_41]]] : (!fir.ref<!fir.array<10x?xi32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.array<10x?xi32>
! CHECK:         %[[VAL_43:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_44:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_45:.*]] = arith.subi %[[VAL_13]], %[[VAL_43]] : index
! CHECK:         %[[VAL_46:.*]] = arith.subi %[[VAL_25]], %[[VAL_43]] : index
! CHECK:         %[[VAL_47:.*]] = fir.do_loop %[[VAL_48:.*]] = %[[VAL_44]] to %[[VAL_46]] step %[[VAL_43]] unordered iter_args(%[[VAL_49:.*]] = %[[VAL_28]]) -> (!fir.array<10x?xi32>) {
! CHECK:           %[[VAL_50:.*]] = fir.do_loop %[[VAL_51:.*]] = %[[VAL_44]] to %[[VAL_45]] step %[[VAL_43]] unordered iter_args(%[[VAL_52:.*]] = %[[VAL_49]]) -> (!fir.array<10x?xi32>) {
! CHECK:             %[[VAL_53:.*]] = fir.array_fetch %[[VAL_42]], %[[VAL_51]], %[[VAL_48]] : (!fir.array<10x?xi32>, index, index) -> i32
! CHECK:             %[[VAL_54:.*]] = fir.array_update %[[VAL_52]], %[[VAL_53]], %[[VAL_51]], %[[VAL_48]] : (!fir.array<10x?xi32>, i32, index, index) -> !fir.array<10x?xi32>
! CHECK:             fir.result %[[VAL_54]] : !fir.array<10x?xi32>
! CHECK:           }
! CHECK:           fir.result %[[VAL_55:.*]] : !fir.array<10x?xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_28]], %[[VAL_56:.*]] to %[[VAL_0]]{{\[}}%[[VAL_27]]] : !fir.array<10x?xi32>, !fir.array<10x?xi32>, !fir.ref<!fir.array<10x?xi32>>, !fir.slice<2>
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func @_QPassumed_size_forall_test(
! CHECK-SAME:       %[[VAL_0:.*]]: !fir.ref<!fir.array<10x?xi32>>{{.*}}) {
! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_2A:.*]] = fir.convert %c10{{.*}} : (i64) -> index 
! CHECK:         %[[VAL_2B:.*]] = arith.cmpi sgt, %[[VAL_2A]], %c0{{.*}} : index 
! CHECK:         %[[VAL_2:.*]] = arith.select %[[VAL_2B]], %[[VAL_2A]], %c0{{.*}} : index
! CHECK:         %[[VAL_3:.*]] = fir.undefined index
! CHECK:         %[[VAL_4:.*]] = arith.constant 2 : i32
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
! CHECK:         %[[VAL_6:.*]] = arith.constant 6 : i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_9:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_0]](%[[VAL_9]]) : (!fir.ref<!fir.array<10x?xi32>>, !fir.shape<2>) -> !fir.array<10x?xi32>
! CHECK:         %[[VAL_11:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_0]](%[[VAL_11]]) : (!fir.ref<!fir.array<10x?xi32>>, !fir.shape<2>) -> !fir.array<10x?xi32>
! CHECK:         %[[VAL_13:.*]] = fir.do_loop %[[VAL_14:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_8]] unordered iter_args(%[[VAL_15:.*]] = %[[VAL_10]]) -> (!fir.array<10x?xi32>) {
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_14]] : (index) -> i32
! CHECK:           fir.store %[[VAL_16]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_17:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> i64
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i64) -> index
! CHECK:           %[[VAL_21:.*]] = arith.subi %[[VAL_20]], %[[VAL_17]] : index
! CHECK:           %[[VAL_22:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i64) -> index
! CHECK:           %[[VAL_24:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i64) -> index
! CHECK:           %[[VAL_26:.*]] = arith.constant 2 : i64
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i64) -> index
! CHECK:           %[[VAL_28:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_29:.*]] = arith.subi %[[VAL_27]], %[[VAL_23]] : index
! CHECK:           %[[VAL_30:.*]] = arith.addi %[[VAL_29]], %[[VAL_25]] : index
! CHECK:           %[[VAL_31:.*]] = arith.divsi %[[VAL_30]], %[[VAL_25]] : index
! CHECK:           %[[VAL_32:.*]] = arith.cmpi sgt, %[[VAL_31]], %[[VAL_28]] : index
! CHECK:           %[[VAL_33:.*]] = arith.select %[[VAL_32]], %[[VAL_31]], %[[VAL_28]] : index
! CHECK:           %[[VAL_34:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_35:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i32) -> i64
! CHECK:           %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i64) -> index
! CHECK:           %[[VAL_38:.*]] = arith.subi %[[VAL_37]], %[[VAL_34]] : index
! CHECK:           %[[VAL_39:.*]] = arith.constant 3 : i64
! CHECK:           %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (i64) -> index
! CHECK:           %[[VAL_41:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (i64) -> index
! CHECK:           %[[VAL_43:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_44:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_45:.*]] = arith.subi %[[VAL_33]], %[[VAL_43]] : index
! CHECK:           %[[VAL_46:.*]] = fir.do_loop %[[VAL_47:.*]] = %[[VAL_44]] to %[[VAL_45]] step %[[VAL_43]] unordered iter_args(%[[VAL_48:.*]] = %[[VAL_15]]) -> (!fir.array<10x?xi32>) {
! CHECK:             %[[VAL_49:.*]] = arith.subi %[[VAL_40]], %[[VAL_34]] : index
! CHECK:             %[[VAL_50:.*]] = arith.muli %[[VAL_47]], %[[VAL_42]] : index
! CHECK:             %[[VAL_51:.*]] = arith.addi %[[VAL_49]], %[[VAL_50]] : index
! CHECK:             %[[VAL_52:.*]] = fir.array_fetch %[[VAL_12]], %[[VAL_38]], %[[VAL_51]] : (!fir.array<10x?xi32>, index, index) -> i32
! CHECK:             %[[VAL_53:.*]] = arith.subi %[[VAL_23]], %[[VAL_17]] : index
! CHECK:             %[[VAL_54:.*]] = arith.muli %[[VAL_47]], %[[VAL_25]] : index
! CHECK:             %[[VAL_55:.*]] = arith.addi %[[VAL_53]], %[[VAL_54]] : index
! CHECK:             %[[VAL_56:.*]] = fir.array_update %[[VAL_48]], %[[VAL_52]], %[[VAL_21]], %[[VAL_55]] : (!fir.array<10x?xi32>, i32, index, index) -> !fir.array<10x?xi32>
! CHECK:             fir.result %[[VAL_56]] : !fir.array<10x?xi32>
! CHECK:           }
! CHECK:           fir.result %[[VAL_57:.*]] : !fir.array<10x?xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_10]], %[[VAL_58:.*]] to %[[VAL_0]] : !fir.array<10x?xi32>, !fir.array<10x?xi32>, !fir.ref<!fir.array<10x?xi32>>
! CHECK:         return
! CHECK:       }

! PostOpt-LABEL: func @_QPassumed_size_test(
! PostOpt-SAME:        %[[VAL_0:.*]]: !fir.ref<!fir.array<10x?xi32>>{{.*}}) {
! PostOpt-DAG:         %[[VAL_1:.*]] = arith.constant 10 : index
! PostOpt-DAG:         %[[VAL_2:.*]] = arith.constant 1 : index
! PostOpt-DAG:         %[[VAL_3:.*]] = arith.constant 2 : index
! PostOpt-DAG:         %[[VAL_4:.*]] = arith.constant 0 : index
! PostOpt-DAG:         %[[VAL_5:.*]] = arith.constant 3 : index
! PostOpt-DAG:         %[[VAL_6:.*]] = arith.constant 4 : index
! PostOpt:         %[[VAL_7:.*]] = fir.undefined index
! PostOpt:         %[[VAL_8:.*]] = fir.shape %[[VAL_1]], %[[VAL_7]] : (index, index) -> !fir.shape<2>
! PostOpt:         %[[VAL_9:.*]] = fir.slice %[[VAL_2]], %[[VAL_1]], %[[VAL_2]], %[[VAL_2]], %[[VAL_3]], %[[VAL_2]] : (index, index, index, index, index, index) -> !fir.slice<2>
! PostOpt:         %[[VAL_10:.*]] = fir.allocmem !fir.array<10x?xi32>, %[[VAL_3]]
! PostOpt:         br ^bb1(%[[VAL_4]], %[[VAL_3]] : index, index)
! PostOpt:       ^bb1(%[[VAL_11:.*]]: index, %[[VAL_12:.*]]: index):
! PostOpt:         %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[VAL_4]] : index
! PostOpt:         cond_br %[[VAL_13]], ^bb2(%[[VAL_4]], %[[VAL_1]] : index, index), ^bb5
! PostOpt:       ^bb2(%[[VAL_14:.*]]: index, %[[VAL_15:.*]]: index):
! PostOpt:         %[[VAL_16:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[VAL_4]] : index
! PostOpt:         cond_br %[[VAL_16]], ^bb3, ^bb4
! PostOpt:       ^bb3:
! PostOpt:         %[[VAL_17:.*]] = arith.addi %[[VAL_14]], %[[VAL_2]] : index
! PostOpt:         %[[VAL_18:.*]] = arith.addi %[[VAL_11]], %[[VAL_2]] : index
! PostOpt:         %[[VAL_19:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_8]]) {{\[}}%[[VAL_9]]] %[[VAL_17]], %[[VAL_18]] : (!fir.ref<!fir.array<10x?xi32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<i32>
! PostOpt:         %[[VAL_20:.*]] = fir.array_coor %[[VAL_10]](%[[VAL_8]]) %[[VAL_17]], %[[VAL_18]] : (!fir.heap<!fir.array<10x?xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
! PostOpt:         %[[VAL_21:.*]] = fir.load %[[VAL_19]] : !fir.ref<i32>
! PostOpt:         fir.store %[[VAL_21]] to %[[VAL_20]] : !fir.ref<i32>
! PostOpt:         %[[VAL_22:.*]] = arith.subi %[[VAL_15]], %[[VAL_2]] : index
! PostOpt:         br ^bb2(%[[VAL_17]], %[[VAL_22]] : index, index)
! PostOpt:       ^bb4:
! PostOpt:         %[[VAL_23:.*]] = arith.addi %[[VAL_11]], %[[VAL_2]] : index
! PostOpt:         %[[VAL_24:.*]] = arith.subi %[[VAL_12]], %[[VAL_2]] : index
! PostOpt:         br ^bb1(%[[VAL_23]], %[[VAL_24]] : index, index)
! PostOpt:       ^bb5:
! PostOpt:         %[[VAL_25:.*]] = fir.slice %[[VAL_2]], %[[VAL_1]], %[[VAL_2]], %[[VAL_5]], %[[VAL_6]], %[[VAL_2]] : (index, index, index, index, index, index) -> !fir.slice<2>
! PostOpt:         br ^bb6(%[[VAL_4]], %[[VAL_3]] : index, index)
! PostOpt:       ^bb6(%[[VAL_26:.*]]: index, %[[VAL_27:.*]]: index):
! PostOpt:         %[[VAL_28:.*]] = arith.cmpi sgt, %[[VAL_27]], %[[VAL_4]] : index
! PostOpt:         cond_br %[[VAL_28]], ^bb7(%[[VAL_4]], %[[VAL_1]] : index, index), ^bb10(%[[VAL_4]], %[[VAL_3]] : index, index)
! PostOpt:       ^bb7(%[[VAL_29:.*]]: index, %[[VAL_30:.*]]: index):
! PostOpt:         %[[VAL_31:.*]] = arith.cmpi sgt, %[[VAL_30]], %[[VAL_4]] : index
! PostOpt:         cond_br %[[VAL_31]], ^bb8, ^bb9
! PostOpt:       ^bb8:
! PostOpt:         %[[VAL_32:.*]] = arith.addi %[[VAL_29]], %[[VAL_2]] : index
! PostOpt:         %[[VAL_33:.*]] = arith.addi %[[VAL_26]], %[[VAL_2]] : index
! PostOpt:         %[[VAL_34:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_8]]) {{\[}}%[[VAL_25]]] %[[VAL_32]], %[[VAL_33]] : (!fir.ref<!fir.array<10x?xi32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<i32>
! PostOpt:         %[[VAL_35:.*]] = fir.load %[[VAL_34]] : !fir.ref<i32>
! PostOpt:         %[[VAL_36:.*]] = fir.array_coor %[[VAL_10]](%[[VAL_8]]) %[[VAL_32]], %[[VAL_33]] : (!fir.heap<!fir.array<10x?xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
! PostOpt:         fir.store %[[VAL_35]] to %[[VAL_36]] : !fir.ref<i32>
! PostOpt:         %[[VAL_37:.*]] = arith.subi %[[VAL_30]], %[[VAL_2]] : index
! PostOpt:         br ^bb7(%[[VAL_32]], %[[VAL_37]] : index, index)
! PostOpt:       ^bb9:
! PostOpt:         %[[VAL_38:.*]] = arith.addi %[[VAL_26]], %[[VAL_2]] : index
! PostOpt:         %[[VAL_39:.*]] = arith.subi %[[VAL_27]], %[[VAL_2]] : index
! PostOpt:         br ^bb6(%[[VAL_38]], %[[VAL_39]] : index, index)
! PostOpt:       ^bb10(%[[VAL_40:.*]]: index, %[[VAL_41:.*]]: index):
! PostOpt:         %[[VAL_42:.*]] = arith.cmpi sgt, %[[VAL_41]], %[[VAL_4]] : index
! PostOpt:         cond_br %[[VAL_42]], ^bb11(%[[VAL_4]], %[[VAL_1]] : index, index), ^bb14
! PostOpt:       ^bb11(%[[VAL_43:.*]]: index, %[[VAL_44:.*]]: index):
! PostOpt:         %[[VAL_45:.*]] = arith.cmpi sgt, %[[VAL_44]], %[[VAL_4]] : index
! PostOpt:         cond_br %[[VAL_45]], ^bb12, ^bb13
! PostOpt:       ^bb12:
! PostOpt:         %[[VAL_46:.*]] = arith.addi %[[VAL_43]], %[[VAL_2]] : index
! PostOpt:         %[[VAL_47:.*]] = arith.addi %[[VAL_40]], %[[VAL_2]] : index
! PostOpt:         %[[VAL_48:.*]] = fir.array_coor %[[VAL_10]](%[[VAL_8]]) %[[VAL_46]], %[[VAL_47]] : (!fir.heap<!fir.array<10x?xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
! PostOpt:         %[[VAL_49:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_8]]) {{\[}}%[[VAL_9]]] %[[VAL_46]], %[[VAL_47]] : (!fir.ref<!fir.array<10x?xi32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<i32>
! PostOpt:         %[[VAL_50:.*]] = fir.load %[[VAL_48]] : !fir.ref<i32>
! PostOpt:         fir.store %[[VAL_50]] to %[[VAL_49]] : !fir.ref<i32>
! PostOpt:         %[[VAL_51:.*]] = arith.subi %[[VAL_44]], %[[VAL_2]] : index
! PostOpt:         br ^bb11(%[[VAL_46]], %[[VAL_51]] : index, index)
! PostOpt:       ^bb13:
! PostOpt:         %[[VAL_52:.*]] = arith.addi %[[VAL_40]], %[[VAL_2]] : index
! PostOpt:         %[[VAL_53:.*]] = arith.subi %[[VAL_41]], %[[VAL_2]] : index
! PostOpt:         br ^bb10(%[[VAL_52]], %[[VAL_53]] : index, index)
! PostOpt:       ^bb14:
! PostOpt:         fir.freemem %[[VAL_10]] : !fir.heap<!fir.array<10x?xi32>>
! PostOpt:         return
! PostOpt:       }

! PostOpt-LABEL: func @_QPassumed_size_forall_test(
! PostOpt-SAME:        %[[VAL_0:.*]]: !fir.ref<!fir.array<10x?xi32>>{{.*}}) {
! PostOpt-DAG:         %[[VAL_1:.*]] = arith.constant 3 : index
! PostOpt-DAG:         %[[VAL_2:.*]] = arith.constant 10 : index
! PostOpt-DAG:         %[[VAL_3:.*]] = arith.constant 2 : index
! PostOpt-DAG:         %[[VAL_4:.*]] = arith.constant 1 : index
! PostOpt-DAG:         %[[VAL_5:.*]] = arith.constant 0 : index
! PostOpt-DAG:         %[[VAL_6:.*]] = arith.constant 5 : index
! PostOpt:         %[[VAL_7:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! PostOpt:         %[[VAL_8:.*]] = fir.undefined index
! PostOpt:         %[[VAL_9:.*]] = fir.shape %[[VAL_2]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
! PostOpt:         %[[VAL_10:.*]] = fir.allocmem !fir.array<10x?xi32>, %[[VAL_4]]
! PostOpt:         br ^bb1(%[[VAL_5]], %[[VAL_4]] : index, index)
! PostOpt:       ^bb1(%[[VAL_11:.*]]: index, %[[VAL_12:.*]]: index):
! PostOpt:         %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[VAL_5]] : index
! PostOpt:         cond_br %[[VAL_13]], ^bb2(%[[VAL_5]], %[[VAL_2]] : index, index), ^bb5
! PostOpt:       ^bb2(%[[VAL_14:.*]]: index, %[[VAL_15:.*]]: index):
! PostOpt:         %[[VAL_16:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[VAL_5]] : index
! PostOpt:         cond_br %[[VAL_16]], ^bb3, ^bb4
! PostOpt:       ^bb3:
! PostOpt:         %[[VAL_17:.*]] = arith.addi %[[VAL_14]], %[[VAL_4]] : index
! PostOpt:         %[[VAL_18:.*]] = arith.addi %[[VAL_11]], %[[VAL_4]] : index
! PostOpt:         %[[VAL_19:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_9]]) %[[VAL_17]], %[[VAL_18]] : (!fir.ref<!fir.array<10x?xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
! PostOpt:         %[[VAL_20:.*]] = fir.array_coor %[[VAL_10]](%[[VAL_9]]) %[[VAL_17]], %[[VAL_18]] : (!fir.heap<!fir.array<10x?xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
! PostOpt:         %[[VAL_21:.*]] = fir.load %[[VAL_19]] : !fir.ref<i32>
! PostOpt:         fir.store %[[VAL_21]] to %[[VAL_20]] : !fir.ref<i32>
! PostOpt:         %[[VAL_22:.*]] = arith.subi %[[VAL_15]], %[[VAL_4]] : index
! PostOpt:         br ^bb2(%[[VAL_17]], %[[VAL_22]] : index, index)
! PostOpt:       ^bb4:
! PostOpt:         %[[VAL_23:.*]] = arith.addi %[[VAL_11]], %[[VAL_4]] : index
! PostOpt:         %[[VAL_24:.*]] = arith.subi %[[VAL_12]], %[[VAL_4]] : index
! PostOpt:         br ^bb1(%[[VAL_23]], %[[VAL_24]] : index, index)
! PostOpt:       ^bb5:
! PostOpt:         br ^bb6(%[[VAL_3]], %[[VAL_6]] : index, index)
! PostOpt:       ^bb6(%[[VAL_25:.*]]: index, %[[VAL_26:.*]]: index):
! PostOpt:         %[[VAL_27:.*]] = arith.cmpi sgt, %[[VAL_26]], %[[VAL_5]] : index
! PostOpt:         cond_br %[[VAL_27]], ^bb7, ^bb11(%[[VAL_5]], %[[VAL_4]] : index, index)
! PostOpt:       ^bb7:
! PostOpt:         %[[VAL_28:.*]] = fir.convert %[[VAL_25]] : (index) -> i32
! PostOpt:         fir.store %[[VAL_28]] to %[[VAL_7]] : !fir.ref<i32>
! PostOpt:         %[[VAL_29:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
! PostOpt:         %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> index
! PostOpt:         br ^bb8(%[[VAL_5]], %[[VAL_3]] : index, index)
! PostOpt:       ^bb8(%[[VAL_31:.*]]: index, %[[VAL_32:.*]]: index):
! PostOpt:         %[[VAL_33:.*]] = arith.cmpi sgt, %[[VAL_32]], %[[VAL_5]] : index
! PostOpt:         cond_br %[[VAL_33]], ^bb9, ^bb10
! PostOpt:       ^bb9:
! PostOpt:         %[[VAL_34:.*]] = arith.addi %[[VAL_31]], %[[VAL_1]] : index
! PostOpt:         %[[VAL_35:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_9]]) %[[VAL_30]], %[[VAL_34]] : (!fir.ref<!fir.array<10x?xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
! PostOpt:         %[[VAL_36:.*]] = fir.load %[[VAL_35]] : !fir.ref<i32>
! PostOpt:         %[[VAL_37:.*]] = arith.addi %[[VAL_31]], %[[VAL_4]] : index
! PostOpt:         %[[VAL_38:.*]] = fir.array_coor %[[VAL_10]](%[[VAL_9]]) %[[VAL_30]], %[[VAL_37]] : (!fir.heap<!fir.array<10x?xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
! PostOpt:         fir.store %[[VAL_36]] to %[[VAL_38]] : !fir.ref<i32>
! PostOpt:         %[[VAL_39:.*]] = arith.subi %[[VAL_32]], %[[VAL_4]] : index
! PostOpt:         br ^bb8(%[[VAL_37]], %[[VAL_39]] : index, index)
! PostOpt:       ^bb10:
! PostOpt:         %[[VAL_40:.*]] = arith.addi %[[VAL_25]], %[[VAL_4]] : index
! PostOpt:         %[[VAL_41:.*]] = arith.subi %[[VAL_26]], %[[VAL_4]] : index
! PostOpt:         br ^bb6(%[[VAL_40]], %[[VAL_41]] : index, index)
! PostOpt:       ^bb11(%[[VAL_42:.*]]: index, %[[VAL_43:.*]]: index):
! PostOpt:         %[[VAL_44:.*]] = arith.cmpi sgt, %[[VAL_43]], %[[VAL_5]] : index
! PostOpt:         cond_br %[[VAL_44]], ^bb12(%[[VAL_5]], %[[VAL_2]] : index, index), ^bb15
! PostOpt:       ^bb12(%[[VAL_45:.*]]: index, %[[VAL_46:.*]]: index):
! PostOpt:         %[[VAL_47:.*]] = arith.cmpi sgt, %[[VAL_46]], %[[VAL_5]] : index
! PostOpt:         cond_br %[[VAL_47]], ^bb13, ^bb14
! PostOpt:       ^bb13:
! PostOpt:         %[[VAL_48:.*]] = arith.addi %[[VAL_45]], %[[VAL_4]] : index
! PostOpt:         %[[VAL_49:.*]] = arith.addi %[[VAL_42]], %[[VAL_4]] : index
! PostOpt:         %[[VAL_50:.*]] = fir.array_coor %[[VAL_10]](%[[VAL_9]]) %[[VAL_48]], %[[VAL_49]] : (!fir.heap<!fir.array<10x?xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
! PostOpt:         %[[VAL_51:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_9]]) %[[VAL_48]], %[[VAL_49]] : (!fir.ref<!fir.array<10x?xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
! PostOpt:         %[[VAL_52:.*]] = fir.load %[[VAL_50]] : !fir.ref<i32>
! PostOpt:         fir.store %[[VAL_52]] to %[[VAL_51]] : !fir.ref<i32>
! PostOpt:         %[[VAL_53:.*]] = arith.subi %[[VAL_46]], %[[VAL_4]] : index
! PostOpt:         br ^bb12(%[[VAL_48]], %[[VAL_53]] : index, index)
! PostOpt:       ^bb14:
! PostOpt:         %[[VAL_54:.*]] = arith.addi %[[VAL_42]], %[[VAL_4]] : index
! PostOpt:         %[[VAL_55:.*]] = arith.subi %[[VAL_43]], %[[VAL_4]] : index
! PostOpt:         br ^bb11(%[[VAL_54]], %[[VAL_55]] : index, index)
! PostOpt:       ^bb15:
! PostOpt:         fir.freemem %[[VAL_10]] : !fir.heap<!fir.array<10x?xi32>>
! PostOpt:         return
! PostOpt:       }
