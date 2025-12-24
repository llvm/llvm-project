! RUN: bbc --emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: bbc -hlfir=false -fwrapv %s -o - | FileCheck --check-prefix=PostOpt %s


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
! CHECK:         %[[VAL_2:.*]] = fir.assumed_size_extent : index
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
! CHECK:         %[[VAL_3:.*]] = fir.assumed_size_extent : index
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

! PostOpt-LABEL:   func.func @_QPassumed_size_test(
! PostOpt-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<10x?xi32>> {fir.bindc_name = "a"}) {
! PostOpt:           %[[CONSTANT_0:.*]] = arith.constant 4 : index
! PostOpt:           %[[CONSTANT_1:.*]] = arith.constant 3 : index
! PostOpt:           %[[CONSTANT_2:.*]] = arith.constant 2 : index
! PostOpt:           %[[CONSTANT_3:.*]] = arith.constant 1 : index
! PostOpt:           %[[CONSTANT_4:.*]] = arith.constant 0 : index
! PostOpt:           %[[CONSTANT_5:.*]] = arith.constant 10 : index
! PostOpt:           %[[ASSUMED_SIZE_EXTENT_0:.*]] = fir.assumed_size_extent : index
! PostOpt:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_5]], %[[ASSUMED_SIZE_EXTENT_0]] : (index, index) -> !fir.shape<2>
! PostOpt:           %[[SLICE_0:.*]] = fir.slice %[[CONSTANT_3]], %[[CONSTANT_5]], %[[CONSTANT_3]], %[[CONSTANT_3]], %[[CONSTANT_2]], %[[CONSTANT_3]] : (index, index, index, index, index, index) -> !fir.slice<2>
! PostOpt:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.array<10x?xi32>, %[[CONSTANT_2]]
! PostOpt:           cf.br ^bb1(%[[CONSTANT_4]], %[[CONSTANT_2]] : index, index)
! PostOpt:         ^bb1(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index):
! PostOpt:           %[[CMPI_0:.*]] = arith.cmpi sgt, %[[VAL_1]], %[[CONSTANT_4]] : index
! PostOpt:           cf.cond_br %[[CMPI_0]], ^bb2, ^bb6
! PostOpt:         ^bb2:
! PostOpt:           %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[CONSTANT_3]] : index
! PostOpt:           cf.br ^bb3(%[[CONSTANT_4]], %[[CONSTANT_5]] : index, index)
! PostOpt:         ^bb3(%[[VAL_2:.*]]: index, %[[VAL_3:.*]]: index):
! PostOpt:           %[[CMPI_1:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[CONSTANT_4]] : index
! PostOpt:           cf.cond_br %[[CMPI_1]], ^bb4, ^bb5
! PostOpt:         ^bb4:
! PostOpt:           %[[ADDI_1:.*]] = arith.addi %[[VAL_2]], %[[CONSTANT_3]] : index
! PostOpt:           %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[ARG0]](%[[SHAPE_0]]) {{\[}}%[[SLICE_0]]] %[[ADDI_1]], %[[ADDI_0]] : (!fir.ref<!fir.array<10x?xi32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<i32>
! PostOpt:           %[[ARRAY_COOR_1:.*]] = fir.array_coor %[[ALLOCMEM_0]](%[[SHAPE_0]]) %[[ADDI_1]], %[[ADDI_0]] : (!fir.heap<!fir.array<10x?xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
! PostOpt:           %[[LOAD_0:.*]] = fir.load %[[ARRAY_COOR_0]] : !fir.ref<i32>
! PostOpt:           fir.store %[[LOAD_0]] to %[[ARRAY_COOR_1]] : !fir.ref<i32>
! PostOpt:           %[[SUBI_0:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_3]] : index
! PostOpt:           cf.br ^bb3(%[[ADDI_1]], %[[SUBI_0]] : index, index)
! PostOpt:         ^bb5:
! PostOpt:           %[[SUBI_1:.*]] = arith.subi %[[VAL_1]], %[[CONSTANT_3]] : index
! PostOpt:           cf.br ^bb1(%[[ADDI_0]], %[[SUBI_1]] : index, index)
! PostOpt:         ^bb6:
! PostOpt:           %[[SLICE_1:.*]] = fir.slice %[[CONSTANT_3]], %[[CONSTANT_5]], %[[CONSTANT_3]], %[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_3]] : (index, index, index, index, index, index) -> !fir.slice<2>
! PostOpt:           cf.br ^bb7(%[[CONSTANT_4]], %[[CONSTANT_2]] : index, index)
! PostOpt:         ^bb7(%[[VAL_4:.*]]: index, %[[VAL_5:.*]]: index):
! PostOpt:           %[[CMPI_2:.*]] = arith.cmpi sgt, %[[VAL_5]], %[[CONSTANT_4]] : index
! PostOpt:           cf.cond_br %[[CMPI_2]], ^bb8, ^bb12(%[[CONSTANT_4]], %[[CONSTANT_2]] : index, index)
! PostOpt:         ^bb8:
! PostOpt:           %[[ADDI_2:.*]] = arith.addi %[[VAL_4]], %[[CONSTANT_3]] : index
! PostOpt:           cf.br ^bb9(%[[CONSTANT_4]], %[[CONSTANT_5]] : index, index)
! PostOpt:         ^bb9(%[[VAL_6:.*]]: index, %[[VAL_7:.*]]: index):
! PostOpt:           %[[CMPI_3:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[CONSTANT_4]] : index
! PostOpt:           cf.cond_br %[[CMPI_3]], ^bb10, ^bb11
! PostOpt:         ^bb10:
! PostOpt:           %[[ADDI_3:.*]] = arith.addi %[[VAL_6]], %[[CONSTANT_3]] : index
! PostOpt:           %[[ARRAY_COOR_2:.*]] = fir.array_coor %[[ARG0]](%[[SHAPE_0]]) {{\[}}%[[SLICE_1]]] %[[ADDI_3]], %[[ADDI_2]] : (!fir.ref<!fir.array<10x?xi32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<i32>
! PostOpt:           %[[LOAD_1:.*]] = fir.load %[[ARRAY_COOR_2]] : !fir.ref<i32>
! PostOpt:           %[[ARRAY_COOR_3:.*]] = fir.array_coor %[[ALLOCMEM_0]](%[[SHAPE_0]]) %[[ADDI_3]], %[[ADDI_2]] : (!fir.heap<!fir.array<10x?xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
! PostOpt:           fir.store %[[LOAD_1]] to %[[ARRAY_COOR_3]] : !fir.ref<i32>
! PostOpt:           %[[SUBI_2:.*]] = arith.subi %[[VAL_7]], %[[CONSTANT_3]] : index
! PostOpt:           cf.br ^bb9(%[[ADDI_3]], %[[SUBI_2]] : index, index)
! PostOpt:         ^bb11:
! PostOpt:           %[[SUBI_3:.*]] = arith.subi %[[VAL_5]], %[[CONSTANT_3]] : index
! PostOpt:           cf.br ^bb7(%[[ADDI_2]], %[[SUBI_3]] : index, index)
! PostOpt:         ^bb12(%[[VAL_8:.*]]: index, %[[VAL_9:.*]]: index):
! PostOpt:           %[[CMPI_4:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[CONSTANT_4]] : index
! PostOpt:           cf.cond_br %[[CMPI_4]], ^bb13, ^bb17
! PostOpt:         ^bb13:
! PostOpt:           %[[ADDI_4:.*]] = arith.addi %[[VAL_8]], %[[CONSTANT_3]] : index
! PostOpt:           cf.br ^bb14(%[[CONSTANT_4]], %[[CONSTANT_5]] : index, index)
! PostOpt:         ^bb14(%[[VAL_10:.*]]: index, %[[VAL_11:.*]]: index):
! PostOpt:           %[[CMPI_5:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[CONSTANT_4]] : index
! PostOpt:           cf.cond_br %[[CMPI_5]], ^bb15, ^bb16
! PostOpt:         ^bb15:
! PostOpt:           %[[ADDI_5:.*]] = arith.addi %[[VAL_10]], %[[CONSTANT_3]] : index
! PostOpt:           %[[ARRAY_COOR_4:.*]] = fir.array_coor %[[ALLOCMEM_0]](%[[SHAPE_0]]) %[[ADDI_5]], %[[ADDI_4]] : (!fir.heap<!fir.array<10x?xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
! PostOpt:           %[[ARRAY_COOR_5:.*]] = fir.array_coor %[[ARG0]](%[[SHAPE_0]]) {{\[}}%[[SLICE_0]]] %[[ADDI_5]], %[[ADDI_4]] : (!fir.ref<!fir.array<10x?xi32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<i32>
! PostOpt:           %[[LOAD_2:.*]] = fir.load %[[ARRAY_COOR_4]] : !fir.ref<i32>
! PostOpt:           fir.store %[[LOAD_2]] to %[[ARRAY_COOR_5]] : !fir.ref<i32>
! PostOpt:           %[[SUBI_4:.*]] = arith.subi %[[VAL_11]], %[[CONSTANT_3]] : index
! PostOpt:           cf.br ^bb14(%[[ADDI_5]], %[[SUBI_4]] : index, index)
! PostOpt:         ^bb16:
! PostOpt:           %[[SUBI_5:.*]] = arith.subi %[[VAL_9]], %[[CONSTANT_3]] : index
! PostOpt:           cf.br ^bb12(%[[ADDI_4]], %[[SUBI_5]] : index, index)
! PostOpt:         ^bb17:
! PostOpt:           fir.freemem %[[ALLOCMEM_0]] : !fir.heap<!fir.array<10x?xi32>>
! PostOpt:           return
! PostOpt:         }

! PostOpt-LABEL:   func.func @_QPassumed_size_forall_test(
! PostOpt-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<10x?xi32>> {fir.bindc_name = "b"}) {
! PostOpt:           %[[CONSTANT_0:.*]] = arith.constant 5 : index
! PostOpt:           %[[CONSTANT_1:.*]] = arith.constant 3 : index
! PostOpt:           %[[CONSTANT_2:.*]] = arith.constant 2 : index
! PostOpt:           %[[CONSTANT_3:.*]] = arith.constant 10 : index
! PostOpt:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
! PostOpt:           %[[CONSTANT_5:.*]] = arith.constant 0 : index
! PostOpt:           %[[ALLOCA_0:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! PostOpt:           %[[ASSUMED_SIZE_EXTENT_0:.*]] = fir.assumed_size_extent : index
! PostOpt:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_3]], %[[ASSUMED_SIZE_EXTENT_0]] : (index, index) -> !fir.shape<2>
! PostOpt:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.array<10x?xi32>, %[[CONSTANT_4]]
! PostOpt:           cf.br ^bb1(%[[CONSTANT_5]], %[[CONSTANT_4]] : index, index)
! PostOpt:         ^bb1(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index):
! PostOpt:           %[[CMPI_0:.*]] = arith.cmpi sgt, %[[VAL_1]], %[[CONSTANT_5]] : index
! PostOpt:           cf.cond_br %[[CMPI_0]], ^bb2, ^bb6
! PostOpt:         ^bb2:
! PostOpt:           %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[CONSTANT_4]] : index
! PostOpt:           cf.br ^bb3(%[[CONSTANT_5]], %[[CONSTANT_3]] : index, index)
! PostOpt:         ^bb3(%[[VAL_2:.*]]: index, %[[VAL_3:.*]]: index):
! PostOpt:           %[[CMPI_1:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[CONSTANT_5]] : index
! PostOpt:           cf.cond_br %[[CMPI_1]], ^bb4, ^bb5
! PostOpt:         ^bb4:
! PostOpt:           %[[ADDI_1:.*]] = arith.addi %[[VAL_2]], %[[CONSTANT_4]] : index
! PostOpt:           %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[ARG0]](%[[SHAPE_0]]) %[[ADDI_1]], %[[ADDI_0]] : (!fir.ref<!fir.array<10x?xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
! PostOpt:           %[[ARRAY_COOR_1:.*]] = fir.array_coor %[[ALLOCMEM_0]](%[[SHAPE_0]]) %[[ADDI_1]], %[[ADDI_0]] : (!fir.heap<!fir.array<10x?xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
! PostOpt:           %[[LOAD_0:.*]] = fir.load %[[ARRAY_COOR_0]] : !fir.ref<i32>
! PostOpt:           fir.store %[[LOAD_0]] to %[[ARRAY_COOR_1]] : !fir.ref<i32>
! PostOpt:           %[[SUBI_0:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_4]] : index
! PostOpt:           cf.br ^bb3(%[[ADDI_1]], %[[SUBI_0]] : index, index)
! PostOpt:         ^bb5:
! PostOpt:           %[[SUBI_1:.*]] = arith.subi %[[VAL_1]], %[[CONSTANT_4]] : index
! PostOpt:           cf.br ^bb1(%[[ADDI_0]], %[[SUBI_1]] : index, index)
! PostOpt:         ^bb6:
! PostOpt:           cf.br ^bb7(%[[CONSTANT_2]], %[[CONSTANT_0]] : index, index)
! PostOpt:         ^bb7(%[[VAL_4:.*]]: index, %[[VAL_5:.*]]: index):
! PostOpt:           %[[CMPI_2:.*]] = arith.cmpi sgt, %[[VAL_5]], %[[CONSTANT_5]] : index
! PostOpt:           cf.cond_br %[[CMPI_2]], ^bb8, ^bb12(%[[CONSTANT_5]], %[[CONSTANT_4]] : index, index)
! PostOpt:         ^bb8:
! PostOpt:           %[[CONVERT_0:.*]] = fir.convert %[[VAL_4]] : (index) -> i32
! PostOpt:           fir.store %[[CONVERT_0]] to %[[ALLOCA_0]] : !fir.ref<i32>
! PostOpt:           %[[LOAD_1:.*]] = fir.load %[[ALLOCA_0]] : !fir.ref<i32>
! PostOpt:           %[[CONVERT_1:.*]] = fir.convert %[[LOAD_1]] : (i32) -> index
! PostOpt:           cf.br ^bb9(%[[CONSTANT_5]], %[[CONSTANT_2]] : index, index)
! PostOpt:         ^bb9(%[[VAL_6:.*]]: index, %[[VAL_7:.*]]: index):
! PostOpt:           %[[CMPI_3:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[CONSTANT_5]] : index
! PostOpt:           cf.cond_br %[[CMPI_3]], ^bb10, ^bb11
! PostOpt:         ^bb10:
! PostOpt:           %[[ADDI_2:.*]] = arith.addi %[[VAL_6]], %[[CONSTANT_1]] : index
! PostOpt:           %[[ARRAY_COOR_2:.*]] = fir.array_coor %[[ARG0]](%[[SHAPE_0]]) %[[CONVERT_1]], %[[ADDI_2]] : (!fir.ref<!fir.array<10x?xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
! PostOpt:           %[[LOAD_2:.*]] = fir.load %[[ARRAY_COOR_2]] : !fir.ref<i32>
! PostOpt:           %[[ADDI_3:.*]] = arith.addi %[[VAL_6]], %[[CONSTANT_4]] : index
! PostOpt:           %[[ARRAY_COOR_3:.*]] = fir.array_coor %[[ALLOCMEM_0]](%[[SHAPE_0]]) %[[CONVERT_1]], %[[ADDI_3]] : (!fir.heap<!fir.array<10x?xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
! PostOpt:           fir.store %[[LOAD_2]] to %[[ARRAY_COOR_3]] : !fir.ref<i32>
! PostOpt:           %[[SUBI_2:.*]] = arith.subi %[[VAL_7]], %[[CONSTANT_4]] : index
! PostOpt:           cf.br ^bb9(%[[ADDI_3]], %[[SUBI_2]] : index, index)
! PostOpt:         ^bb11:
! PostOpt:           %[[ADDI_4:.*]] = arith.addi %[[VAL_4]], %[[CONSTANT_4]] : index
! PostOpt:           %[[SUBI_3:.*]] = arith.subi %[[VAL_5]], %[[CONSTANT_4]] : index
! PostOpt:           cf.br ^bb7(%[[ADDI_4]], %[[SUBI_3]] : index, index)
! PostOpt:         ^bb12(%[[VAL_8:.*]]: index, %[[VAL_9:.*]]: index):
! PostOpt:           %[[CMPI_4:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[CONSTANT_5]] : index
! PostOpt:           cf.cond_br %[[CMPI_4]], ^bb13, ^bb17
! PostOpt:         ^bb13:
! PostOpt:           %[[ADDI_5:.*]] = arith.addi %[[VAL_8]], %[[CONSTANT_4]] : index
! PostOpt:           cf.br ^bb14(%[[CONSTANT_5]], %[[CONSTANT_3]] : index, index)
! PostOpt:         ^bb14(%[[VAL_10:.*]]: index, %[[VAL_11:.*]]: index):
! PostOpt:           %[[CMPI_5:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[CONSTANT_5]] : index
! PostOpt:           cf.cond_br %[[CMPI_5]], ^bb15, ^bb16
! PostOpt:         ^bb15:
! PostOpt:           %[[ADDI_6:.*]] = arith.addi %[[VAL_10]], %[[CONSTANT_4]] : index
! PostOpt:           %[[ARRAY_COOR_4:.*]] = fir.array_coor %[[ALLOCMEM_0]](%[[SHAPE_0]]) %[[ADDI_6]], %[[ADDI_5]] : (!fir.heap<!fir.array<10x?xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
! PostOpt:           %[[ARRAY_COOR_5:.*]] = fir.array_coor %[[ARG0]](%[[SHAPE_0]]) %[[ADDI_6]], %[[ADDI_5]] : (!fir.ref<!fir.array<10x?xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
! PostOpt:           %[[LOAD_3:.*]] = fir.load %[[ARRAY_COOR_4]] : !fir.ref<i32>
! PostOpt:           fir.store %[[LOAD_3]] to %[[ARRAY_COOR_5]] : !fir.ref<i32>
! PostOpt:           %[[SUBI_4:.*]] = arith.subi %[[VAL_11]], %[[CONSTANT_4]] : index
! PostOpt:           cf.br ^bb14(%[[ADDI_6]], %[[SUBI_4]] : index, index)
! PostOpt:         ^bb16:
! PostOpt:           %[[SUBI_5:.*]] = arith.subi %[[VAL_9]], %[[CONSTANT_4]] : index
! PostOpt:           cf.br ^bb12(%[[ADDI_5]], %[[SUBI_5]] : index, index)
! PostOpt:         ^bb17:
! PostOpt:           fir.freemem %[[ALLOCMEM_0]] : !fir.heap<!fir.array<10x?xi32>>
! PostOpt:           return
! PostOpt:         }
