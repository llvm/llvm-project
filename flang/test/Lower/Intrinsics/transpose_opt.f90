! RUN: bbc -emit-fir %s -opt-transpose=true -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtranspose_test(
! CHECK-SAME:                               %[[VAL_0:.*]]: !fir.ref<!fir.array<2x3xf32>> {fir.bindc_name = "mat"}) {
subroutine transpose_test(mat)
   real :: mat(2,3)
   call bar_transpose_test(transpose(mat))
! CHECK:         %[[VAL_1:.*]] = arith.constant 2 : index
! CHECK:         %[[VAL_2:.*]] = arith.constant 3 : index
! CHECK:         %[[VAL_3:.*]] = arith.constant 3 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 2 : index
! CHECK:         %[[VAL_5:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_6:.*]] = fir.array_load %[[VAL_0]](%[[VAL_5]]) : (!fir.ref<!fir.array<2x3xf32>>, !fir.shape<2>) -> !fir.array<2x3xf32>
! CHECK:         %[[VAL_7:.*]] = fir.allocmem !fir.array<3x2xf32>
! CHECK:         %[[VAL_8:.*]] = fir.shape %[[VAL_3]], %[[VAL_4]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_7]](%[[VAL_8]]) : (!fir.heap<!fir.array<3x2xf32>>, !fir.shape<2>) -> !fir.array<3x2xf32>
! CHECK:         %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_11:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_12:.*]] = arith.subi %[[VAL_3]], %[[VAL_10]] : index
! CHECK:         %[[VAL_13:.*]] = arith.subi %[[VAL_4]], %[[VAL_10]] : index
! CHECK:         %[[VAL_14:.*]] = fir.do_loop %[[VAL_15:.*]] = %[[VAL_11]] to %[[VAL_13]] step %[[VAL_10]] unordered iter_args(%[[VAL_16:.*]] = %[[VAL_9]]) -> (!fir.array<3x2xf32>) {
! CHECK:           %[[VAL_17:.*]] = fir.do_loop %[[VAL_18:.*]] = %[[VAL_11]] to %[[VAL_12]] step %[[VAL_10]] unordered iter_args(%[[VAL_19:.*]] = %[[VAL_16]]) -> (!fir.array<3x2xf32>) {
! CHECK:             %[[VAL_20:.*]] = fir.array_fetch %[[VAL_6]], %[[VAL_15]], %[[VAL_18]] : (!fir.array<2x3xf32>, index, index) -> f32
! CHECK:             %[[VAL_21:.*]] = fir.array_update %[[VAL_19]], %[[VAL_20]], %[[VAL_18]], %[[VAL_15]] : (!fir.array<3x2xf32>, f32, index, index) -> !fir.array<3x2xf32>
! CHECK:             fir.result %[[VAL_21]] : !fir.array<3x2xf32>
! CHECK:           }
! CHECK:           fir.result %[[VAL_22:.*]] : !fir.array<3x2xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_9]], %[[VAL_23:.*]] to %[[VAL_7]] : !fir.array<3x2xf32>, !fir.array<3x2xf32>, !fir.heap<!fir.array<3x2xf32>>
! CHECK:         %[[VAL_24:.*]] = fir.convert %[[VAL_7]] : (!fir.heap<!fir.array<3x2xf32>>) -> !fir.ref<!fir.array<3x2xf32>>
! CHECK:         fir.call @_QPbar_transpose_test(%[[VAL_24]]) : (!fir.ref<!fir.array<3x2xf32>>) -> ()
! CHECK:         fir.freemem %[[VAL_7]] : !fir.heap<!fir.array<3x2xf32>>
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func.func @_QPtranspose_allocatable_test(
! CHECK-SAME:                                           %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>> {fir.bindc_name = "mat"}) {
subroutine transpose_allocatable_test(mat)
  real, allocatable :: mat(:,:)
  mat = transpose(mat)
! CHECK:         %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:         %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_3:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_4]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_6:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
! CHECK:         %[[VAL_7:.*]] = fir.shape_shift %[[VAL_3]]#0, %[[VAL_3]]#1, %[[VAL_5]]#0, %[[VAL_5]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_6]](%[[VAL_7]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
! CHECK:         %[[VAL_9:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:         %[[VAL_10:.*]] = fir.box_addr %[[VAL_9]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.heap<!fir.array<?x?xf32>>) -> i64
! CHECK:         %[[VAL_12:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_13:.*]] = arith.cmpi ne, %[[VAL_11]], %[[VAL_12]] : i64
! CHECK:         %[[VAL_14:.*]]:2 = fir.if %[[VAL_13]] -> (i1, !fir.heap<!fir.array<?x?xf32>>) {
! CHECK:           %[[VAL_15:.*]] = arith.constant false
! CHECK:           %[[VAL_16:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_9]], %[[VAL_16]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_18:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_19:.*]]:3 = fir.box_dims %[[VAL_9]], %[[VAL_18]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_20:.*]] = arith.cmpi ne, %[[VAL_17]]#1, %[[VAL_5]]#1 : index
! CHECK:           %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_20]], %[[VAL_15]] : i1
! CHECK:           %[[VAL_22:.*]] = arith.cmpi ne, %[[VAL_19]]#1, %[[VAL_3]]#1 : index
! CHECK:           %[[VAL_23:.*]] = arith.select %[[VAL_22]], %[[VAL_22]], %[[VAL_21]] : i1
! CHECK:           %[[VAL_24:.*]] = fir.if %[[VAL_23]] -> (!fir.heap<!fir.array<?x?xf32>>) {
! CHECK:             %[[VAL_25:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_5]]#1, %[[VAL_3]]#1 {uniq_name = ".auto.alloc"}
! CHECK:             %[[VAL_26:.*]] = fir.shape %[[VAL_5]]#1, %[[VAL_3]]#1 : (index, index) -> !fir.shape<2>
! CHECK:             %[[VAL_27:.*]] = fir.array_load %[[VAL_25]](%[[VAL_26]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
! CHECK:             %[[VAL_28:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_29:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_30:.*]] = arith.subi %[[VAL_5]]#1, %[[VAL_28]] : index
! CHECK:             %[[VAL_31:.*]] = arith.subi %[[VAL_3]]#1, %[[VAL_28]] : index
! CHECK:             %[[VAL_32:.*]] = fir.do_loop %[[VAL_33:.*]] = %[[VAL_29]] to %[[VAL_31]] step %[[VAL_28]] unordered iter_args(%[[VAL_34:.*]] = %[[VAL_27]]) -> (!fir.array<?x?xf32>) {
! CHECK:               %[[VAL_35:.*]] = fir.do_loop %[[VAL_36:.*]] = %[[VAL_29]] to %[[VAL_30]] step %[[VAL_28]] unordered iter_args(%[[VAL_37:.*]] = %[[VAL_34]]) -> (!fir.array<?x?xf32>) {
! CHECK:                 %[[VAL_38:.*]] = fir.array_fetch %[[VAL_8]], %[[VAL_33]], %[[VAL_36]] : (!fir.array<?x?xf32>, index, index) -> f32
! CHECK:                 %[[VAL_39:.*]] = fir.array_update %[[VAL_37]], %[[VAL_38]], %[[VAL_36]], %[[VAL_33]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
! CHECK:                 fir.result %[[VAL_39]] : !fir.array<?x?xf32>
! CHECK:               }
! CHECK:               fir.result %[[VAL_40:.*]] : !fir.array<?x?xf32>
! CHECK:             }
! CHECK:             fir.array_merge_store %[[VAL_27]], %[[VAL_41:.*]] to %[[VAL_25]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>
! CHECK:             fir.result %[[VAL_25]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:           } else {
! CHECK:             %[[VAL_42:.*]] = fir.shape %[[VAL_5]]#1, %[[VAL_3]]#1 : (index, index) -> !fir.shape<2>
! CHECK:             %[[VAL_43:.*]] = fir.array_load %[[VAL_10]](%[[VAL_42]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
! CHECK:             %[[VAL_44:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_45:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_46:.*]] = arith.subi %[[VAL_5]]#1, %[[VAL_44]] : index
! CHECK:             %[[VAL_47:.*]] = arith.subi %[[VAL_3]]#1, %[[VAL_44]] : index
! CHECK:             %[[VAL_48:.*]] = fir.do_loop %[[VAL_49:.*]] = %[[VAL_45]] to %[[VAL_47]] step %[[VAL_44]] unordered iter_args(%[[VAL_50:.*]] = %[[VAL_43]]) -> (!fir.array<?x?xf32>) {
! CHECK:               %[[VAL_51:.*]] = fir.do_loop %[[VAL_52:.*]] = %[[VAL_45]] to %[[VAL_46]] step %[[VAL_44]] unordered iter_args(%[[VAL_53:.*]] = %[[VAL_50]]) -> (!fir.array<?x?xf32>) {
! CHECK:                 %[[VAL_54:.*]] = fir.array_fetch %[[VAL_8]], %[[VAL_49]], %[[VAL_52]] : (!fir.array<?x?xf32>, index, index) -> f32
! CHECK:                 %[[VAL_55:.*]] = fir.array_update %[[VAL_53]], %[[VAL_54]], %[[VAL_52]], %[[VAL_49]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
! CHECK:                 fir.result %[[VAL_55]] : !fir.array<?x?xf32>
! CHECK:               }
! CHECK:               fir.result %[[VAL_56:.*]] : !fir.array<?x?xf32>
! CHECK:             }
! CHECK:             fir.array_merge_store %[[VAL_43]], %[[VAL_57:.*]] to %[[VAL_10]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>
! CHECK:             fir.result %[[VAL_10]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_23]], %[[VAL_58:.*]] : i1, !fir.heap<!fir.array<?x?xf32>>
! CHECK:         } else {
! CHECK:           %[[VAL_59:.*]] = arith.constant true
! CHECK:           %[[VAL_60:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_5]]#1, %[[VAL_3]]#1 {uniq_name = ".auto.alloc"}
! CHECK:           %[[VAL_61:.*]] = fir.shape %[[VAL_5]]#1, %[[VAL_3]]#1 : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_62:.*]] = fir.array_load %[[VAL_60]](%[[VAL_61]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
! CHECK:           %[[VAL_63:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_64:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_65:.*]] = arith.subi %[[VAL_5]]#1, %[[VAL_63]] : index
! CHECK:           %[[VAL_66:.*]] = arith.subi %[[VAL_3]]#1, %[[VAL_63]] : index
! CHECK:           %[[VAL_67:.*]] = fir.do_loop %[[VAL_68:.*]] = %[[VAL_64]] to %[[VAL_66]] step %[[VAL_63]] unordered iter_args(%[[VAL_69:.*]] = %[[VAL_62]]) -> (!fir.array<?x?xf32>) {
! CHECK:             %[[VAL_70:.*]] = fir.do_loop %[[VAL_71:.*]] = %[[VAL_64]] to %[[VAL_65]] step %[[VAL_63]] unordered iter_args(%[[VAL_72:.*]] = %[[VAL_69]]) -> (!fir.array<?x?xf32>) {
! CHECK:               %[[VAL_73:.*]] = fir.array_fetch %[[VAL_8]], %[[VAL_68]], %[[VAL_71]] : (!fir.array<?x?xf32>, index, index) -> f32
! CHECK:               %[[VAL_74:.*]] = fir.array_update %[[VAL_72]], %[[VAL_73]], %[[VAL_71]], %[[VAL_68]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
! CHECK:               fir.result %[[VAL_74]] : !fir.array<?x?xf32>
! CHECK:             }
! CHECK:             fir.result %[[VAL_75:.*]] : !fir.array<?x?xf32>
! CHECK:           }
! CHECK:           fir.array_merge_store %[[VAL_62]], %[[VAL_76:.*]] to %[[VAL_60]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>
! CHECK:           fir.result %[[VAL_59]], %[[VAL_60]] : i1, !fir.heap<!fir.array<?x?xf32>>
! CHECK:         }
! CHECK:         fir.if %[[VAL_77:.*]]#0 {
! CHECK:           fir.if %[[VAL_13]] {
! CHECK:             fir.freemem %[[VAL_10]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:           }
! CHECK:           %[[VAL_78:.*]] = fir.shape %[[VAL_5]]#1, %[[VAL_3]]#1 : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_79:.*]] = fir.embox %[[VAL_77]]#1(%[[VAL_78]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
! CHECK:           fir.store %[[VAL_79]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:         }
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK:       func.func private @_QPbar_transpose_test(!fir.ref<!fir.array<3x2xf32>>)
