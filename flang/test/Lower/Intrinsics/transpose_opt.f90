! RUN: bbc -emit-fir %s -opt-transpose=true -o - | FileCheck %s
! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -O1 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -O2 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -O3 %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtranspose_test(
! CHECK-SAME:                               %[[VAL_0:.*]]: !fir.ref<!fir.array<2x3xf32>> {fir.bindc_name = "mat"}) {
subroutine transpose_test(mat)
   real :: mat(2,3)
   call bar_transpose_test(transpose(mat))
! CHECK:         %[[VAL_6:.*]] = fir.array_load %[[VAL_0]](%{{.*}}) : (!fir.ref<!fir.array<2x3xf32>>, !fir.shape<2>) -> !fir.array<2x3xf32>
! CHECK:         %[[VAL_7:.*]] = fir.allocmem !fir.array<3x2xf32>
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_7]](%{{.*}}) : (!fir.heap<!fir.array<3x2xf32>>, !fir.shape<2>) -> !fir.array<3x2xf32>
! CHECK:         %[[VAL_14:.*]] = fir.do_loop %[[VAL_15:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%[[VAL_16:.*]] = %[[VAL_9]]) -> (!fir.array<3x2xf32>) {
! CHECK:           %[[VAL_17:.*]] = fir.do_loop %[[VAL_18:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%[[VAL_19:.*]] = %[[VAL_16]]) -> (!fir.array<3x2xf32>) {
! CHECK:             %[[VAL_20:.*]] = fir.array_fetch %[[VAL_6]], %[[VAL_15]], %[[VAL_18]] : (!fir.array<2x3xf32>, index, index) -> f32
! CHECK:             %[[VAL_21:.*]] = fir.array_update %[[VAL_19]], %[[VAL_20]], %[[VAL_18]], %[[VAL_15]] : (!fir.array<3x2xf32>, f32, index, index) -> !fir.array<3x2xf32>
! CHECK:             fir.result %[[VAL_21]] : !fir.array<3x2xf32>
! CHECK:           }
! CHECK:           fir.result %[[VAL_17]] : !fir.array<3x2xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_9]], %[[VAL_14]] to %[[VAL_7]] : !fir.array<3x2xf32>, !fir.array<3x2xf32>, !fir.heap<!fir.array<3x2xf32>>
! CHECK:         %[[VAL_24:.*]] = fir.convert %[[VAL_7]] : (!fir.heap<!fir.array<3x2xf32>>) -> !fir.ref<!fir.array<3x2xf32>>
! CHECK:         fir.call @_QPbar_transpose_test(%[[VAL_24]]) {{.*}}: (!fir.ref<!fir.array<3x2xf32>>) -> ()
! CHECK:         fir.freemem %[[VAL_7]] : !fir.heap<!fir.array<3x2xf32>>

! CHECK-NOT: @_FortranATranspose
end subroutine

! CHECK-LABEL: func.func @_QPtranspose_allocatable_test(
! CHECK-SAME:                                           %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>> {fir.bindc_name = "mat"}) {
subroutine transpose_allocatable_test(mat)
  real, allocatable :: mat(:,:)
  mat = transpose(mat)
! Verify that the "optimized" TRANSPOSE loops are generated
! three times in each branch checking the status of LHS allocatable.

! CHECK:         %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:         %[[VAL_6:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_6]](%{{.*}}) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>

! CHECK:         %[[VAL_9:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:         %[[VAL_10:.*]] = fir.box_addr %[[VAL_9]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>

! CHECK:         %[[VAL_14:.*]]:2 = fir.if %{{.*}} -> (i1, !fir.heap<!fir.array<?x?xf32>>) {

! CHECK:           %[[VAL_24:.*]] = fir.if %{{.*}} -> (!fir.heap<!fir.array<?x?xf32>>) {

! CHECK:             %[[VAL_25:.*]] = fir.allocmem !fir.array<?x?xf32>
! CHECK:             %[[VAL_27:.*]] = fir.array_load %[[VAL_25]](%{{.*}}) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>

! CHECK:             %[[VAL_32:.*]] = fir.do_loop %[[VAL_33:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%[[VAL_34:.*]] = %[[VAL_27]]) -> (!fir.array<?x?xf32>) {
! CHECK:               %[[VAL_35:.*]] = fir.do_loop %[[VAL_36:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%[[VAL_37:.*]] = %[[VAL_34]]) -> (!fir.array<?x?xf32>) {
! CHECK:                 %[[VAL_38:.*]] = fir.array_fetch %[[VAL_8]], %[[VAL_33]], %[[VAL_36]] : (!fir.array<?x?xf32>, index, index) -> f32
! CHECK:                 %[[VAL_39:.*]] = fir.array_update %[[VAL_37]], %[[VAL_38]], %[[VAL_36]], %[[VAL_33]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
! CHECK:                 fir.result %[[VAL_39]] : !fir.array<?x?xf32>
! CHECK:               }
! CHECK:               fir.result %[[VAL_35]] : !fir.array<?x?xf32>
! CHECK:             }
! CHECK:             fir.array_merge_store %[[VAL_27]], %[[VAL_32]] to %[[VAL_25]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>

! CHECK:           } else {

! CHECK:             %[[VAL_43:.*]] = fir.array_load %[[VAL_10]](%{{.*}}) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>

! CHECK:             %[[VAL_48:.*]] = fir.do_loop %[[VAL_49:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%[[VAL_50:.*]] = %[[VAL_43]]) -> (!fir.array<?x?xf32>) {
! CHECK:               %[[VAL_51:.*]] = fir.do_loop %[[VAL_52:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%[[VAL_53:.*]] = %[[VAL_50]]) -> (!fir.array<?x?xf32>) {
! CHECK:                 %[[VAL_54:.*]] = fir.array_fetch %[[VAL_8]], %[[VAL_49]], %[[VAL_52]] : (!fir.array<?x?xf32>, index, index) -> f32
! CHECK:                 %[[VAL_55:.*]] = fir.array_update %[[VAL_53]], %[[VAL_54]], %[[VAL_52]], %[[VAL_49]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
! CHECK:                 fir.result %[[VAL_55]] : !fir.array<?x?xf32>
! CHECK:               }
! CHECK:               fir.result %[[VAL_51]] : !fir.array<?x?xf32>
! CHECK:             }
! CHECK:             fir.array_merge_store %[[VAL_43]], %[[VAL_48]] to %[[VAL_10]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>

! CHECK:             fir.result %[[VAL_10]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:           }

! CHECK:         } else {

! CHECK:           %[[VAL_60:.*]] = fir.allocmem !fir.array<?x?xf32>
! CHECK:           %[[VAL_62:.*]] = fir.array_load %[[VAL_60]](%{{.*}}) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>

! CHECK:           %[[VAL_67:.*]] = fir.do_loop %[[VAL_68:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%[[VAL_69:.*]] = %[[VAL_62]]) -> (!fir.array<?x?xf32>) {
! CHECK:             %[[VAL_70:.*]] = fir.do_loop %[[VAL_71:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%[[VAL_72:.*]] = %[[VAL_69]]) -> (!fir.array<?x?xf32>) {
! CHECK:               %[[VAL_73:.*]] = fir.array_fetch %[[VAL_8]], %[[VAL_68]], %[[VAL_71]] : (!fir.array<?x?xf32>, index, index) -> f32
! CHECK:               %[[VAL_74:.*]] = fir.array_update %[[VAL_72]], %[[VAL_73]], %[[VAL_71]], %[[VAL_68]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
! CHECK:               fir.result %[[VAL_74]] : !fir.array<?x?xf32>
! CHECK:             }
! CHECK:             fir.result %[[VAL_70]] : !fir.array<?x?xf32>
! CHECK:           }
! CHECK:           fir.array_merge_store %[[VAL_62]], %[[VAL_67]] to %[[VAL_60]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>

! CHECK:         }

! CHECK-NOT: @_FortranATranspose
end subroutine
