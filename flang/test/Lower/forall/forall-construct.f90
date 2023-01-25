! Test forall lowering

! RUN: bbc -emit-fir %s -o - | FileCheck %s

!*** Test a FORALL construct
subroutine test_forall_construct(a,b)
  integer :: i, j
  real :: a(:,:), b(:,:)
  forall (i=1:ubound(a,1), j=1:ubound(a,2), b(j,i) > 0.0)
     a(i,j) = b(j,i) / 3.14
  end forall
end subroutine test_forall_construct

! CHECK-LABEL: func @_QPtest_forall_construct(
! CHECK-SAME:     %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf32>>{{.*}}, %[[VAL_1:.*]]: !fir.box<!fir.array<?x?xf32>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
! CHECK:         %[[VAL_6:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_6]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]]#1 : (index) -> i64
! CHECK:         %[[VAL_9:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (index) -> i64
! CHECK:         %[[VAL_11:.*]] = arith.addi %[[VAL_8]], %[[VAL_10]] : i64
! CHECK:         %[[VAL_12:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_13:.*]] = arith.subi %[[VAL_11]], %[[VAL_12]] : i64
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> i32
! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i32) -> index
! CHECK:         %[[VAL_16:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_17:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> index
! CHECK:         %[[VAL_19:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_20:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_19]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_20]]#1 : (index) -> i64
! CHECK:         %[[VAL_22:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (index) -> i64
! CHECK:         %[[VAL_24:.*]] = arith.addi %[[VAL_21]], %[[VAL_23]] : i64
! CHECK:         %[[VAL_25:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_26:.*]] = arith.subi %[[VAL_24]], %[[VAL_25]] : i64
! CHECK:         %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i64) -> i32
! CHECK:         %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> index
! CHECK:         %[[VAL_29:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_30:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.array<?x?xf32>
! CHECK:         %[[VAL_31:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.array<?x?xf32>
! CHECK:         %[[VAL_32:.*]] = fir.do_loop %[[VAL_33:.*]] = %[[VAL_5]] to %[[VAL_15]] step %[[VAL_16]] unordered iter_args(%[[VAL_34:.*]] = %[[VAL_30]]) -> (!fir.array<?x?xf32>) {
! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_33]] : (index) -> i32
! CHECK:           fir.store %[[VAL_35]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_36:.*]] = fir.do_loop %[[VAL_37:.*]] = %[[VAL_18]] to %[[VAL_28]] step %[[VAL_29]] unordered iter_args(%[[VAL_38:.*]] = %[[VAL_34]]) -> (!fir.array<?x?xf32>) {
! CHECK:             %[[VAL_39:.*]] = fir.convert %[[VAL_37]] : (index) -> i32
! CHECK:             fir.store %[[VAL_39]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:             %[[VAL_40:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:             %[[VAL_41:.*]] = fir.convert %[[VAL_40]] : (i32) -> i64
! CHECK:             %[[VAL_42:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_43:.*]] = arith.subi %[[VAL_41]], %[[VAL_42]] : i64
! CHECK:             %[[VAL_44:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i32) -> i64
! CHECK:             %[[VAL_46:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_47:.*]] = arith.subi %[[VAL_45]], %[[VAL_46]] : i64
! CHECK:             %[[VAL_48:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_43]], %[[VAL_47]] : (!fir.box<!fir.array<?x?xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK:             %[[VAL_49:.*]] = fir.load %[[VAL_48]] : !fir.ref<f32>
! CHECK:             %[[VAL_50:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:             %[[VAL_51:.*]] = arith.cmpf ogt, %[[VAL_49]], %[[VAL_50]] : f32
! CHECK:             %[[VAL_52:.*]] = fir.if %[[VAL_51]] -> (!fir.array<?x?xf32>) {
! CHECK:               %[[VAL_53:.*]] = arith.constant 1 : index
! CHECK:               %[[VAL_54:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:               %[[VAL_55:.*]] = fir.convert %[[VAL_54]] : (i32) -> i64
! CHECK:               %[[VAL_56:.*]] = fir.convert %[[VAL_55]] : (i64) -> index
! CHECK:               %[[VAL_57:.*]] = arith.subi %[[VAL_56]], %[[VAL_53]] : index
! CHECK:               %[[VAL_58:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:               %[[VAL_59:.*]] = fir.convert %[[VAL_58]] : (i32) -> i64
! CHECK:               %[[VAL_60:.*]] = fir.convert %[[VAL_59]] : (i64) -> index
! CHECK:               %[[VAL_61:.*]] = arith.subi %[[VAL_60]], %[[VAL_53]] : index
! CHECK-DAG:           %[[VAL_62:.*]] = arith.constant 3.140000e+00 : f32
! CHECK-DAG:           %[[VAL_63:.*]] = fir.array_fetch %[[VAL_31]], %[[VAL_57]], %[[VAL_61]] : (!fir.array<?x?xf32>, index, index) -> f32
! CHECK:               %[[VAL_64:.*]] = arith.divf %[[VAL_63]], %[[VAL_62]] {{.*}}: f32
! CHECK:               %[[VAL_65:.*]] = arith.constant 1 : index
! CHECK:               %[[VAL_66:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:               %[[VAL_67:.*]] = fir.convert %[[VAL_66]] : (i32) -> i64
! CHECK:               %[[VAL_68:.*]] = fir.convert %[[VAL_67]] : (i64) -> index
! CHECK:               %[[VAL_69:.*]] = arith.subi %[[VAL_68]], %[[VAL_65]] : index
! CHECK:               %[[VAL_70:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:               %[[VAL_71:.*]] = fir.convert %[[VAL_70]] : (i32) -> i64
! CHECK:               %[[VAL_72:.*]] = fir.convert %[[VAL_71]] : (i64) -> index
! CHECK:               %[[VAL_73:.*]] = arith.subi %[[VAL_72]], %[[VAL_65]] : index
! CHECK:               %[[VAL_74:.*]] = fir.array_update %[[VAL_38]], %[[VAL_64]], %[[VAL_69]], %[[VAL_73]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
! CHECK:               fir.result %[[VAL_74]] : !fir.array<?x?xf32>
! CHECK:             } else {
! CHECK:               fir.result %[[VAL_38]] : !fir.array<?x?xf32>
! CHECK:             }
! CHECK:             fir.result %[[VAL_75:.*]] : !fir.array<?x?xf32>
! CHECK:           }
! CHECK:           fir.result %[[VAL_76:.*]] : !fir.array<?x?xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_30]], %[[VAL_77:.*]] to %[[VAL_0]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.box<!fir.array<?x?xf32>>
! CHECK:         return
! CHECK:       }
