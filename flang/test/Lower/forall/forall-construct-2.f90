! Test forall lowering

! RUN: bbc -emit-fir %s -o - | FileCheck %s

!*** Test forall with multiple assignment statements
subroutine test2_forall_construct(a,b)
  real :: a(100,400), b(200,200)
  forall (i=1:100, j=1:200)
     a(i,j) = b(i,j) + b(i+1,j)
     a(i,200+j) = 1.0 / b(j, i)
  end forall
end subroutine test2_forall_construct

! CHECK-LABEL: func @_QPtest2_forall_construct(
! CHECK-SAME:       %[[VAL_0:.*]]: !fir.ref<!fir.array<100x400xf32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<200x200xf32>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_4:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:         %[[VAL_5:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_6:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_7:.*]] = arith.constant 400 : index
! CHECK:         %[[VAL_8:.*]] = arith.constant 200 : index
! CHECK:         %[[VAL_9:.*]] = arith.constant 200 : index
! CHECK:         %[[VAL_10:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> index
! CHECK:         %[[VAL_12:.*]] = arith.constant 100 : i32
! CHECK:         %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> index
! CHECK:         %[[VAL_14:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_15:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> index
! CHECK:         %[[VAL_17:.*]] = arith.constant 200 : i32
! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> index
! CHECK:         %[[VAL_19:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_20:.*]] = fir.shape %[[VAL_6]], %[[VAL_7]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_21:.*]] = fir.array_load %[[VAL_0]](%[[VAL_20]]) : (!fir.ref<!fir.array<100x400xf32>>, !fir.shape<2>) -> !fir.array<100x400xf32>
! CHECK:         %[[VAL_22:.*]] = fir.shape %[[VAL_8]], %[[VAL_9]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_23:.*]] = fir.array_load %[[VAL_1]](%[[VAL_22]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
! CHECK:         %[[VAL_24:.*]] = fir.shape %[[VAL_8]], %[[VAL_9]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_25:.*]] = fir.array_load %[[VAL_1]](%[[VAL_24]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
! CHECK:         %[[VAL_26:.*]] = fir.do_loop %[[VAL_27:.*]] = %[[VAL_11]] to %[[VAL_13]] step %[[VAL_14]] unordered iter_args(%[[VAL_28:.*]] = %[[VAL_21]]) -> (!fir.array<100x400xf32>) {
! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_27]] : (index) -> i32
! CHECK:           fir.store %[[VAL_29]] to %[[VAL_5]] : !fir.ref<i32>
! CHECK:           %[[VAL_30:.*]] = fir.do_loop %[[VAL_31:.*]] = %[[VAL_16]] to %[[VAL_18]] step %[[VAL_19]] unordered iter_args(%[[VAL_32:.*]] = %[[VAL_28]]) -> (!fir.array<100x400xf32>) {
! CHECK:             %[[VAL_33:.*]] = fir.convert %[[VAL_31]] : (index) -> i32
! CHECK:             fir.store %[[VAL_33]] to %[[VAL_4]] : !fir.ref<i32>
! CHECK:             %[[VAL_34:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_35:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! CHECK:             %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i32) -> i64
! CHECK:             %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i64) -> index
! CHECK:             %[[VAL_38:.*]] = arith.subi %[[VAL_37]], %[[VAL_34]] : index
! CHECK:             %[[VAL_39:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (i32) -> i64
! CHECK:             %[[VAL_41:.*]] = fir.convert %[[VAL_40]] : (i64) -> index
! CHECK:             %[[VAL_42:.*]] = arith.subi %[[VAL_41]], %[[VAL_34]] : index
! CHECK:             %[[VAL_54:.*]] = fir.array_fetch %[[VAL_23]], %[[VAL_38]], %[[VAL_42]] : (!fir.array<200x200xf32>, index, index) -> f32
! CHECK:             %[[VAL_43:.*]] = arith.constant 1 : index
! CHECK-DAG:         %[[VAL_44:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! CHECK-DAG:         %[[VAL_45:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_46:.*]] = arith.addi %[[VAL_44]], %[[VAL_45]] : i32
! CHECK:             %[[VAL_47:.*]] = fir.convert %[[VAL_46]] : (i32) -> i64
! CHECK:             %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (i64) -> index
! CHECK:             %[[VAL_49:.*]] = arith.subi %[[VAL_48]], %[[VAL_43]] : index
! CHECK:             %[[VAL_50:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:             %[[VAL_51:.*]] = fir.convert %[[VAL_50]] : (i32) -> i64
! CHECK:             %[[VAL_52:.*]] = fir.convert %[[VAL_51]] : (i64) -> index
! CHECK:             %[[VAL_53:.*]] = arith.subi %[[VAL_52]], %[[VAL_43]] : index
! CHECK:             %[[VAL_55:.*]] = fir.array_fetch %[[VAL_25]], %[[VAL_49]], %[[VAL_53]] : (!fir.array<200x200xf32>, index, index) -> f32
! CHECK:             %[[VAL_56:.*]] = arith.addf %[[VAL_54]], %[[VAL_55]] {{.*}}: f32
! CHECK:             %[[VAL_57:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_58:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! CHECK:             %[[VAL_59:.*]] = fir.convert %[[VAL_58]] : (i32) -> i64
! CHECK:             %[[VAL_60:.*]] = fir.convert %[[VAL_59]] : (i64) -> index
! CHECK:             %[[VAL_61:.*]] = arith.subi %[[VAL_60]], %[[VAL_57]] : index
! CHECK:             %[[VAL_62:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:             %[[VAL_63:.*]] = fir.convert %[[VAL_62]] : (i32) -> i64
! CHECK:             %[[VAL_64:.*]] = fir.convert %[[VAL_63]] : (i64) -> index
! CHECK:             %[[VAL_65:.*]] = arith.subi %[[VAL_64]], %[[VAL_57]] : index
! CHECK:             %[[VAL_66:.*]] = fir.array_update %[[VAL_32]], %[[VAL_56]], %[[VAL_61]], %[[VAL_65]] : (!fir.array<100x400xf32>, f32, index, index) -> !fir.array<100x400xf32>
! CHECK:             fir.result %[[VAL_66]] : !fir.array<100x400xf32>
! CHECK:           }
! CHECK:           fir.result %[[VAL_67:.*]] : !fir.array<100x400xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_21]], %[[VAL_68:.*]] to %[[VAL_0]] : !fir.array<100x400xf32>, !fir.array<100x400xf32>, !fir.ref<!fir.array<100x400xf32>>
! CHECK:         %[[VAL_69:.*]] = fir.shape %[[VAL_6]], %[[VAL_7]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_70:.*]] = fir.array_load %[[VAL_0]](%[[VAL_69]]) : (!fir.ref<!fir.array<100x400xf32>>, !fir.shape<2>) -> !fir.array<100x400xf32>
! CHECK:         %[[VAL_71:.*]] = fir.shape %[[VAL_8]], %[[VAL_9]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_72:.*]] = fir.array_load %[[VAL_1]](%[[VAL_71]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
! CHECK:         %[[VAL_73:.*]] = fir.do_loop %[[VAL_74:.*]] = %[[VAL_11]] to %[[VAL_13]] step %[[VAL_14]] unordered iter_args(%[[VAL_75:.*]] = %[[VAL_70]]) -> (!fir.array<100x400xf32>) {
! CHECK:           %[[VAL_76:.*]] = fir.convert %[[VAL_74]] : (index) -> i32
! CHECK:           fir.store %[[VAL_76]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_77:.*]] = fir.do_loop %[[VAL_78:.*]] = %[[VAL_16]] to %[[VAL_18]] step %[[VAL_19]] unordered iter_args(%[[VAL_79:.*]] = %[[VAL_75]]) -> (!fir.array<100x400xf32>) {
! CHECK:             %[[VAL_80:.*]] = fir.convert %[[VAL_78]] : (index) -> i32
! CHECK:             fir.store %[[VAL_80]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:             %[[VAL_81:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:             %[[VAL_82:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_83:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:             %[[VAL_84:.*]] = fir.convert %[[VAL_83]] : (i32) -> i64
! CHECK:             %[[VAL_85:.*]] = fir.convert %[[VAL_84]] : (i64) -> index
! CHECK:             %[[VAL_86:.*]] = arith.subi %[[VAL_85]], %[[VAL_82]] : index
! CHECK:             %[[VAL_87:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:             %[[VAL_88:.*]] = fir.convert %[[VAL_87]] : (i32) -> i64
! CHECK:             %[[VAL_89:.*]] = fir.convert %[[VAL_88]] : (i64) -> index
! CHECK:             %[[VAL_90:.*]] = arith.subi %[[VAL_89]], %[[VAL_82]] : index
! CHECK:             %[[VAL_91:.*]] = fir.array_fetch %[[VAL_72]], %[[VAL_86]], %[[VAL_90]] : (!fir.array<200x200xf32>, index, index) -> f32
! CHECK:             %[[VAL_92:.*]] = arith.divf %[[VAL_81]], %[[VAL_91]] {{.*}}: f32
! CHECK:             %[[VAL_93:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_94:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:             %[[VAL_95:.*]] = fir.convert %[[VAL_94]] : (i32) -> i64
! CHECK:             %[[VAL_96:.*]] = fir.convert %[[VAL_95]] : (i64) -> index
! CHECK:             %[[VAL_97:.*]] = arith.subi %[[VAL_96]], %[[VAL_93]] : index
! CHECK:             %[[VAL_98:.*]] = arith.constant 200 : i32
! CHECK:             %[[VAL_99:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:             %[[VAL_100:.*]] = arith.addi %[[VAL_98]], %[[VAL_99]] : i32
! CHECK:             %[[VAL_101:.*]] = fir.convert %[[VAL_100]] : (i32) -> i64
! CHECK:             %[[VAL_102:.*]] = fir.convert %[[VAL_101]] : (i64) -> index
! CHECK:             %[[VAL_103:.*]] = arith.subi %[[VAL_102]], %[[VAL_93]] : index
! CHECK:             %[[VAL_104:.*]] = fir.array_update %[[VAL_79]], %[[VAL_92]], %[[VAL_97]], %[[VAL_103]] : (!fir.array<100x400xf32>, f32, index, index) -> !fir.array<100x400xf32>
! CHECK:             fir.result %[[VAL_104]] : !fir.array<100x400xf32>
! CHECK:           }
! CHECK:           fir.result %[[VAL_105:.*]] : !fir.array<100x400xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_70]], %[[VAL_106:.*]] to %[[VAL_0]] : !fir.array<100x400xf32>, !fir.array<100x400xf32>, !fir.ref<!fir.array<100x400xf32>>
! CHECK:         return
! CHECK:       }
