! Test forall lowering

! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

!*** Test forall with multiple assignment statements and mask
subroutine test3_forall_construct(a,b, mask)
  real :: a(100,400), b(200,200)
  logical :: mask(100,200)
  forall (i=1:100, j=1:200, mask(i,j))
     a(i,j) = b(i,j) + b(i+1,j)
     a(i,200+j) = 1.0 / b(j, i)
  end forall
end subroutine test3_forall_construct

! CHECK-LABEL: func @_QPtest3_forall_construct(
! CHECK-SAME:        %[[VAL_0:.*]]: !fir.ref<!fir.array<100x400xf32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<200x200xf32>>{{.*}}, %[[VAL_2:.*]]: !fir.ref<!fir.array<100x200x!fir.logical<4>>>{{.*}}) {
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:         %[[VAL_4:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_5:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:         %[[VAL_6:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_7:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_8:.*]] = arith.constant 400 : index
! CHECK:         %[[VAL_9:.*]] = arith.constant 200 : index
! CHECK:         %[[VAL_10:.*]] = arith.constant 200 : index
! CHECK:         %[[VAL_11:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
! CHECK:         %[[VAL_13:.*]] = arith.constant 100 : i32
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> index
! CHECK:         %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_16:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i32) -> index
! CHECK:         %[[VAL_18:.*]] = arith.constant 200 : i32
! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> index
! CHECK:         %[[VAL_20:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_21:.*]] = fir.shape %[[VAL_7]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_22:.*]] = fir.array_load %[[VAL_0]](%[[VAL_21]]) : (!fir.ref<!fir.array<100x400xf32>>, !fir.shape<2>) -> !fir.array<100x400xf32>
! CHECK:         %[[VAL_23:.*]] = fir.shape %[[VAL_9]], %[[VAL_10]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_24:.*]] = fir.array_load %[[VAL_1]](%[[VAL_23]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
! CHECK:         %[[VAL_25:.*]] = fir.shape %[[VAL_9]], %[[VAL_10]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_26:.*]] = fir.array_load %[[VAL_1]](%[[VAL_25]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
! CHECK:         %[[VAL_27:.*]] = fir.do_loop %[[VAL_28:.*]] = %[[VAL_12]] to %[[VAL_14]] step %[[VAL_15]] unordered iter_args(%[[VAL_29:.*]] = %[[VAL_22]]) -> (!fir.array<100x400xf32>) {
! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_28]] : (index) -> i32
! CHECK:           fir.store %[[VAL_30]] to %[[VAL_6]] : !fir.ref<i32>
! CHECK:           %[[VAL_31:.*]] = fir.do_loop %[[VAL_32:.*]] = %[[VAL_17]] to %[[VAL_19]] step %[[VAL_20]] unordered iter_args(%[[VAL_33:.*]] = %[[VAL_29]]) -> (!fir.array<100x400xf32>) {
! CHECK:             %[[VAL_34:.*]] = fir.convert %[[VAL_32]] : (index) -> i32
! CHECK:             fir.store %[[VAL_34]] to %[[VAL_5]] : !fir.ref<i32>
! CHECK:             %[[VAL_35:.*]] = fir.load %[[VAL_6]] : !fir.ref<i32>
! CHECK:             %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i32) -> i64
! CHECK:             %[[VAL_37:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_38:.*]] = arith.subi %[[VAL_36]], %[[VAL_37]] : i64
! CHECK:             %[[VAL_39:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (i32) -> i64
! CHECK:             %[[VAL_41:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_42:.*]] = arith.subi %[[VAL_40]], %[[VAL_41]] : i64
! CHECK:             %[[VAL_43:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_38]], %[[VAL_42]] : (!fir.ref<!fir.array<100x200x!fir.logical<4>>>, i64, i64) -> !fir.ref<!fir.logical<4>>
! CHECK:             %[[VAL_44:.*]] = fir.load %[[VAL_43]] : !fir.ref<!fir.logical<4>>
! CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (!fir.logical<4>) -> i1
! CHECK:             %[[VAL_46:.*]] = fir.if %[[VAL_45]] -> (!fir.array<100x400xf32>) {
! CHECK:               %[[VAL_47:.*]] = arith.constant 1 : index
! CHECK:               %[[VAL_48:.*]] = fir.load %[[VAL_6]] : !fir.ref<i32>
! CHECK:               %[[VAL_49:.*]] = fir.convert %[[VAL_48]] : (i32) -> i64
! CHECK:               %[[VAL_50:.*]] = fir.convert %[[VAL_49]] : (i64) -> index
! CHECK:               %[[VAL_51:.*]] = arith.subi %[[VAL_50]], %[[VAL_47]] : index
! CHECK:               %[[VAL_52:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! CHECK:               %[[VAL_53:.*]] = fir.convert %[[VAL_52]] : (i32) -> i64
! CHECK:               %[[VAL_54:.*]] = fir.convert %[[VAL_53]] : (i64) -> index
! CHECK:               %[[VAL_55:.*]] = arith.subi %[[VAL_54]], %[[VAL_47]] : index
! CHECK:               %[[VAL_67:.*]] = fir.array_fetch %[[VAL_24]], %[[VAL_51]], %[[VAL_55]] : (!fir.array<200x200xf32>, index, index) -> f32
! CHECK:               %[[VAL_56:.*]] = arith.constant 1 : index
! CHECK-DAG:           %[[VAL_57:.*]] = fir.load %[[VAL_6]] : !fir.ref<i32>
! CHECK-DAG:           %[[VAL_58:.*]] = arith.constant 1 : i32
! CHECK:               %[[VAL_59:.*]] = arith.addi %[[VAL_57]], %[[VAL_58]] : i32
! CHECK:               %[[VAL_60:.*]] = fir.convert %[[VAL_59]] : (i32) -> i64
! CHECK:               %[[VAL_61:.*]] = fir.convert %[[VAL_60]] : (i64) -> index
! CHECK:               %[[VAL_62:.*]] = arith.subi %[[VAL_61]], %[[VAL_56]] : index
! CHECK:               %[[VAL_63:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! CHECK:               %[[VAL_64:.*]] = fir.convert %[[VAL_63]] : (i32) -> i64
! CHECK:               %[[VAL_65:.*]] = fir.convert %[[VAL_64]] : (i64) -> index
! CHECK:               %[[VAL_66:.*]] = arith.subi %[[VAL_65]], %[[VAL_56]] : index
! CHECK:               %[[VAL_68:.*]] = fir.array_fetch %[[VAL_26]], %[[VAL_62]], %[[VAL_66]] : (!fir.array<200x200xf32>, index, index) -> f32
! CHECK:               %[[VAL_69:.*]] = arith.addf %[[VAL_67]], %[[VAL_68]] {{.*}}: f32
! CHECK:               %[[VAL_70:.*]] = arith.constant 1 : index
! CHECK:               %[[VAL_71:.*]] = fir.load %[[VAL_6]] : !fir.ref<i32>
! CHECK:               %[[VAL_72:.*]] = fir.convert %[[VAL_71]] : (i32) -> i64
! CHECK:               %[[VAL_73:.*]] = fir.convert %[[VAL_72]] : (i64) -> index
! CHECK:               %[[VAL_74:.*]] = arith.subi %[[VAL_73]], %[[VAL_70]] : index
! CHECK:               %[[VAL_75:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! CHECK:               %[[VAL_76:.*]] = fir.convert %[[VAL_75]] : (i32) -> i64
! CHECK:               %[[VAL_77:.*]] = fir.convert %[[VAL_76]] : (i64) -> index
! CHECK:               %[[VAL_78:.*]] = arith.subi %[[VAL_77]], %[[VAL_70]] : index
! CHECK:               %[[VAL_79:.*]] = fir.array_update %[[VAL_33]], %[[VAL_69]], %[[VAL_74]], %[[VAL_78]] : (!fir.array<100x400xf32>, f32, index, index) -> !fir.array<100x400xf32>
! CHECK:               fir.result %[[VAL_79]] : !fir.array<100x400xf32>
! CHECK:             } else {
! CHECK:               fir.result %[[VAL_33]] : !fir.array<100x400xf32>
! CHECK:             }
! CHECK:             fir.result %[[VAL_80:.*]] : !fir.array<100x400xf32>
! CHECK:           }
! CHECK:           fir.result %[[VAL_81:.*]] : !fir.array<100x400xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_22]], %[[VAL_82:.*]] to %[[VAL_0]] : !fir.array<100x400xf32>, !fir.array<100x400xf32>, !fir.ref<!fir.array<100x400xf32>>
! CHECK:         %[[VAL_83:.*]] = fir.shape %[[VAL_7]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_84:.*]] = fir.array_load %[[VAL_0]](%[[VAL_83]]) : (!fir.ref<!fir.array<100x400xf32>>, !fir.shape<2>) -> !fir.array<100x400xf32>
! CHECK:         %[[VAL_85:.*]] = fir.shape %[[VAL_9]], %[[VAL_10]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_86:.*]] = fir.array_load %[[VAL_1]](%[[VAL_85]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
! CHECK:         %[[VAL_87:.*]] = fir.do_loop %[[VAL_88:.*]] = %[[VAL_12]] to %[[VAL_14]] step %[[VAL_15]] unordered iter_args(%[[VAL_89:.*]] = %[[VAL_84]]) -> (!fir.array<100x400xf32>) {
! CHECK:           %[[VAL_90:.*]] = fir.convert %[[VAL_88]] : (index) -> i32
! CHECK:           fir.store %[[VAL_90]] to %[[VAL_4]] : !fir.ref<i32>
! CHECK:           %[[VAL_91:.*]] = fir.do_loop %[[VAL_92:.*]] = %[[VAL_17]] to %[[VAL_19]] step %[[VAL_20]] unordered iter_args(%[[VAL_93:.*]] = %[[VAL_89]]) -> (!fir.array<100x400xf32>) {
! CHECK:             %[[VAL_94:.*]] = fir.convert %[[VAL_92]] : (index) -> i32
! CHECK:             fir.store %[[VAL_94]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:             %[[VAL_95:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:             %[[VAL_96:.*]] = fir.convert %[[VAL_95]] : (i32) -> i64
! CHECK:             %[[VAL_97:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_98:.*]] = arith.subi %[[VAL_96]], %[[VAL_97]] : i64
! CHECK:             %[[VAL_99:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:             %[[VAL_100:.*]] = fir.convert %[[VAL_99]] : (i32) -> i64
! CHECK:             %[[VAL_101:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_102:.*]] = arith.subi %[[VAL_100]], %[[VAL_101]] : i64
! CHECK:             %[[VAL_103:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_98]], %[[VAL_102]] : (!fir.ref<!fir.array<100x200x!fir.logical<4>>>, i64, i64) -> !fir.ref<!fir.logical<4>>
! CHECK:             %[[VAL_104:.*]] = fir.load %[[VAL_103]] : !fir.ref<!fir.logical<4>>
! CHECK:             %[[VAL_105:.*]] = fir.convert %[[VAL_104]] : (!fir.logical<4>) -> i1
! CHECK:             %[[VAL_106:.*]] = fir.if %[[VAL_105]] -> (!fir.array<100x400xf32>) {
! CHECK:               %[[VAL_107:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:               %[[VAL_108:.*]] = arith.constant 1 : index
! CHECK:               %[[VAL_109:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:               %[[VAL_110:.*]] = fir.convert %[[VAL_109]] : (i32) -> i64
! CHECK:               %[[VAL_111:.*]] = fir.convert %[[VAL_110]] : (i64) -> index
! CHECK:               %[[VAL_112:.*]] = arith.subi %[[VAL_111]], %[[VAL_108]] : index
! CHECK:               %[[VAL_113:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:               %[[VAL_114:.*]] = fir.convert %[[VAL_113]] : (i32) -> i64
! CHECK:               %[[VAL_115:.*]] = fir.convert %[[VAL_114]] : (i64) -> index
! CHECK:               %[[VAL_116:.*]] = arith.subi %[[VAL_115]], %[[VAL_108]] : index
! CHECK:               %[[VAL_117:.*]] = fir.array_fetch %[[VAL_86]], %[[VAL_112]], %[[VAL_116]] : (!fir.array<200x200xf32>, index, index) -> f32
! CHECK:               %[[VAL_118:.*]] = arith.divf %[[VAL_107]], %[[VAL_117]] {{.*}}: f32
! CHECK:               %[[VAL_119:.*]] = arith.constant 1 : index
! CHECK:               %[[VAL_120:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:               %[[VAL_121:.*]] = fir.convert %[[VAL_120]] : (i32) -> i64
! CHECK:               %[[VAL_122:.*]] = fir.convert %[[VAL_121]] : (i64) -> index
! CHECK:               %[[VAL_123:.*]] = arith.subi %[[VAL_122]], %[[VAL_119]] : index
! CHECK:               %[[VAL_124:.*]] = arith.constant 200 : i32
! CHECK:               %[[VAL_125:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:               %[[VAL_126:.*]] = arith.addi %[[VAL_124]], %[[VAL_125]] : i32
! CHECK:               %[[VAL_127:.*]] = fir.convert %[[VAL_126]] : (i32) -> i64
! CHECK:               %[[VAL_128:.*]] = fir.convert %[[VAL_127]] : (i64) -> index
! CHECK:               %[[VAL_129:.*]] = arith.subi %[[VAL_128]], %[[VAL_119]] : index
! CHECK:               %[[VAL_130:.*]] = fir.array_update %[[VAL_93]], %[[VAL_118]], %[[VAL_123]], %[[VAL_129]] : (!fir.array<100x400xf32>, f32, index, index) -> !fir.array<100x400xf32>
! CHECK:               fir.result %[[VAL_130]] : !fir.array<100x400xf32>
! CHECK:             } else {
! CHECK:               fir.result %[[VAL_93]] : !fir.array<100x400xf32>
! CHECK:             }
! CHECK:             fir.result %[[VAL_131:.*]] : !fir.array<100x400xf32>
! CHECK:           }
! CHECK:           fir.result %[[VAL_132:.*]] : !fir.array<100x400xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_84]], %[[VAL_133:.*]] to %[[VAL_0]] : !fir.array<100x400xf32>, !fir.array<100x400xf32>, !fir.ref<!fir.array<100x400xf32>>
! CHECK:         return
! CHECK:       }
