! Test forall lowering

! RUN: bbc -emit-fir %s -o - | FileCheck %s

!*** Test a FORALL construct with a nested WHERE construct.
!    This has both an explicit and implicit iteration space. The WHERE construct
!    makes the assignments conditional and the where mask evaluation must happen
!    prior to evaluating the array assignment statement.
subroutine test_nested_forall_where(a,b)  
  type t
     real data(100)
  end type t
  type(t) :: a(:,:), b(:,:)
  forall (i=1:ubound(a,1), j=1:ubound(a,2))
     where (b(j,i)%data > 0.0)
        a(i,j)%data = b(j,i)%data / 3.14
     elsewhere
        a(i,j)%data = -b(j,i)%data
     end where
  end forall
end subroutine test_nested_forall_where

! CHECK-LABEL: func @_QPtest_nested_forall_where(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>{{.*}}, %[[VAL_1:.*]]: !fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_4:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:         %[[VAL_5:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_6:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:         %[[VAL_7:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_8:.*]] = fir.alloca tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>
! CHECK:         %[[VAL_9:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_10:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_9]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<i64>
! CHECK:         fir.store %[[VAL_10]] to %[[VAL_11]] : !fir.ref<i64>
! CHECK:         %[[VAL_12:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_13:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi8>>
! CHECK:         %[[VAL_14:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_12]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:         fir.store %[[VAL_13]] to %[[VAL_14]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:         %[[VAL_15:.*]] = arith.constant 2 : i32
! CHECK:         %[[VAL_16:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi64>>
! CHECK:         %[[VAL_17:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_15]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi64>>>
! CHECK:         fir.store %[[VAL_16]] to %[[VAL_17]] : !fir.ref<!fir.heap<!fir.array<?xi64>>>
! CHECK:         %[[VAL_18:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> index
! CHECK:         %[[VAL_20:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_21:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_20]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_22:.*]] = fir.convert %[[VAL_21]]#1 : (index) -> i64
! CHECK:         %[[VAL_23:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (index) -> i64
! CHECK:         %[[VAL_25:.*]] = arith.addi %[[VAL_22]], %[[VAL_24]] : i64
! CHECK:         %[[VAL_26:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_27:.*]] = arith.subi %[[VAL_25]], %[[VAL_26]] : i64
! CHECK:         %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i64) -> i32
! CHECK:         %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i32) -> index
! CHECK:         %[[VAL_30:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_31:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> index
! CHECK:         %[[VAL_33:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_34:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_33]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_35:.*]] = fir.convert %[[VAL_34]]#1 : (index) -> i64
! CHECK:         %[[VAL_36:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (index) -> i64
! CHECK:         %[[VAL_38:.*]] = arith.addi %[[VAL_35]], %[[VAL_37]] : i64
! CHECK:         %[[VAL_39:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_40:.*]] = arith.subi %[[VAL_38]], %[[VAL_39]] : i64
! CHECK:         %[[VAL_41:.*]] = fir.convert %[[VAL_40]] : (i64) -> i32
! CHECK:         %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (i32) -> index
! CHECK:         %[[VAL_43:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_44:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
! CHECK:         %[[VAL_45:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
! CHECK:         %[[VAL_46:.*]] = fir.do_loop %[[VAL_47:.*]] = %[[VAL_19]] to %[[VAL_29]] step %[[VAL_30]] unordered iter_args(%[[VAL_48:.*]] = %[[VAL_44]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
! CHECK:           %[[VAL_49:.*]] = fir.convert %[[VAL_47]] : (index) -> i32
! CHECK:           fir.store %[[VAL_49]] to %[[VAL_7]] : !fir.ref<i32>
! CHECK:           %[[VAL_50:.*]] = fir.do_loop %[[VAL_51:.*]] = %[[VAL_32]] to %[[VAL_42]] step %[[VAL_43]] unordered iter_args(%[[VAL_52:.*]] = %[[VAL_48]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
! CHECK:             %[[VAL_53:.*]] = fir.convert %[[VAL_51]] : (index) -> i32
! CHECK:             fir.store %[[VAL_53]] to %[[VAL_6]] : !fir.ref<i32>
! CHECK:             %[[VAL_54:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_55:.*]] = arith.constant 0 : i64
! CHECK:             %[[VAL_56:.*]] = fir.convert %[[VAL_19]] : (index) -> i64
! CHECK:             %[[VAL_57:.*]] = fir.convert %[[VAL_29]] : (index) -> i64
! CHECK:             %[[VAL_58:.*]] = fir.convert %[[VAL_30]] : (index) -> i64
! CHECK:             %[[VAL_59:.*]] = arith.subi %[[VAL_57]], %[[VAL_56]] : i64
! CHECK:             %[[VAL_60:.*]] = arith.addi %[[VAL_59]], %[[VAL_58]] : i64
! CHECK:             %[[VAL_61:.*]] = arith.divsi %[[VAL_60]], %[[VAL_58]] : i64
! CHECK:             %[[VAL_62:.*]] = arith.cmpi sgt, %[[VAL_61]], %[[VAL_55]] : i64
! CHECK:             %[[VAL_63:.*]] = arith.select %[[VAL_62]], %[[VAL_61]], %[[VAL_55]] : i64
! CHECK:             %[[VAL_64:.*]] = arith.constant 0 : i64
! CHECK:             %[[VAL_65:.*]] = fir.convert %[[VAL_32]] : (index) -> i64
! CHECK:             %[[VAL_66:.*]] = fir.convert %[[VAL_42]] : (index) -> i64
! CHECK:             %[[VAL_67:.*]] = fir.convert %[[VAL_43]] : (index) -> i64
! CHECK:             %[[VAL_68:.*]] = arith.subi %[[VAL_66]], %[[VAL_65]] : i64
! CHECK:             %[[VAL_69:.*]] = arith.addi %[[VAL_68]], %[[VAL_67]] : i64
! CHECK:             %[[VAL_70:.*]] = arith.divsi %[[VAL_69]], %[[VAL_67]] : i64
! CHECK:             %[[VAL_71:.*]] = arith.cmpi sgt, %[[VAL_70]], %[[VAL_64]] : i64
! CHECK:             %[[VAL_72:.*]] = arith.select %[[VAL_71]], %[[VAL_70]], %[[VAL_64]] : i64
! CHECK:             %[[VAL_73:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_74:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_73]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             %[[VAL_75:.*]] = fir.load %[[VAL_74]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             %[[VAL_76:.*]] = fir.convert %[[VAL_75]] : (!fir.heap<!fir.array<?xi8>>) -> i64
! CHECK:             %[[VAL_77:.*]] = arith.constant 0 : i64
! CHECK:             %[[VAL_78:.*]] = arith.cmpi eq, %[[VAL_76]], %[[VAL_77]] : i64
! CHECK:             fir.if %[[VAL_78]] {
! CHECK:               %[[VAL_79:.*]] = arith.constant true
! CHECK:               %[[VAL_80:.*]] = arith.constant 2 : i64
! CHECK:               %[[VAL_81:.*]] = fir.allocmem !fir.array<2xi64>
! CHECK:               %[[VAL_82:.*]] = arith.constant 0 : i32
! CHECK:               %[[VAL_83:.*]] = fir.coordinate_of %[[VAL_81]], %[[VAL_82]] : (!fir.heap<!fir.array<2xi64>>, i32) -> !fir.ref<i64>
! CHECK:               fir.store %[[VAL_63]] to %[[VAL_83]] : !fir.ref<i64>
! CHECK:               %[[VAL_84:.*]] = arith.constant 1 : i32
! CHECK:               %[[VAL_85:.*]] = fir.coordinate_of %[[VAL_81]], %[[VAL_84]] : (!fir.heap<!fir.array<2xi64>>, i32) -> !fir.ref<i64>
! CHECK:               fir.store %[[VAL_72]] to %[[VAL_85]] : !fir.ref<i64>
! CHECK:               %[[VAL_86:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>) -> !fir.llvm_ptr<i8>
! CHECK:               %[[VAL_87:.*]] = fir.convert %[[VAL_81]] : (!fir.heap<!fir.array<2xi64>>) -> !fir.ref<i64>
! CHECK:               %[[VAL_88:.*]] = fir.call @_FortranARaggedArrayAllocate(%[[VAL_86]], %[[VAL_79]], %[[VAL_80]], %[[VAL_54]], %[[VAL_87]]) {{.*}}: (!fir.llvm_ptr<i8>, i1, i64, i64, !fir.ref<i64>) -> !fir.llvm_ptr<i8>
! CHECK:             }
! CHECK:             %[[VAL_89:.*]] = arith.subi %[[VAL_47]], %[[VAL_19]] : index
! CHECK:             %[[VAL_90:.*]] = arith.divsi %[[VAL_89]], %[[VAL_30]] : index
! CHECK:             %[[VAL_91:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_92:.*]] = arith.addi %[[VAL_90]], %[[VAL_91]] : index
! CHECK:             %[[VAL_93:.*]] = arith.subi %[[VAL_51]], %[[VAL_32]] : index
! CHECK:             %[[VAL_94:.*]] = arith.divsi %[[VAL_93]], %[[VAL_43]] : index
! CHECK:             %[[VAL_95:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_96:.*]] = arith.addi %[[VAL_94]], %[[VAL_95]] : index
! CHECK:             %[[VAL_97:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_98:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_97]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             %[[VAL_99:.*]] = fir.load %[[VAL_98]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             %[[VAL_100:.*]] = fir.convert %[[VAL_99]] : (!fir.heap<!fir.array<?xi8>>) -> !fir.ref<!fir.array<?x?xtuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>>
! CHECK:             %[[VAL_101:.*]] = fir.shape %[[VAL_63]], %[[VAL_72]] : (i64, i64) -> !fir.shape<2>
! CHECK:             %[[VAL_102:.*]] = fir.array_coor %[[VAL_100]](%[[VAL_101]]) %[[VAL_92]], %[[VAL_96]] : (!fir.ref<!fir.array<?x?xtuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>>, !fir.shape<2>, index, index) -> !fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>
! CHECK:             %[[VAL_103:.*]] = fir.load %[[VAL_6]] : !fir.ref<i32>
! CHECK:             %[[VAL_104:.*]] = fir.convert %[[VAL_103]] : (i32) -> i64
! CHECK:             %[[VAL_105:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_106:.*]] = arith.subi %[[VAL_104]], %[[VAL_105]] : i64
! CHECK:             %[[VAL_107:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
! CHECK:             %[[VAL_108:.*]] = fir.convert %[[VAL_107]] : (i32) -> i64
! CHECK:             %[[VAL_109:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_110:.*]] = arith.subi %[[VAL_108]], %[[VAL_109]] : i64
! CHECK:             %[[VAL_111:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_106]], %[[VAL_110]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, i64, i64) -> !fir.ref<!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
! CHECK:             %[[VAL_112:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
! CHECK:             %[[VAL_113:.*]] = fir.coordinate_of %[[VAL_111]], %[[VAL_112]] : (!fir.ref<!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.field) -> !fir.ref<!fir.array<100xf32>>
! CHECK:             %[[VAL_114:.*]] = arith.constant 100 : index
! CHECK:             %[[VAL_115:.*]] = fir.shape %[[VAL_114]] : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_116:.*]] = fir.array_load %[[VAL_113]](%[[VAL_115]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
! CHECK:             %[[VAL_117:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:             %[[VAL_118:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_119:.*]] = fir.coordinate_of %[[VAL_102]], %[[VAL_118]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             %[[VAL_120:.*]] = fir.load %[[VAL_119]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             %[[VAL_121:.*]] = fir.shape %[[VAL_114]] : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_122:.*]] = fir.array_load %[[VAL_120]](%[[VAL_121]]) : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>) -> !fir.array<?xi8>
! CHECK:             %[[VAL_123:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_124:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_125:.*]] = fir.coordinate_of %[[VAL_102]], %[[VAL_124]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             %[[VAL_126:.*]] = fir.load %[[VAL_125]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             %[[VAL_127:.*]] = fir.convert %[[VAL_126]] : (!fir.heap<!fir.array<?xi8>>) -> i64
! CHECK:             %[[VAL_128:.*]] = arith.constant 0 : i64
! CHECK:             %[[VAL_129:.*]] = arith.cmpi eq, %[[VAL_127]], %[[VAL_128]] : i64
! CHECK:             fir.if %[[VAL_129]] {
! CHECK:               %[[VAL_130:.*]] = arith.constant false
! CHECK:               %[[VAL_131:.*]] = arith.constant 1 : i64
! CHECK:               %[[VAL_132:.*]] = fir.allocmem !fir.array<1xi64>
! CHECK:               %[[VAL_133:.*]] = arith.constant 0 : i32
! CHECK:               %[[VAL_134:.*]] = fir.coordinate_of %[[VAL_132]], %[[VAL_133]] : (!fir.heap<!fir.array<1xi64>>, i32) -> !fir.ref<i64>
! CHECK:               %[[VAL_135:.*]] = fir.convert %[[VAL_114]] : (index) -> i64
! CHECK:               fir.store %[[VAL_135]] to %[[VAL_134]] : !fir.ref<i64>
! CHECK:               %[[VAL_136:.*]] = fir.convert %[[VAL_102]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>) -> !fir.llvm_ptr<i8>
! CHECK:               %[[VAL_137:.*]] = fir.convert %[[VAL_132]] : (!fir.heap<!fir.array<1xi64>>) -> !fir.ref<i64>
! CHECK:               %[[VAL_138:.*]] = fir.call @_FortranARaggedArrayAllocate(%[[VAL_136]], %[[VAL_130]], %[[VAL_131]], %[[VAL_123]], %[[VAL_137]]) {{.*}}: (!fir.llvm_ptr<i8>, i1, i64, i64, !fir.ref<i64>) -> !fir.llvm_ptr<i8>
! CHECK:             }
! CHECK:             %[[VAL_139:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_140:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_141:.*]] = arith.subi %[[VAL_114]], %[[VAL_139]] : index
! CHECK:             %[[VAL_142:.*]] = fir.do_loop %[[VAL_143:.*]] = %[[VAL_140]] to %[[VAL_141]] step %[[VAL_139]] unordered iter_args(%[[VAL_144:.*]] = %[[VAL_122]]) -> (!fir.array<?xi8>) {
! CHECK:               %[[VAL_145:.*]] = fir.array_fetch %[[VAL_116]], %[[VAL_143]] : (!fir.array<100xf32>, index) -> f32
! CHECK:               %[[VAL_146:.*]] = arith.cmpf ogt, %[[VAL_145]], %[[VAL_117]] : f32
! CHECK:               %[[VAL_147:.*]] = arith.constant 1 : i32
! CHECK:               %[[VAL_148:.*]] = fir.coordinate_of %[[VAL_102]], %[[VAL_147]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:               %[[VAL_149:.*]] = fir.load %[[VAL_148]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:               %[[VAL_150:.*]] = fir.shape %[[VAL_114]] : (index) -> !fir.shape<1>
! CHECK:               %[[VAL_151:.*]] = arith.constant 1 : index
! CHECK:               %[[VAL_152:.*]] = arith.addi %[[VAL_143]], %[[VAL_151]] : index
! CHECK:               %[[VAL_153:.*]] = fir.array_coor %[[VAL_149]](%[[VAL_150]]) %[[VAL_152]] : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
! CHECK:               %[[VAL_154:.*]] = fir.convert %[[VAL_146]] : (i1) -> i8
! CHECK:               fir.store %[[VAL_154]] to %[[VAL_153]] : !fir.ref<i8>
! CHECK:               fir.result %[[VAL_144]] : !fir.array<?xi8>
! CHECK:             }
! CHECK:             fir.result %[[VAL_52]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_155:.*]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
! CHECK:         }
! CHECK:         %[[VAL_156:.*]] = fir.do_loop %[[VAL_157:.*]] = %[[VAL_19]] to %[[VAL_29]] step %[[VAL_30]] unordered iter_args(%[[VAL_158:.*]] = %[[VAL_44]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
! CHECK:           %[[VAL_159:.*]] = fir.convert %[[VAL_157]] : (index) -> i32
! CHECK:           fir.store %[[VAL_159]] to %[[VAL_5]] : !fir.ref<i32>
! CHECK:           %[[VAL_160:.*]] = fir.do_loop %[[VAL_161:.*]] = %[[VAL_32]] to %[[VAL_42]] step %[[VAL_43]] unordered iter_args(%[[VAL_162:.*]] = %[[VAL_158]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
! CHECK:             %[[VAL_163:.*]] = fir.convert %[[VAL_161]] : (index) -> i32
! CHECK:             fir.store %[[VAL_163]] to %[[VAL_4]] : !fir.ref<i32>
! CHECK:             %[[VAL_164:.*]] = arith.constant 0 : i64
! CHECK:             %[[VAL_165:.*]] = fir.convert %[[VAL_19]] : (index) -> i64
! CHECK:             %[[VAL_166:.*]] = fir.convert %[[VAL_29]] : (index) -> i64
! CHECK:             %[[VAL_167:.*]] = fir.convert %[[VAL_30]] : (index) -> i64
! CHECK:             %[[VAL_168:.*]] = arith.subi %[[VAL_166]], %[[VAL_165]] : i64
! CHECK:             %[[VAL_169:.*]] = arith.addi %[[VAL_168]], %[[VAL_167]] : i64
! CHECK:             %[[VAL_170:.*]] = arith.divsi %[[VAL_169]], %[[VAL_167]] : i64
! CHECK:             %[[VAL_171:.*]] = arith.cmpi sgt, %[[VAL_170]], %[[VAL_164]] : i64
! CHECK:             %[[VAL_172:.*]] = arith.select %[[VAL_171]], %[[VAL_170]], %[[VAL_164]] : i64
! CHECK:             %[[VAL_173:.*]] = arith.constant 0 : i64
! CHECK:             %[[VAL_174:.*]] = fir.convert %[[VAL_32]] : (index) -> i64
! CHECK:             %[[VAL_175:.*]] = fir.convert %[[VAL_42]] : (index) -> i64
! CHECK:             %[[VAL_176:.*]] = fir.convert %[[VAL_43]] : (index) -> i64
! CHECK:             %[[VAL_177:.*]] = arith.subi %[[VAL_175]], %[[VAL_174]] : i64
! CHECK:             %[[VAL_178:.*]] = arith.addi %[[VAL_177]], %[[VAL_176]] : i64
! CHECK:             %[[VAL_179:.*]] = arith.divsi %[[VAL_178]], %[[VAL_176]] : i64
! CHECK:             %[[VAL_180:.*]] = arith.cmpi sgt, %[[VAL_179]], %[[VAL_173]] : i64
! CHECK:             %[[VAL_181:.*]] = arith.select %[[VAL_180]], %[[VAL_179]], %[[VAL_173]] : i64
! CHECK:             %[[VAL_182:.*]] = arith.subi %[[VAL_157]], %[[VAL_19]] : index
! CHECK:             %[[VAL_183:.*]] = arith.divsi %[[VAL_182]], %[[VAL_30]] : index
! CHECK:             %[[VAL_184:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_185:.*]] = arith.addi %[[VAL_183]], %[[VAL_184]] : index
! CHECK:             %[[VAL_186:.*]] = arith.subi %[[VAL_161]], %[[VAL_32]] : index
! CHECK:             %[[VAL_187:.*]] = arith.divsi %[[VAL_186]], %[[VAL_43]] : index
! CHECK:             %[[VAL_188:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_189:.*]] = arith.addi %[[VAL_187]], %[[VAL_188]] : index
! CHECK:             %[[VAL_190:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_191:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_190]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             %[[VAL_192:.*]] = fir.load %[[VAL_191]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             %[[VAL_193:.*]] = fir.convert %[[VAL_192]] : (!fir.heap<!fir.array<?xi8>>) -> !fir.ref<!fir.array<?x?xtuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>>
! CHECK:             %[[VAL_194:.*]] = fir.shape %[[VAL_172]], %[[VAL_181]] : (i64, i64) -> !fir.shape<2>
! CHECK:             %[[VAL_195:.*]] = fir.array_coor %[[VAL_193]](%[[VAL_194]]) %[[VAL_185]], %[[VAL_189]] : (!fir.ref<!fir.array<?x?xtuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>>, !fir.shape<2>, index, index) -> !fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>
! CHECK:             %[[VAL_196:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_197:.*]] = fir.coordinate_of %[[VAL_195]], %[[VAL_196]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             %[[VAL_198:.*]] = fir.load %[[VAL_197]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             %[[VAL_199:.*]] = fir.convert %[[VAL_198]] : (!fir.heap<!fir.array<?xi8>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK:             %[[VAL_200:.*]] = arith.constant 2 : i32
! CHECK:             %[[VAL_201:.*]] = fir.coordinate_of %[[VAL_195]], %[[VAL_200]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi64>>>
! CHECK:             %[[VAL_202:.*]] = fir.load %[[VAL_201]] : !fir.ref<!fir.heap<!fir.array<?xi64>>>
! CHECK:             %[[VAL_203:.*]] = arith.constant 0 : i32
! CHECK:             %[[VAL_204:.*]] = fir.coordinate_of %[[VAL_202]], %[[VAL_203]] : (!fir.heap<!fir.array<?xi64>>, i32) -> !fir.ref<i64>
! CHECK:             %[[VAL_205:.*]] = fir.load %[[VAL_204]] : !fir.ref<i64>
! CHECK:             %[[VAL_206:.*]] = fir.convert %[[VAL_205]] : (i64) -> index
! CHECK:             %[[VAL_207:.*]] = fir.shape %[[VAL_206]] : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_208:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_209:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! CHECK:             %[[VAL_210:.*]] = fir.convert %[[VAL_209]] : (i32) -> i64
! CHECK:             %[[VAL_211:.*]] = fir.convert %[[VAL_210]] : (i64) -> index
! CHECK:             %[[VAL_212:.*]] = arith.subi %[[VAL_211]], %[[VAL_208]] : index
! CHECK:             %[[VAL_213:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:             %[[VAL_214:.*]] = fir.convert %[[VAL_213]] : (i32) -> i64
! CHECK:             %[[VAL_215:.*]] = fir.convert %[[VAL_214]] : (i64) -> index
! CHECK:             %[[VAL_216:.*]] = arith.subi %[[VAL_215]], %[[VAL_208]] : index
! CHECK:             %[[VAL_217:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
! CHECK:             %[[VAL_218:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_219:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:             %[[VAL_220:.*]] = fir.convert %[[VAL_219]] : (i32) -> i64
! CHECK:             %[[VAL_221:.*]] = fir.convert %[[VAL_220]] : (i64) -> index
! CHECK:             %[[VAL_222:.*]] = arith.subi %[[VAL_221]], %[[VAL_218]] : index
! CHECK:             %[[VAL_223:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! CHECK:             %[[VAL_224:.*]] = fir.convert %[[VAL_223]] : (i32) -> i64
! CHECK:             %[[VAL_225:.*]] = fir.convert %[[VAL_224]] : (i64) -> index
! CHECK:             %[[VAL_226:.*]] = arith.subi %[[VAL_225]], %[[VAL_218]] : index
! CHECK:             %[[VAL_227:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
! CHECK:             %[[VAL_228:.*]] = arith.constant 3.140000e+00 : f32
! CHECK:             %[[VAL_229:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_230:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_231:.*]] = arith.subi %[[VAL_206]], %[[VAL_229]] : index
! CHECK:             %[[VAL_232:.*]] = fir.do_loop %[[VAL_233:.*]] = %[[VAL_230]] to %[[VAL_231]] step %[[VAL_229]] unordered iter_args(%[[VAL_234:.*]] = %[[VAL_162]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
! CHECK:               %[[VAL_235:.*]] = arith.constant 1 : index
! CHECK:               %[[VAL_236:.*]] = arith.addi %[[VAL_233]], %[[VAL_235]] : index
! CHECK:               %[[VAL_237:.*]] = fir.array_coor %[[VAL_199]](%[[VAL_207]]) %[[VAL_236]] : (!fir.ref<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
! CHECK:               %[[VAL_238:.*]] = fir.load %[[VAL_237]] : !fir.ref<i8>
! CHECK:               %[[VAL_239:.*]] = fir.convert %[[VAL_238]] : (i8) -> i1
! CHECK:               %[[VAL_240:.*]] = fir.if %[[VAL_239]] -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
! CHECK:                 %[[VAL_241:.*]] = fir.array_fetch %[[VAL_45]], %[[VAL_222]], %[[VAL_226]], %[[VAL_227]], %[[VAL_233]] : (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, index, index, !fir.field, index) -> f32
! CHECK:                 %[[VAL_242:.*]] = arith.divf %[[VAL_241]], %[[VAL_228]] {{.*}}: f32
! CHECK:                 %[[VAL_243:.*]] = fir.array_update %[[VAL_234]], %[[VAL_242]], %[[VAL_212]], %[[VAL_216]], %[[VAL_217]], %[[VAL_233]] : (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, f32, index, index, !fir.field, index) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
! CHECK:                 fir.result %[[VAL_243]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
! CHECK:               } else {
! CHECK:                 fir.result %[[VAL_234]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
! CHECK:               }
! CHECK:               fir.result %[[VAL_244:.*]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
! CHECK:             }
! CHECK:             fir.result %[[VAL_245:.*]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_246:.*]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_44]], %[[VAL_247:.*]] to %[[VAL_0]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>
! CHECK:         %[[VAL_248:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
! CHECK:         %[[VAL_249:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
! CHECK:         %[[VAL_250:.*]] = fir.do_loop %[[VAL_251:.*]] = %[[VAL_19]] to %[[VAL_29]] step %[[VAL_30]] unordered iter_args(%[[VAL_252:.*]] = %[[VAL_248]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
! CHECK:           %[[VAL_253:.*]] = fir.convert %[[VAL_251]] : (index) -> i32
! CHECK:           fir.store %[[VAL_253]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_254:.*]] = fir.do_loop %[[VAL_255:.*]] = %[[VAL_32]] to %[[VAL_42]] step %[[VAL_43]] unordered iter_args(%[[VAL_256:.*]] = %[[VAL_252]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
! CHECK:             %[[VAL_257:.*]] = fir.convert %[[VAL_255]] : (index) -> i32
! CHECK:             fir.store %[[VAL_257]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:             %[[VAL_258:.*]] = arith.constant 0 : i64
! CHECK:             %[[VAL_259:.*]] = fir.convert %[[VAL_19]] : (index) -> i64
! CHECK:             %[[VAL_260:.*]] = fir.convert %[[VAL_29]] : (index) -> i64
! CHECK:             %[[VAL_261:.*]] = fir.convert %[[VAL_30]] : (index) -> i64
! CHECK:             %[[VAL_262:.*]] = arith.subi %[[VAL_260]], %[[VAL_259]] : i64
! CHECK:             %[[VAL_263:.*]] = arith.addi %[[VAL_262]], %[[VAL_261]] : i64
! CHECK:             %[[VAL_264:.*]] = arith.divsi %[[VAL_263]], %[[VAL_261]] : i64
! CHECK:             %[[VAL_265:.*]] = arith.cmpi sgt, %[[VAL_264]], %[[VAL_258]] : i64
! CHECK:             %[[VAL_266:.*]] = arith.select %[[VAL_265]], %[[VAL_264]], %[[VAL_258]] : i64
! CHECK:             %[[VAL_267:.*]] = arith.constant 0 : i64
! CHECK:             %[[VAL_268:.*]] = fir.convert %[[VAL_32]] : (index) -> i64
! CHECK:             %[[VAL_269:.*]] = fir.convert %[[VAL_42]] : (index) -> i64
! CHECK:             %[[VAL_270:.*]] = fir.convert %[[VAL_43]] : (index) -> i64
! CHECK:             %[[VAL_271:.*]] = arith.subi %[[VAL_269]], %[[VAL_268]] : i64
! CHECK:             %[[VAL_272:.*]] = arith.addi %[[VAL_271]], %[[VAL_270]] : i64
! CHECK:             %[[VAL_273:.*]] = arith.divsi %[[VAL_272]], %[[VAL_270]] : i64
! CHECK:             %[[VAL_274:.*]] = arith.cmpi sgt, %[[VAL_273]], %[[VAL_267]] : i64
! CHECK:             %[[VAL_275:.*]] = arith.select %[[VAL_274]], %[[VAL_273]], %[[VAL_267]] : i64
! CHECK:             %[[VAL_276:.*]] = arith.subi %[[VAL_251]], %[[VAL_19]] : index
! CHECK:             %[[VAL_277:.*]] = arith.divsi %[[VAL_276]], %[[VAL_30]] : index
! CHECK:             %[[VAL_278:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_279:.*]] = arith.addi %[[VAL_277]], %[[VAL_278]] : index
! CHECK:             %[[VAL_280:.*]] = arith.subi %[[VAL_255]], %[[VAL_32]] : index
! CHECK:             %[[VAL_281:.*]] = arith.divsi %[[VAL_280]], %[[VAL_43]] : index
! CHECK:             %[[VAL_282:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_283:.*]] = arith.addi %[[VAL_281]], %[[VAL_282]] : index
! CHECK:             %[[VAL_284:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_285:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_284]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             %[[VAL_286:.*]] = fir.load %[[VAL_285]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             %[[VAL_287:.*]] = fir.convert %[[VAL_286]] : (!fir.heap<!fir.array<?xi8>>) -> !fir.ref<!fir.array<?x?xtuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>>
! CHECK:             %[[VAL_288:.*]] = fir.shape %[[VAL_266]], %[[VAL_275]] : (i64, i64) -> !fir.shape<2>
! CHECK:             %[[VAL_289:.*]] = fir.array_coor %[[VAL_287]](%[[VAL_288]]) %[[VAL_279]], %[[VAL_283]] : (!fir.ref<!fir.array<?x?xtuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>>, !fir.shape<2>, index, index) -> !fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>
! CHECK:             %[[VAL_290:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_291:.*]] = fir.coordinate_of %[[VAL_289]], %[[VAL_290]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             %[[VAL_292:.*]] = fir.load %[[VAL_291]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             %[[VAL_293:.*]] = fir.convert %[[VAL_292]] : (!fir.heap<!fir.array<?xi8>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK:             %[[VAL_294:.*]] = arith.constant 2 : i32
! CHECK:             %[[VAL_295:.*]] = fir.coordinate_of %[[VAL_289]], %[[VAL_294]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi64>>>
! CHECK:             %[[VAL_296:.*]] = fir.load %[[VAL_295]] : !fir.ref<!fir.heap<!fir.array<?xi64>>>
! CHECK:             %[[VAL_297:.*]] = arith.constant 0 : i32
! CHECK:             %[[VAL_298:.*]] = fir.coordinate_of %[[VAL_296]], %[[VAL_297]] : (!fir.heap<!fir.array<?xi64>>, i32) -> !fir.ref<i64>
! CHECK:             %[[VAL_299:.*]] = fir.load %[[VAL_298]] : !fir.ref<i64>
! CHECK:             %[[VAL_300:.*]] = fir.convert %[[VAL_299]] : (i64) -> index
! CHECK:             %[[VAL_301:.*]] = fir.shape %[[VAL_300]] : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_302:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_303:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:             %[[VAL_304:.*]] = fir.convert %[[VAL_303]] : (i32) -> i64
! CHECK:             %[[VAL_305:.*]] = fir.convert %[[VAL_304]] : (i64) -> index
! CHECK:             %[[VAL_306:.*]] = arith.subi %[[VAL_305]], %[[VAL_302]] : index
! CHECK:             %[[VAL_307:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:             %[[VAL_308:.*]] = fir.convert %[[VAL_307]] : (i32) -> i64
! CHECK:             %[[VAL_309:.*]] = fir.convert %[[VAL_308]] : (i64) -> index
! CHECK:             %[[VAL_310:.*]] = arith.subi %[[VAL_309]], %[[VAL_302]] : index
! CHECK:             %[[VAL_311:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
! CHECK:             %[[VAL_312:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_313:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:             %[[VAL_314:.*]] = fir.convert %[[VAL_313]] : (i32) -> i64
! CHECK:             %[[VAL_315:.*]] = fir.convert %[[VAL_314]] : (i64) -> index
! CHECK:             %[[VAL_316:.*]] = arith.subi %[[VAL_315]], %[[VAL_312]] : index
! CHECK:             %[[VAL_317:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:             %[[VAL_318:.*]] = fir.convert %[[VAL_317]] : (i32) -> i64
! CHECK:             %[[VAL_319:.*]] = fir.convert %[[VAL_318]] : (i64) -> index
! CHECK:             %[[VAL_320:.*]] = arith.subi %[[VAL_319]], %[[VAL_312]] : index
! CHECK:             %[[VAL_321:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
! CHECK:             %[[VAL_322:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_323:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_324:.*]] = arith.subi %[[VAL_300]], %[[VAL_322]] : index
! CHECK:             %[[VAL_325:.*]] = fir.do_loop %[[VAL_326:.*]] = %[[VAL_323]] to %[[VAL_324]] step %[[VAL_322]] unordered iter_args(%[[VAL_327:.*]] = %[[VAL_256]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
! CHECK:               %[[VAL_328:.*]] = arith.constant 1 : index
! CHECK:               %[[VAL_329:.*]] = arith.addi %[[VAL_326]], %[[VAL_328]] : index
! CHECK:               %[[VAL_330:.*]] = fir.array_coor %[[VAL_293]](%[[VAL_301]]) %[[VAL_329]] : (!fir.ref<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
! CHECK:               %[[VAL_331:.*]] = fir.load %[[VAL_330]] : !fir.ref<i8>
! CHECK:               %[[VAL_332:.*]] = fir.convert %[[VAL_331]] : (i8) -> i1
! CHECK:               %[[VAL_333:.*]] = fir.if %[[VAL_332]] -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
! CHECK:                 fir.result %[[VAL_327]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
! CHECK:               } else {
! CHECK:                 %[[VAL_334:.*]] = fir.array_fetch %[[VAL_249]], %[[VAL_316]], %[[VAL_320]], %[[VAL_321]], %[[VAL_326]] : (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, index, index, !fir.field, index) -> f32
! CHECK:                 %[[VAL_335:.*]] = arith.negf %[[VAL_334]] {{.*}}: f32
! CHECK:                 %[[VAL_336:.*]] = fir.array_update %[[VAL_327]], %[[VAL_335]], %[[VAL_306]], %[[VAL_310]], %[[VAL_311]], %[[VAL_326]] : (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, f32, index, index, !fir.field, index) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
! CHECK:                 fir.result %[[VAL_336]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
! CHECK:               }
! CHECK:               fir.result %[[VAL_337:.*]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
! CHECK:             }
! CHECK:             fir.result %[[VAL_338:.*]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_339:.*]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_248]], %[[VAL_340:.*]] to %[[VAL_0]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>
! CHECK:         %[[VAL_341:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_342:.*]] = fir.call @_FortranARaggedArrayDeallocate(%[[VAL_341]]) {{.*}}: (!fir.llvm_ptr<i8>) -> none
! CHECK:         return
! CHECK:       }
