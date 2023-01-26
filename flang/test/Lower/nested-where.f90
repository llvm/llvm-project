! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QQmain() {
program nested_where

  ! CHECK:  %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:  %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:  %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:  %[[VAL_3:.*]] = fir.alloca tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>
  ! CHECK:  %[[VAL_4:.*]] = fir.alloca tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>
  ! CHECK:  %[[VAL_5:.*]] = fir.address_of(@_QFEa) : !fir.ref<!fir.array<3xi32>>
  ! CHECK:  %[[VAL_6:.*]] = arith.constant 3 : index
  ! CHECK:  %[[VAL_7:.*]] = fir.address_of(@_QFEmask1) : !fir.ref<!fir.array<3x!fir.logical<4>>>
  ! CHECK:  %[[VAL_8:.*]] = arith.constant 3 : index
  ! CHECK:  %[[VAL_9:.*]] = fir.address_of(@_QFEmask2) : !fir.ref<!fir.array<3x!fir.logical<4>>>
  ! CHECK:  %[[VAL_10:.*]] = arith.constant 3 : index
  ! CHECK:  %[[VAL_11:.*]] = arith.constant 0 : i32
  ! CHECK:  %[[VAL_12:.*]] = arith.constant 0 : i64
  ! CHECK:  %[[VAL_13:.*]] = fir.coordinate_of %[[VAL_4]], %[[VAL_11]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<i64>
  ! CHECK:  fir.store %[[VAL_12]] to %[[VAL_13]] : !fir.ref<i64>
  ! CHECK:  %[[VAL_14:.*]] = arith.constant 1 : i32
  ! CHECK:  %[[VAL_15:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi8>>
  ! CHECK:  %[[VAL_16:.*]] = fir.coordinate_of %[[VAL_4]], %[[VAL_14]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:  fir.store %[[VAL_15]] to %[[VAL_16]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:  %[[VAL_17:.*]] = arith.constant 2 : i32
  ! CHECK:  %[[VAL_18:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi64>>
  ! CHECK:  %[[VAL_19:.*]] = fir.coordinate_of %[[VAL_4]], %[[VAL_17]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi64>>>
  ! CHECK:  fir.store %[[VAL_18]] to %[[VAL_19]] : !fir.ref<!fir.heap<!fir.array<?xi64>>>
  ! CHECK:  %[[VAL_20:.*]] = arith.constant 0 : i32
  ! CHECK:  %[[VAL_21:.*]] = arith.constant 0 : i64
  ! CHECK:  %[[VAL_22:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_20]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<i64>
  ! CHECK:  fir.store %[[VAL_21]] to %[[VAL_22]] : !fir.ref<i64>
  ! CHECK:  %[[VAL_23:.*]] = arith.constant 1 : i32
  ! CHECK:  %[[VAL_24:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi8>>
  ! CHECK:  %[[VAL_25:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_23]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:  fir.store %[[VAL_24]] to %[[VAL_25]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:  %[[VAL_26:.*]] = arith.constant 2 : i32
  ! CHECK:  %[[VAL_27:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi64>>
  ! CHECK:  %[[VAL_28:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_26]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi64>>>
  ! CHECK:  fir.store %[[VAL_27]] to %[[VAL_28]] : !fir.ref<!fir.heap<!fir.array<?xi64>>>
  ! CHECK:  %[[VAL_29:.*]] = arith.constant 1 : i32
  ! CHECK:  %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> index
  ! CHECK:  %[[VAL_31:.*]] = arith.constant 3 : i32
  ! CHECK:  %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> index
  ! CHECK:  %[[VAL_33:.*]] = arith.constant 1 : index
  ! CHECK:  %[[VAL_34:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
  ! CHECK:  %[[VAL_35:.*]] = fir.array_load %[[VAL_5]](%[[VAL_34]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.array<3xi32>
  ! CHECK:  %[[VAL_36:.*]] = fir.do_loop %[[VAL_37:.*]] = %[[VAL_30]] to %[[VAL_32]] step %[[VAL_33]] unordered iter_args(%[[VAL_38:.*]] = %[[VAL_35]]) -> (!fir.array<3xi32>) {
  ! CHECK:    %[[VAL_39:.*]] = fir.convert %[[VAL_37]] : (index) -> i32
  ! CHECK:    fir.store %[[VAL_39]] to %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:    %[[VAL_40:.*]] = arith.constant 1 : i64
  ! CHECK:    %[[VAL_41:.*]] = arith.constant 0 : i64
  ! CHECK:    %[[VAL_42:.*]] = fir.convert %[[VAL_30]] : (index) -> i64
  ! CHECK:    %[[VAL_43:.*]] = fir.convert %[[VAL_32]] : (index) -> i64
  ! CHECK:    %[[VAL_44:.*]] = fir.convert %[[VAL_33]] : (index) -> i64
  ! CHECK:    %[[VAL_45:.*]] = arith.subi %[[VAL_43]], %[[VAL_42]] : i64
  ! CHECK:    %[[VAL_46:.*]] = arith.addi %[[VAL_45]], %[[VAL_44]] : i64
  ! CHECK:    %[[VAL_47:.*]] = arith.divsi %[[VAL_46]], %[[VAL_44]] : i64
  ! CHECK:    %[[VAL_48:.*]] = arith.cmpi sgt, %[[VAL_47]], %[[VAL_41]] : i64
  ! CHECK:    %[[VAL_49:.*]] = arith.select %[[VAL_48]], %[[VAL_47]], %[[VAL_41]] : i64
  ! CHECK:    %[[VAL_50:.*]] = arith.constant 1 : i32
  ! CHECK:    %[[VAL_51:.*]] = fir.coordinate_of %[[VAL_4]], %[[VAL_50]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_52:.*]] = fir.load %[[VAL_51]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_53:.*]] = fir.convert %[[VAL_52]] : (!fir.heap<!fir.array<?xi8>>) -> i64
  ! CHECK:    %[[VAL_54:.*]] = arith.constant 0 : i64
  ! CHECK:    %[[VAL_55:.*]] = arith.cmpi eq, %[[VAL_53]], %[[VAL_54]] : i64
  ! CHECK:    fir.if %[[VAL_55]] {
  ! CHECK:      %[[VAL_56:.*]] = arith.constant true
  ! CHECK:      %[[VAL_57:.*]] = arith.constant 1 : i64
  ! CHECK:      %[[VAL_58:.*]] = fir.allocmem !fir.array<1xi64>
  ! CHECK:      %[[VAL_59:.*]] = arith.constant 0 : i32
  ! CHECK:      %[[VAL_60:.*]] = fir.coordinate_of %[[VAL_58]], %[[VAL_59]] : (!fir.heap<!fir.array<1xi64>>, i32) -> !fir.ref<i64>
  ! CHECK:      fir.store %[[VAL_49]] to %[[VAL_60]] : !fir.ref<i64>
  ! CHECK:      %[[VAL_61:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>) -> !fir.llvm_ptr<i8>
  ! CHECK:      %[[VAL_62:.*]] = fir.convert %[[VAL_58]] : (!fir.heap<!fir.array<1xi64>>) -> !fir.ref<i64>
  ! CHECK:      %[[VAL_63:.*]] = fir.call @_FortranARaggedArrayAllocate(%[[VAL_61]], %[[VAL_56]], %[[VAL_57]], %[[VAL_40]], %[[VAL_62]]) {{.*}}: (!fir.llvm_ptr<i8>, i1, i64, i64, !fir.ref<i64>) -> !fir.llvm_ptr<i8>
  ! CHECK:    }
  ! CHECK:    %[[VAL_64:.*]] = arith.subi %[[VAL_37]], %[[VAL_30]] : index
  ! CHECK:    %[[VAL_65:.*]] = arith.divsi %[[VAL_64]], %[[VAL_33]] : index
  ! CHECK:    %[[VAL_66:.*]] = arith.constant 1 : index
  ! CHECK:    %[[VAL_67:.*]] = arith.addi %[[VAL_65]], %[[VAL_66]] : index
  ! CHECK:    %[[VAL_68:.*]] = arith.constant 1 : i32
  ! CHECK:    %[[VAL_69:.*]] = fir.coordinate_of %[[VAL_4]], %[[VAL_68]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_70:.*]] = fir.load %[[VAL_69]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_71:.*]] = fir.convert %[[VAL_70]] : (!fir.heap<!fir.array<?xi8>>) -> !fir.ref<!fir.array<?xtuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>>
  ! CHECK:    %[[VAL_72:.*]] = fir.shape %[[VAL_49]] : (i64) -> !fir.shape<1>
  ! CHECK:    %[[VAL_73:.*]] = fir.array_coor %[[VAL_71]](%[[VAL_72]]) %[[VAL_67]] : (!fir.ref<!fir.array<?xtuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>>, !fir.shape<1>, index) -> !fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>
  ! CHECK:    %[[VAL_74:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
  ! CHECK:    %[[VAL_75:.*]] = fir.array_load %[[VAL_7]](%[[VAL_74]]) : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<3x!fir.logical<4>>
  ! CHECK:    %[[VAL_76:.*]] = arith.constant 1 : i32
  ! CHECK:    %[[VAL_77:.*]] = fir.coordinate_of %[[VAL_73]], %[[VAL_76]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_78:.*]] = fir.load %[[VAL_77]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_79:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
  ! CHECK:    %[[VAL_80:.*]] = fir.array_load %[[VAL_78]](%[[VAL_79]]) : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>) -> !fir.array<?xi8>
  ! CHECK:    %[[VAL_81:.*]] = arith.constant 1 : i64
  ! CHECK:    %[[VAL_82:.*]] = arith.constant 1 : i32
  ! CHECK:    %[[VAL_83:.*]] = fir.coordinate_of %[[VAL_73]], %[[VAL_82]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_84:.*]] = fir.load %[[VAL_83]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_85:.*]] = fir.convert %[[VAL_84]] : (!fir.heap<!fir.array<?xi8>>) -> i64
  ! CHECK:    %[[VAL_86:.*]] = arith.constant 0 : i64
  ! CHECK:    %[[VAL_87:.*]] = arith.cmpi eq, %[[VAL_85]], %[[VAL_86]] : i64
  ! CHECK:    fir.if %[[VAL_87]] {
  ! CHECK:      %[[VAL_88:.*]] = arith.constant false
  ! CHECK:      %[[VAL_89:.*]] = arith.constant 1 : i64
  ! CHECK:      %[[VAL_90:.*]] = fir.allocmem !fir.array<1xi64>
  ! CHECK:      %[[VAL_91:.*]] = arith.constant 0 : i32
  ! CHECK:      %[[VAL_92:.*]] = fir.coordinate_of %[[VAL_90]], %[[VAL_91]] : (!fir.heap<!fir.array<1xi64>>, i32) -> !fir.ref<i64>
  ! CHECK:      %[[VAL_93:.*]] = fir.convert %[[VAL_8]] : (index) -> i64
  ! CHECK:      fir.store %[[VAL_93]] to %[[VAL_92]] : !fir.ref<i64>
  ! CHECK:      %[[VAL_94:.*]] = fir.convert %[[VAL_73]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>) -> !fir.llvm_ptr<i8>
  ! CHECK:      %[[VAL_95:.*]] = fir.convert %[[VAL_90]] : (!fir.heap<!fir.array<1xi64>>) -> !fir.ref<i64>
  ! CHECK:      %[[VAL_96:.*]] = fir.call @_FortranARaggedArrayAllocate(%[[VAL_94]], %[[VAL_88]], %[[VAL_89]], %[[VAL_81]], %[[VAL_95]]) {{.*}}: (!fir.llvm_ptr<i8>, i1, i64, i64, !fir.ref<i64>) -> !fir.llvm_ptr<i8>
  ! CHECK:    }
  ! CHECK:    %[[VAL_97:.*]] = arith.constant 1 : index
  ! CHECK:    %[[VAL_98:.*]] = arith.constant 0 : index
  ! CHECK:    %[[VAL_99:.*]] = arith.subi %[[VAL_8]], %[[VAL_97]] : index
  ! CHECK:    %[[VAL_100:.*]] = fir.do_loop %[[VAL_101:.*]] = %[[VAL_98]] to %[[VAL_99]] step %[[VAL_97]] unordered iter_args(%[[VAL_102:.*]] = %[[VAL_80]]) -> (!fir.array<?xi8>) {
  ! CHECK:      %[[VAL_103:.*]] = fir.array_fetch %[[VAL_75]], %[[VAL_101]] : (!fir.array<3x!fir.logical<4>>, index) -> !fir.logical<4>
  ! CHECK:      %[[VAL_104:.*]] = arith.constant 1 : i32
  ! CHECK:      %[[VAL_105:.*]] = fir.coordinate_of %[[VAL_73]], %[[VAL_104]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:      %[[VAL_106:.*]] = fir.load %[[VAL_105]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:      %[[VAL_107:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
  ! CHECK:      %[[VAL_108:.*]] = arith.constant 1 : index
  ! CHECK:      %[[VAL_109:.*]] = arith.addi %[[VAL_101]], %[[VAL_108]] : index
  ! CHECK:      %[[VAL_110:.*]] = fir.array_coor %[[VAL_106]](%[[VAL_107]]) %[[VAL_109]] : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
  ! CHECK:      %[[VAL_111:.*]] = fir.convert %[[VAL_103]] : (!fir.logical<4>) -> i8
  ! CHECK:      fir.store %[[VAL_111]] to %[[VAL_110]] : !fir.ref<i8>
  ! CHECK:      fir.result %[[VAL_102]] : !fir.array<?xi8>
  ! CHECK:    }
  ! CHECK:    fir.result %[[VAL_38]] : !fir.array<3xi32>
  ! CHECK:  }
  ! CHECK:  %[[VAL_112:.*]] = fir.do_loop %[[VAL_113:.*]] = %[[VAL_30]] to %[[VAL_32]] step %[[VAL_33]] unordered iter_args(%[[VAL_114:.*]] = %[[VAL_35]]) -> (!fir.array<3xi32>) {
  ! CHECK:    %[[VAL_115:.*]] = fir.convert %[[VAL_113]] : (index) -> i32
  ! CHECK:    fir.store %[[VAL_115]] to %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:    %[[VAL_116:.*]] = arith.constant 1 : i64
  ! CHECK:    %[[VAL_117:.*]] = arith.constant 0 : i64
  ! CHECK:    %[[VAL_118:.*]] = fir.convert %[[VAL_30]] : (index) -> i64
  ! CHECK:    %[[VAL_119:.*]] = fir.convert %[[VAL_32]] : (index) -> i64
  ! CHECK:    %[[VAL_120:.*]] = fir.convert %[[VAL_33]] : (index) -> i64
  ! CHECK:    %[[VAL_121:.*]] = arith.subi %[[VAL_119]], %[[VAL_118]] : i64
  ! CHECK:    %[[VAL_122:.*]] = arith.addi %[[VAL_121]], %[[VAL_120]] : i64
  ! CHECK:    %[[VAL_123:.*]] = arith.divsi %[[VAL_122]], %[[VAL_120]] : i64
  ! CHECK:    %[[VAL_124:.*]] = arith.cmpi sgt, %[[VAL_123]], %[[VAL_117]] : i64
  ! CHECK:    %[[VAL_125:.*]] = arith.select %[[VAL_124]], %[[VAL_123]], %[[VAL_117]] : i64
  ! CHECK:    %[[VAL_126:.*]] = arith.constant 1 : i32
  ! CHECK:    %[[VAL_127:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_126]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_128:.*]] = fir.load %[[VAL_127]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_129:.*]] = fir.convert %[[VAL_128]] : (!fir.heap<!fir.array<?xi8>>) -> i64
  ! CHECK:    %[[VAL_130:.*]] = arith.constant 0 : i64
  ! CHECK:    %[[VAL_131:.*]] = arith.cmpi eq, %[[VAL_129]], %[[VAL_130]] : i64
  ! CHECK:    fir.if %[[VAL_131]] {
  ! CHECK:      %[[VAL_132:.*]] = arith.constant true
  ! CHECK:      %[[VAL_133:.*]] = arith.constant 1 : i64
  ! CHECK:      %[[VAL_134:.*]] = fir.allocmem !fir.array<1xi64>
  ! CHECK:      %[[VAL_135:.*]] = arith.constant 0 : i32
  ! CHECK:      %[[VAL_136:.*]] = fir.coordinate_of %[[VAL_134]], %[[VAL_135]] : (!fir.heap<!fir.array<1xi64>>, i32) -> !fir.ref<i64>
  ! CHECK:      fir.store %[[VAL_125]] to %[[VAL_136]] : !fir.ref<i64>
  ! CHECK:      %[[VAL_137:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>) -> !fir.llvm_ptr<i8>
  ! CHECK:      %[[VAL_138:.*]] = fir.convert %[[VAL_134]] : (!fir.heap<!fir.array<1xi64>>) -> !fir.ref<i64>
  ! CHECK:      %[[VAL_139:.*]] = fir.call @_FortranARaggedArrayAllocate(%[[VAL_137]], %[[VAL_132]], %[[VAL_133]], %[[VAL_116]], %[[VAL_138]]) {{.*}}: (!fir.llvm_ptr<i8>, i1, i64, i64, !fir.ref<i64>) -> !fir.llvm_ptr<i8>
  ! CHECK:    }
  ! CHECK:    %[[VAL_140:.*]] = arith.subi %[[VAL_113]], %[[VAL_30]] : index
  ! CHECK:    %[[VAL_141:.*]] = arith.divsi %[[VAL_140]], %[[VAL_33]] : index
  ! CHECK:    %[[VAL_142:.*]] = arith.constant 1 : index
  ! CHECK:    %[[VAL_143:.*]] = arith.addi %[[VAL_141]], %[[VAL_142]] : index
  ! CHECK:    %[[VAL_144:.*]] = arith.constant 1 : i32
  ! CHECK:    %[[VAL_145:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_144]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_146:.*]] = fir.load %[[VAL_145]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_147:.*]] = fir.convert %[[VAL_146]] : (!fir.heap<!fir.array<?xi8>>) -> !fir.ref<!fir.array<?xtuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>>
  ! CHECK:    %[[VAL_148:.*]] = fir.shape %[[VAL_125]] : (i64) -> !fir.shape<1>
  ! CHECK:    %[[VAL_149:.*]] = fir.array_coor %[[VAL_147]](%[[VAL_148]]) %[[VAL_143]] : (!fir.ref<!fir.array<?xtuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>>, !fir.shape<1>, index) -> !fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>
  ! CHECK:    %[[VAL_150:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
  ! CHECK:    %[[VAL_151:.*]] = fir.array_load %[[VAL_9]](%[[VAL_150]]) : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<3x!fir.logical<4>>
  ! CHECK:    %[[VAL_152:.*]] = arith.constant 1 : i32
  ! CHECK:    %[[VAL_153:.*]] = fir.coordinate_of %[[VAL_149]], %[[VAL_152]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_154:.*]] = fir.load %[[VAL_153]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_155:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
  ! CHECK:    %[[VAL_156:.*]] = fir.array_load %[[VAL_154]](%[[VAL_155]]) : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>) -> !fir.array<?xi8>
  ! CHECK:    %[[VAL_157:.*]] = arith.constant 1 : i64
  ! CHECK:    %[[VAL_158:.*]] = arith.constant 1 : i32
  ! CHECK:    %[[VAL_159:.*]] = fir.coordinate_of %[[VAL_149]], %[[VAL_158]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_160:.*]] = fir.load %[[VAL_159]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_161:.*]] = fir.convert %[[VAL_160]] : (!fir.heap<!fir.array<?xi8>>) -> i64
  ! CHECK:    %[[VAL_162:.*]] = arith.constant 0 : i64
  ! CHECK:    %[[VAL_163:.*]] = arith.cmpi eq, %[[VAL_161]], %[[VAL_162]] : i64
  ! CHECK:    fir.if %[[VAL_163]] {
  ! CHECK:      %[[VAL_164:.*]] = arith.constant false
  ! CHECK:      %[[VAL_165:.*]] = arith.constant 1 : i64
  ! CHECK:      %[[VAL_166:.*]] = fir.allocmem !fir.array<1xi64>
  ! CHECK:      %[[VAL_167:.*]] = arith.constant 0 : i32
  ! CHECK:      %[[VAL_168:.*]] = fir.coordinate_of %[[VAL_166]], %[[VAL_167]] : (!fir.heap<!fir.array<1xi64>>, i32) -> !fir.ref<i64>
  ! CHECK:      %[[VAL_169:.*]] = fir.convert %[[VAL_10]] : (index) -> i64
  ! CHECK:      fir.store %[[VAL_169]] to %[[VAL_168]] : !fir.ref<i64>
  ! CHECK:      %[[VAL_170:.*]] = fir.convert %[[VAL_149]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>) -> !fir.llvm_ptr<i8>
  ! CHECK:      %[[VAL_171:.*]] = fir.convert %[[VAL_166]] : (!fir.heap<!fir.array<1xi64>>) -> !fir.ref<i64>
  ! CHECK:      %[[VAL_172:.*]] = fir.call @_FortranARaggedArrayAllocate(%[[VAL_170]], %[[VAL_164]], %[[VAL_165]], %[[VAL_157]], %[[VAL_171]]) {{.*}}: (!fir.llvm_ptr<i8>, i1, i64, i64, !fir.ref<i64>) -> !fir.llvm_ptr<i8>
  ! CHECK:    }
  ! CHECK:    %[[VAL_173:.*]] = arith.constant 1 : index
  ! CHECK:    %[[VAL_174:.*]] = arith.constant 0 : index
  ! CHECK:    %[[VAL_175:.*]] = arith.subi %[[VAL_10]], %[[VAL_173]] : index
  ! CHECK:    %[[VAL_176:.*]] = fir.do_loop %[[VAL_177:.*]] = %[[VAL_174]] to %[[VAL_175]] step %[[VAL_173]] unordered iter_args(%[[VAL_178:.*]] = %[[VAL_156]]) -> (!fir.array<?xi8>) {
  ! CHECK:      %[[VAL_179:.*]] = fir.array_fetch %[[VAL_151]], %[[VAL_177]] : (!fir.array<3x!fir.logical<4>>, index) -> !fir.logical<4>
  ! CHECK:      %[[VAL_180:.*]] = arith.constant 1 : i32
  ! CHECK:      %[[VAL_181:.*]] = fir.coordinate_of %[[VAL_149]], %[[VAL_180]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:      %[[VAL_182:.*]] = fir.load %[[VAL_181]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:      %[[VAL_183:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
  ! CHECK:      %[[VAL_184:.*]] = arith.constant 1 : index
  ! CHECK:      %[[VAL_185:.*]] = arith.addi %[[VAL_177]], %[[VAL_184]] : index
  ! CHECK:      %[[VAL_186:.*]] = fir.array_coor %[[VAL_182]](%[[VAL_183]]) %[[VAL_185]] : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
  ! CHECK:      %[[VAL_187:.*]] = fir.convert %[[VAL_179]] : (!fir.logical<4>) -> i8
  ! CHECK:      fir.store %[[VAL_187]] to %[[VAL_186]] : !fir.ref<i8>
  ! CHECK:      fir.result %[[VAL_178]] : !fir.array<?xi8>
  ! CHECK:    }
  ! CHECK:    fir.result %[[VAL_114]] : !fir.array<3xi32>
  ! CHECK:  }
  ! CHECK:  %[[VAL_188:.*]] = fir.do_loop %[[VAL_189:.*]] = %[[VAL_30]] to %[[VAL_32]] step %[[VAL_33]] unordered iter_args(%[[VAL_190:.*]] = %[[VAL_35]]) -> (!fir.array<3xi32>) {
  ! CHECK:    %[[VAL_191:.*]] = fir.convert %[[VAL_189]] : (index) -> i32
  ! CHECK:    fir.store %[[VAL_191]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:    %[[VAL_192:.*]] = arith.constant 0 : i64
  ! CHECK:    %[[VAL_193:.*]] = fir.convert %[[VAL_30]] : (index) -> i64
  ! CHECK:    %[[VAL_194:.*]] = fir.convert %[[VAL_32]] : (index) -> i64
  ! CHECK:    %[[VAL_195:.*]] = fir.convert %[[VAL_33]] : (index) -> i64
  ! CHECK:    %[[VAL_196:.*]] = arith.subi %[[VAL_194]], %[[VAL_193]] : i64
  ! CHECK:    %[[VAL_197:.*]] = arith.addi %[[VAL_196]], %[[VAL_195]] : i64
  ! CHECK:    %[[VAL_198:.*]] = arith.divsi %[[VAL_197]], %[[VAL_195]] : i64
  ! CHECK:    %[[VAL_199:.*]] = arith.cmpi sgt, %[[VAL_198]], %[[VAL_192]] : i64
  ! CHECK:    %[[VAL_200:.*]] = arith.select %[[VAL_199]], %[[VAL_198]], %[[VAL_192]] : i64
  ! CHECK:    %[[VAL_201:.*]] = arith.subi %[[VAL_189]], %[[VAL_30]] : index
  ! CHECK:    %[[VAL_202:.*]] = arith.divsi %[[VAL_201]], %[[VAL_33]] : index
  ! CHECK:    %[[VAL_203:.*]] = arith.constant 1 : index
  ! CHECK:    %[[VAL_204:.*]] = arith.addi %[[VAL_202]], %[[VAL_203]] : index
  ! CHECK:    %[[VAL_205:.*]] = arith.constant 1 : i32
  ! CHECK:    %[[VAL_206:.*]] = fir.coordinate_of %[[VAL_4]], %[[VAL_205]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_207:.*]] = fir.load %[[VAL_206]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_208:.*]] = fir.convert %[[VAL_207]] : (!fir.heap<!fir.array<?xi8>>) -> !fir.ref<!fir.array<?xtuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>>
  ! CHECK:    %[[VAL_209:.*]] = fir.shape %[[VAL_200]] : (i64) -> !fir.shape<1>
  ! CHECK:    %[[VAL_210:.*]] = fir.array_coor %[[VAL_208]](%[[VAL_209]]) %[[VAL_204]] : (!fir.ref<!fir.array<?xtuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>>, !fir.shape<1>, index) -> !fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>
  ! CHECK:    %[[VAL_211:.*]] = arith.constant 1 : i32
  ! CHECK:    %[[VAL_212:.*]] = fir.coordinate_of %[[VAL_210]], %[[VAL_211]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_213:.*]] = fir.load %[[VAL_212]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_214:.*]] = fir.convert %[[VAL_213]] : (!fir.heap<!fir.array<?xi8>>) -> !fir.ref<!fir.array<?xi8>>
  ! CHECK:    %[[VAL_215:.*]] = arith.constant 2 : i32
  ! CHECK:    %[[VAL_216:.*]] = fir.coordinate_of %[[VAL_210]], %[[VAL_215]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi64>>>
  ! CHECK:    %[[VAL_217:.*]] = fir.load %[[VAL_216]] : !fir.ref<!fir.heap<!fir.array<?xi64>>>
  ! CHECK:    %[[VAL_218:.*]] = arith.constant 0 : i32
  ! CHECK:    %[[VAL_219:.*]] = fir.coordinate_of %[[VAL_217]], %[[VAL_218]] : (!fir.heap<!fir.array<?xi64>>, i32) -> !fir.ref<i64>
  ! CHECK:    %[[VAL_220:.*]] = fir.load %[[VAL_219]] : !fir.ref<i64>
  ! CHECK:    %[[VAL_221:.*]] = fir.convert %[[VAL_220]] : (i64) -> index
  ! CHECK:    %[[VAL_222:.*]] = fir.shape %[[VAL_221]] : (index) -> !fir.shape<1>
  ! CHECK:    %[[VAL_223:.*]] = arith.constant 0 : i64
  ! CHECK:    %[[VAL_224:.*]] = fir.convert %[[VAL_30]] : (index) -> i64
  ! CHECK:    %[[VAL_225:.*]] = fir.convert %[[VAL_32]] : (index) -> i64
  ! CHECK:    %[[VAL_226:.*]] = fir.convert %[[VAL_33]] : (index) -> i64
  ! CHECK:    %[[VAL_227:.*]] = arith.subi %[[VAL_225]], %[[VAL_224]] : i64
  ! CHECK:    %[[VAL_228:.*]] = arith.addi %[[VAL_227]], %[[VAL_226]] : i64
  ! CHECK:    %[[VAL_229:.*]] = arith.divsi %[[VAL_228]], %[[VAL_226]] : i64
  ! CHECK:    %[[VAL_230:.*]] = arith.cmpi sgt, %[[VAL_229]], %[[VAL_223]] : i64
  ! CHECK:    %[[VAL_231:.*]] = arith.select %[[VAL_230]], %[[VAL_229]], %[[VAL_223]] : i64
  ! CHECK:    %[[VAL_232:.*]] = arith.subi %[[VAL_189]], %[[VAL_30]] : index
  ! CHECK:    %[[VAL_233:.*]] = arith.divsi %[[VAL_232]], %[[VAL_33]] : index
  ! CHECK:    %[[VAL_234:.*]] = arith.constant 1 : index
  ! CHECK:    %[[VAL_235:.*]] = arith.addi %[[VAL_233]], %[[VAL_234]] : index
  ! CHECK:    %[[VAL_236:.*]] = arith.constant 1 : i32
  ! CHECK:    %[[VAL_237:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_236]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_238:.*]] = fir.load %[[VAL_237]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_239:.*]] = fir.convert %[[VAL_238]] : (!fir.heap<!fir.array<?xi8>>) -> !fir.ref<!fir.array<?xtuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>>
  ! CHECK:    %[[VAL_240:.*]] = fir.shape %[[VAL_231]] : (i64) -> !fir.shape<1>
  ! CHECK:    %[[VAL_241:.*]] = fir.array_coor %[[VAL_239]](%[[VAL_240]]) %[[VAL_235]] : (!fir.ref<!fir.array<?xtuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>>, !fir.shape<1>, index) -> !fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>
  ! CHECK:    %[[VAL_242:.*]] = arith.constant 1 : i32
  ! CHECK:    %[[VAL_243:.*]] = fir.coordinate_of %[[VAL_241]], %[[VAL_242]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_244:.*]] = fir.load %[[VAL_243]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:    %[[VAL_245:.*]] = fir.convert %[[VAL_244]] : (!fir.heap<!fir.array<?xi8>>) -> !fir.ref<!fir.array<?xi8>>
  ! CHECK:    %[[VAL_246:.*]] = arith.constant 2 : i32
  ! CHECK:    %[[VAL_247:.*]] = fir.coordinate_of %[[VAL_241]], %[[VAL_246]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi64>>>
  ! CHECK:    %[[VAL_248:.*]] = fir.load %[[VAL_247]] : !fir.ref<!fir.heap<!fir.array<?xi64>>>
  ! CHECK:    %[[VAL_249:.*]] = arith.constant 0 : i32
  ! CHECK:    %[[VAL_250:.*]] = fir.coordinate_of %[[VAL_248]], %[[VAL_249]] : (!fir.heap<!fir.array<?xi64>>, i32) -> !fir.ref<i64>
  ! CHECK:    %[[VAL_251:.*]] = fir.load %[[VAL_250]] : !fir.ref<i64>
  ! CHECK:    %[[VAL_252:.*]] = fir.convert %[[VAL_251]] : (i64) -> index
  ! CHECK:    %[[VAL_253:.*]] = fir.shape %[[VAL_252]] : (index) -> !fir.shape<1>
  ! CHECK:    %[[VAL_254:.*]] = arith.constant 1 : i32
  ! CHECK:    %[[VAL_255:.*]] = arith.constant 1 : index
  ! CHECK:    %[[VAL_256:.*]] = arith.constant 0 : index
  ! CHECK:    %[[VAL_257:.*]] = arith.subi %[[VAL_221]], %[[VAL_255]] : index
  ! CHECK:    %[[VAL_258:.*]] = fir.do_loop %[[VAL_259:.*]] = %[[VAL_256]] to %[[VAL_257]] step %[[VAL_255]] unordered iter_args(%[[VAL_260:.*]] = %[[VAL_190]]) -> (!fir.array<3xi32>) {
  ! CHECK:      %[[VAL_261:.*]] = arith.constant 1 : index
  ! CHECK:      %[[VAL_262:.*]] = arith.addi %[[VAL_259]], %[[VAL_261]] : index
  ! CHECK:      %[[VAL_263:.*]] = fir.array_coor %[[VAL_214]](%[[VAL_222]]) %[[VAL_262]] : (!fir.ref<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
  ! CHECK:      %[[VAL_264:.*]] = fir.load %[[VAL_263]] : !fir.ref<i8>
  ! CHECK:      %[[VAL_265:.*]] = fir.convert %[[VAL_264]] : (i8) -> i1
  ! CHECK:      %[[VAL_266:.*]] = fir.if %[[VAL_265]] -> (!fir.array<3xi32>) {
  ! CHECK:        %[[VAL_267:.*]] = arith.constant 1 : index
  ! CHECK:        %[[VAL_268:.*]] = arith.addi %[[VAL_259]], %[[VAL_267]] : index
  ! CHECK:        %[[VAL_269:.*]] = fir.array_coor %[[VAL_245]](%[[VAL_253]]) %[[VAL_268]] : (!fir.ref<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
  ! CHECK:        %[[VAL_270:.*]] = fir.load %[[VAL_269]] : !fir.ref<i8>
  ! CHECK:        %[[VAL_271:.*]] = fir.convert %[[VAL_270]] : (i8) -> i1
  ! CHECK:        %[[VAL_272:.*]] = fir.if %[[VAL_271]] -> (!fir.array<3xi32>) {
  ! CHECK:          %[[VAL_273:.*]] = fir.array_update %[[VAL_260]], %[[VAL_254]], %[[VAL_259]] : (!fir.array<3xi32>, i32, index) -> !fir.array<3xi32>
  ! CHECK:          fir.result %[[VAL_273]] : !fir.array<3xi32>
  ! CHECK:        } else {
  ! CHECK:          fir.result %[[VAL_260]] : !fir.array<3xi32>
  ! CHECK:        }
  ! CHECK:        fir.result %[[VAL_274:.*]] : !fir.array<3xi32>
  ! CHECK:      } else {
  ! CHECK:        fir.result %[[VAL_260]] : !fir.array<3xi32>
  ! CHECK:      }
  ! CHECK:      fir.result %[[VAL_275:.*]] : !fir.array<3xi32>
  ! CHECK:    }
  ! CHECK:    fir.result %[[VAL_276:.*]] : !fir.array<3xi32>
  ! CHECK:  }
  ! CHECK:  fir.array_merge_store %[[VAL_35]], %[[VAL_277:.*]] to %[[VAL_5]] : !fir.array<3xi32>, !fir.array<3xi32>, !fir.ref<!fir.array<3xi32>>
  ! CHECK:  %[[VAL_278:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>) -> !fir.llvm_ptr<i8>
  ! CHECK:  %[[VAL_279:.*]] = fir.call @_FortranARaggedArrayDeallocate(%[[VAL_278]]) {{.*}}: (!fir.llvm_ptr<i8>) -> none
  ! CHECK:  %[[VAL_280:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>) -> !fir.llvm_ptr<i8>
  ! CHECK:  %[[VAL_281:.*]] = fir.call @_FortranARaggedArrayDeallocate(%[[VAL_280]]) {{.*}}: (!fir.llvm_ptr<i8>) -> none
  
  integer :: a(3) = 0
  logical :: mask1(3) = (/ .true.,.false.,.true. /)
  logical :: mask2(3) = (/ .true.,.true.,.false. /)
  forall (i=1:3)
    where (mask1)
      where (mask2)
        a = 1
      end where
    endwhere
  end forall
  ! CHECK:  return
  ! CHECK: }
end program nested_where
