  ! RUN: bbc -emit-fir %s -o - | FileCheck %s

  ! CHECK-LABEL: func @_QQmain() {
  ! CHECK:         %[[VAL_0:.*]] = fir.address_of(@_QFEa) : !fir.ref<!fir.array<10xf32>>
  ! CHECK:         %[[VAL_1:.*]] = arith.constant 10 : index
  ! CHECK:         %[[VAL_2:.*]] = fir.address_of(@_QFEb) : !fir.ref<!fir.array<10xf32>>
  ! CHECK:         %[[VAL_3:.*]] = arith.constant 10 : index
  ! CHECK:         %[[VAL_5:.*]] = arith.constant 10 : index
  ! CHECK:         %[[VAL_6:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_7:.*]] = fir.array_load %[[VAL_0]](%[[VAL_6]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK:         %[[VAL_8:.*]] = arith.constant 4.000000e+00 : f32
  ! CHECK:         %[[VAL_9:.*]] = fir.allocmem !fir.array<10x!fir.logical<4>>
  ! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_9]](%[[VAL_10]]) : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
  ! CHECK:         %[[VAL_12:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_13:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_14:.*]] = arith.subi %[[VAL_5]], %[[VAL_12]] : index
  ! CHECK:         %[[VAL_15:.*]] = fir.do_loop %[[VAL_16:.*]] = %[[VAL_13]] to %[[VAL_14]] step %[[VAL_12]] unordered iter_args(%[[VAL_17:.*]] = %[[VAL_11]]) -> (!fir.array<10x!fir.logical<4>>) {
  ! CHECK:           %[[VAL_18:.*]] = fir.array_fetch %[[VAL_7]], %[[VAL_16]] : (!fir.array<10xf32>, index) -> f32
  ! CHECK:           %[[VAL_19:.*]] = arith.cmpf ogt, %[[VAL_18]], %[[VAL_8]] : f32
  ! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i1) -> !fir.logical<4>
  ! CHECK:           %[[VAL_21:.*]] = fir.array_update %[[VAL_17]], %[[VAL_20]], %[[VAL_16]] : (!fir.array<10x!fir.logical<4>>, !fir.logical<4>, index) -> !fir.array<10x!fir.logical<4>>
  ! CHECK:           fir.result %[[VAL_21]] : !fir.array<10x!fir.logical<4>>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_11]], %[[VAL_22:.*]] to %[[VAL_9]] : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.heap<!fir.array<10x!fir.logical<4>>>
  ! CHECK:         %[[VAL_23:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_24:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_25:.*]] = fir.array_load %[[VAL_2]](%[[VAL_24]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK:         %[[VAL_26:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_27:.*]] = fir.array_load %[[VAL_0]](%[[VAL_26]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK:         %[[VAL_28:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_29:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_30:.*]] = arith.subi %[[VAL_3]], %[[VAL_28]] : index
  ! CHECK:         %[[VAL_31:.*]] = fir.do_loop %[[VAL_32:.*]] = %[[VAL_29]] to %[[VAL_30]] step %[[VAL_28]] unordered iter_args(%[[VAL_33:.*]] = %[[VAL_25]]) -> (!fir.array<10xf32>) {
  ! CHECK:           %[[VAL_34:.*]] = arith.constant 1 : index
  ! CHECK:           %[[VAL_35:.*]] = arith.addi %[[VAL_32]], %[[VAL_34]] : index
  ! CHECK:           %[[VAL_36:.*]] = fir.array_coor %[[VAL_9]](%[[VAL_23]]) %[[VAL_35]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:           %[[VAL_37:.*]] = fir.load %[[VAL_36]] : !fir.ref<!fir.logical<4>>
  ! CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (!fir.logical<4>) -> i1
  ! CHECK:           %[[VAL_39:.*]] = fir.if %[[VAL_38]] -> (!fir.array<10xf32>) {
  ! CHECK:             %[[VAL_40:.*]] = fir.array_fetch %[[VAL_27]], %[[VAL_32]] : (!fir.array<10xf32>, index) -> f32
  ! CHECK:             %[[VAL_41:.*]] = arith.negf %[[VAL_40]] {{.*}}: f32
  ! CHECK:             %[[VAL_42:.*]] = fir.array_update %[[VAL_33]], %[[VAL_41]], %[[VAL_32]] : (!fir.array<10xf32>, f32, index) -> !fir.array<10xf32>
  ! CHECK:             fir.result %[[VAL_42]] : !fir.array<10xf32>
  ! CHECK:           } else {
  ! CHECK:             fir.result %[[VAL_33]] : !fir.array<10xf32>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_43:.*]] : !fir.array<10xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_25]], %[[VAL_44:.*]] to %[[VAL_2]] : !fir.array<10xf32>, !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>
  ! CHECK:         fir.freemem %[[VAL_9]] : !fir.heap<!fir.array<10x!fir.logical<4>>>
  ! CHECK:         %[[VAL_46:.*]] = arith.constant 10 : index
  ! CHECK:         %[[VAL_47:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_48:.*]] = fir.array_load %[[VAL_0]](%[[VAL_47]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK:         %[[VAL_49:.*]] = arith.constant 1.000000e+02 : f32
  ! CHECK:         %[[VAL_50:.*]] = fir.allocmem !fir.array<10x!fir.logical<4>>
  ! CHECK:         %[[VAL_51:.*]] = fir.shape %[[VAL_46]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_52:.*]] = fir.array_load %[[VAL_50]](%[[VAL_51]]) : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
  ! CHECK:         %[[VAL_53:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_54:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_55:.*]] = arith.subi %[[VAL_46]], %[[VAL_53]] : index
  ! CHECK:         %[[VAL_56:.*]] = fir.do_loop %[[VAL_57:.*]] = %[[VAL_54]] to %[[VAL_55]] step %[[VAL_53]] unordered iter_args(%[[VAL_58:.*]] = %[[VAL_52]]) -> (!fir.array<10x!fir.logical<4>>) {
  ! CHECK:           %[[VAL_59:.*]] = fir.array_fetch %[[VAL_48]], %[[VAL_57]] : (!fir.array<10xf32>, index) -> f32
  ! CHECK:           %[[VAL_60:.*]] = arith.cmpf ogt, %[[VAL_59]], %[[VAL_49]] : f32
  ! CHECK:           %[[VAL_61:.*]] = fir.convert %[[VAL_60]] : (i1) -> !fir.logical<4>
  ! CHECK:           %[[VAL_62:.*]] = fir.array_update %[[VAL_58]], %[[VAL_61]], %[[VAL_57]] : (!fir.array<10x!fir.logical<4>>, !fir.logical<4>, index) -> !fir.array<10x!fir.logical<4>>
  ! CHECK:           fir.result %[[VAL_62]] : !fir.array<10x!fir.logical<4>>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_52]], %[[VAL_63:.*]] to %[[VAL_50]] : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.heap<!fir.array<10x!fir.logical<4>>>
  ! CHECK:         %[[VAL_64:.*]] = fir.shape %[[VAL_46]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_65:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_66:.*]] = fir.array_load %[[VAL_2]](%[[VAL_65]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK:         %[[VAL_67:.*]] = arith.constant 2.000000e+00 : f32
  ! CHECK:         %[[VAL_68:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_69:.*]] = fir.array_load %[[VAL_0]](%[[VAL_68]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK:         %[[VAL_70:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_71:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_72:.*]] = arith.subi %[[VAL_3]], %[[VAL_70]] : index
  ! CHECK:         %[[VAL_73:.*]] = fir.do_loop %[[VAL_74:.*]] = %[[VAL_71]] to %[[VAL_72]] step %[[VAL_70]] unordered iter_args(%[[VAL_75:.*]] = %[[VAL_66]]) -> (!fir.array<10xf32>) {
  ! CHECK:           %[[VAL_76:.*]] = arith.constant 1 : index
  ! CHECK:           %[[VAL_77:.*]] = arith.addi %[[VAL_74]], %[[VAL_76]] : index
  ! CHECK:           %[[VAL_78:.*]] = fir.array_coor %[[VAL_50]](%[[VAL_64]]) %[[VAL_77]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:           %[[VAL_79:.*]] = fir.load %[[VAL_78]] : !fir.ref<!fir.logical<4>>
  ! CHECK:           %[[VAL_80:.*]] = fir.convert %[[VAL_79]] : (!fir.logical<4>) -> i1
  ! CHECK:           %[[VAL_81:.*]] = fir.if %[[VAL_80]] -> (!fir.array<10xf32>) {
  ! CHECK:             %[[VAL_82:.*]] = fir.array_fetch %[[VAL_69]], %[[VAL_74]] : (!fir.array<10xf32>, index) -> f32
  ! CHECK:             %[[VAL_83:.*]] = arith.mulf %[[VAL_67]], %[[VAL_82]] {{.*}}: f32
  ! CHECK:             %[[VAL_84:.*]] = fir.array_update %[[VAL_75]], %[[VAL_83]], %[[VAL_74]] : (!fir.array<10xf32>, f32, index) -> !fir.array<10xf32>
  ! CHECK:             fir.result %[[VAL_84]] : !fir.array<10xf32>
  ! CHECK:           } else {
  ! CHECK:             fir.result %[[VAL_75]] : !fir.array<10xf32>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_85:.*]] : !fir.array<10xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_66]], %[[VAL_86:.*]] to %[[VAL_2]] : !fir.array<10xf32>, !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>
  ! CHECK:         %[[VAL_88:.*]] = arith.constant 10 : index
  ! CHECK:         %[[VAL_89:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_90:.*]] = fir.array_load %[[VAL_0]](%[[VAL_89]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK:         %[[VAL_91:.*]] = arith.constant 5.000000e+01 : f32
  ! CHECK:         %[[VAL_92:.*]] = fir.allocmem !fir.array<10x!fir.logical<4>>
  ! CHECK:         %[[VAL_93:.*]] = fir.shape %[[VAL_88]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_94:.*]] = fir.array_load %[[VAL_92]](%[[VAL_93]]) : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
  ! CHECK:         %[[VAL_95:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_96:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_97:.*]] = arith.subi %[[VAL_88]], %[[VAL_95]] : index
  ! CHECK:         %[[VAL_98:.*]] = fir.do_loop %[[VAL_99:.*]] = %[[VAL_96]] to %[[VAL_97]] step %[[VAL_95]] unordered iter_args(%[[VAL_100:.*]] = %[[VAL_94]]) -> (!fir.array<10x!fir.logical<4>>) {
  ! CHECK:           %[[VAL_101:.*]] = fir.array_fetch %[[VAL_90]], %[[VAL_99]] : (!fir.array<10xf32>, index) -> f32
  ! CHECK:           %[[VAL_102:.*]] = arith.cmpf ogt, %[[VAL_101]], %[[VAL_91]] : f32
  ! CHECK:           %[[VAL_103:.*]] = fir.convert %[[VAL_102]] : (i1) -> !fir.logical<4>
  ! CHECK:           %[[VAL_104:.*]] = fir.array_update %[[VAL_100]], %[[VAL_103]], %[[VAL_99]] : (!fir.array<10x!fir.logical<4>>, !fir.logical<4>, index) -> !fir.array<10x!fir.logical<4>>
  ! CHECK:           fir.result %[[VAL_104]] : !fir.array<10x!fir.logical<4>>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_94]], %[[VAL_105:.*]] to %[[VAL_92]] : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.heap<!fir.array<10x!fir.logical<4>>>
  ! CHECK:         %[[VAL_106:.*]] = fir.shape %[[VAL_88]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_107:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_108:.*]] = fir.array_load %[[VAL_2]](%[[VAL_107]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK:         %[[VAL_109:.*]] = arith.constant 3.000000e+00 : f32
  ! CHECK:         %[[VAL_110:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_111:.*]] = fir.array_load %[[VAL_0]](%[[VAL_110]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK:         %[[VAL_112:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_113:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_114:.*]] = arith.subi %[[VAL_3]], %[[VAL_112]] : index
  ! CHECK:         %[[VAL_115:.*]] = fir.do_loop %[[VAL_116:.*]] = %[[VAL_113]] to %[[VAL_114]] step %[[VAL_112]] unordered iter_args(%[[VAL_117:.*]] = %[[VAL_108]]) -> (!fir.array<10xf32>) {
  ! CHECK:           %[[VAL_118:.*]] = arith.constant 1 : index
  ! CHECK:           %[[VAL_119:.*]] = arith.addi %[[VAL_116]], %[[VAL_118]] : index
  ! CHECK:           %[[VAL_120:.*]] = fir.array_coor %[[VAL_50]](%[[VAL_64]]) %[[VAL_119]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:           %[[VAL_121:.*]] = fir.load %[[VAL_120]] : !fir.ref<!fir.logical<4>>
  ! CHECK:           %[[VAL_122:.*]] = fir.convert %[[VAL_121]] : (!fir.logical<4>) -> i1
  ! CHECK:           %[[VAL_123:.*]] = fir.if %[[VAL_122]] -> (!fir.array<10xf32>) {
  ! CHECK:             fir.result %[[VAL_117]] : !fir.array<10xf32>
  ! CHECK:           } else {
  ! CHECK:             %[[VAL_124:.*]] = arith.constant 1 : index
  ! CHECK:             %[[VAL_125:.*]] = arith.addi %[[VAL_116]], %[[VAL_124]] : index
  ! CHECK:             %[[VAL_126:.*]] = fir.array_coor %[[VAL_92]](%[[VAL_106]]) %[[VAL_125]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:             %[[VAL_127:.*]] = fir.load %[[VAL_126]] : !fir.ref<!fir.logical<4>>
  ! CHECK:             %[[VAL_128:.*]] = fir.convert %[[VAL_127]] : (!fir.logical<4>) -> i1
  ! CHECK:             %[[VAL_129:.*]] = fir.if %[[VAL_128]] -> (!fir.array<10xf32>) {
  ! CHECK:               %[[VAL_130:.*]] = fir.array_fetch %[[VAL_111]], %[[VAL_116]] : (!fir.array<10xf32>, index) -> f32
  ! CHECK:               %[[VAL_131:.*]] = arith.addf %[[VAL_109]], %[[VAL_130]] {{.*}}: f32
  ! CHECK:               %[[VAL_132:.*]] = fir.array_update %[[VAL_117]], %[[VAL_131]], %[[VAL_116]] : (!fir.array<10xf32>, f32, index) -> !fir.array<10xf32>
  ! CHECK:               fir.result %[[VAL_132]] : !fir.array<10xf32>
  ! CHECK:             } else {
  ! CHECK:               fir.result %[[VAL_117]] : !fir.array<10xf32>
  ! CHECK:             }
  ! CHECK:             fir.result %[[VAL_133:.*]] : !fir.array<10xf32>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_134:.*]] : !fir.array<10xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_108]], %[[VAL_135:.*]] to %[[VAL_2]] : !fir.array<10xf32>, !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>
  ! CHECK:         %[[VAL_136:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_137:.*]] = fir.array_load %[[VAL_0]](%[[VAL_136]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK:         %[[VAL_138:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_139:.*]] = fir.array_load %[[VAL_0]](%[[VAL_138]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK:         %[[VAL_140:.*]] = arith.constant 1.000000e+00 : f32
  ! CHECK:         %[[VAL_141:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_142:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_143:.*]] = arith.subi %[[VAL_1]], %[[VAL_141]] : index
  ! CHECK:         %[[VAL_144:.*]] = fir.do_loop %[[VAL_145:.*]] = %[[VAL_142]] to %[[VAL_143]] step %[[VAL_141]] unordered iter_args(%[[VAL_146:.*]] = %[[VAL_137]]) -> (!fir.array<10xf32>) {
  ! CHECK:           %[[VAL_147:.*]] = arith.constant 1 : index
  ! CHECK:           %[[VAL_148:.*]] = arith.addi %[[VAL_145]], %[[VAL_147]] : index
  ! CHECK:           %[[VAL_149:.*]] = fir.array_coor %[[VAL_50]](%[[VAL_64]]) %[[VAL_148]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:           %[[VAL_150:.*]] = fir.load %[[VAL_149]] : !fir.ref<!fir.logical<4>>
  ! CHECK:           %[[VAL_151:.*]] = fir.convert %[[VAL_150]] : (!fir.logical<4>) -> i1
  ! CHECK:           %[[VAL_152:.*]] = fir.if %[[VAL_151]] -> (!fir.array<10xf32>) {
  ! CHECK:             fir.result %[[VAL_146]] : !fir.array<10xf32>
  ! CHECK:           } else {
  ! CHECK:             %[[VAL_153:.*]] = arith.constant 1 : index
  ! CHECK:             %[[VAL_154:.*]] = arith.addi %[[VAL_145]], %[[VAL_153]] : index
  ! CHECK:             %[[VAL_155:.*]] = fir.array_coor %[[VAL_92]](%[[VAL_106]]) %[[VAL_154]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:             %[[VAL_156:.*]] = fir.load %[[VAL_155]] : !fir.ref<!fir.logical<4>>
  ! CHECK:             %[[VAL_157:.*]] = fir.convert %[[VAL_156]] : (!fir.logical<4>) -> i1
  ! CHECK:             %[[VAL_158:.*]] = fir.if %[[VAL_157]] -> (!fir.array<10xf32>) {
  ! CHECK:               %[[VAL_159:.*]] = fir.array_fetch %[[VAL_139]], %[[VAL_145]] : (!fir.array<10xf32>, index) -> f32
  ! CHECK:               %[[VAL_160:.*]] = arith.subf %[[VAL_159]], %[[VAL_140]] {{.*}}: f32
  ! CHECK:               %[[VAL_161:.*]] = fir.array_update %[[VAL_146]], %[[VAL_160]], %[[VAL_145]] : (!fir.array<10xf32>, f32, index) -> !fir.array<10xf32>
  ! CHECK:               fir.result %[[VAL_161]] : !fir.array<10xf32>
  ! CHECK:             } else {
  ! CHECK:               fir.result %[[VAL_146]] : !fir.array<10xf32>
  ! CHECK:             }
  ! CHECK:             fir.result %[[VAL_162:.*]] : !fir.array<10xf32>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_163:.*]] : !fir.array<10xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_137]], %[[VAL_164:.*]] to %[[VAL_0]] : !fir.array<10xf32>, !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>
  ! CHECK:         %[[VAL_165:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_166:.*]] = fir.array_load %[[VAL_0]](%[[VAL_165]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK:         %[[VAL_167:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_168:.*]] = fir.array_load %[[VAL_0]](%[[VAL_167]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK:         %[[VAL_169:.*]] = arith.constant 2.000000e+00 : f32
  ! CHECK:         %[[VAL_170:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_171:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_172:.*]] = arith.subi %[[VAL_1]], %[[VAL_170]] : index
  ! CHECK:         %[[VAL_173:.*]] = fir.do_loop %[[VAL_174:.*]] = %[[VAL_171]] to %[[VAL_172]] step %[[VAL_170]] unordered iter_args(%[[VAL_175:.*]] = %[[VAL_166]]) -> (!fir.array<10xf32>) {
  ! CHECK:           %[[VAL_176:.*]] = arith.constant 1 : index
  ! CHECK:           %[[VAL_177:.*]] = arith.addi %[[VAL_174]], %[[VAL_176]] : index
  ! CHECK:           %[[VAL_178:.*]] = fir.array_coor %[[VAL_50]](%[[VAL_64]]) %[[VAL_177]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:           %[[VAL_179:.*]] = fir.load %[[VAL_178]] : !fir.ref<!fir.logical<4>>
  ! CHECK:           %[[VAL_180:.*]] = fir.convert %[[VAL_179]] : (!fir.logical<4>) -> i1
  ! CHECK:           %[[VAL_181:.*]] = fir.if %[[VAL_180]] -> (!fir.array<10xf32>) {
  ! CHECK:             fir.result %[[VAL_175]] : !fir.array<10xf32>
  ! CHECK:           } else {
  ! CHECK:             %[[VAL_182:.*]] = arith.constant 1 : index
  ! CHECK:             %[[VAL_183:.*]] = arith.addi %[[VAL_174]], %[[VAL_182]] : index
  ! CHECK:             %[[VAL_184:.*]] = fir.array_coor %[[VAL_92]](%[[VAL_106]]) %[[VAL_183]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:             %[[VAL_185:.*]] = fir.load %[[VAL_184]] : !fir.ref<!fir.logical<4>>
  ! CHECK:             %[[VAL_186:.*]] = fir.convert %[[VAL_185]] : (!fir.logical<4>) -> i1
  ! CHECK:             %[[VAL_187:.*]] = fir.if %[[VAL_186]] -> (!fir.array<10xf32>) {
  ! CHECK:               fir.result %[[VAL_175]] : !fir.array<10xf32>
  ! CHECK:             } else {
  ! CHECK:               %[[VAL_188:.*]] = fir.array_fetch %[[VAL_168]], %[[VAL_174]] : (!fir.array<10xf32>, index) -> f32
  ! CHECK:               %[[VAL_189:.*]] = arith.divf %[[VAL_188]], %[[VAL_169]] {{.*}}: f32
  ! CHECK:               %[[VAL_190:.*]] = fir.array_update %[[VAL_175]], %[[VAL_189]], %[[VAL_174]] : (!fir.array<10xf32>, f32, index) -> !fir.array<10xf32>
  ! CHECK:               fir.result %[[VAL_190]] : !fir.array<10xf32>
  ! CHECK:             }
  ! CHECK:             fir.result %[[VAL_191:.*]] : !fir.array<10xf32>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_192:.*]] : !fir.array<10xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_166]], %[[VAL_193:.*]] to %[[VAL_0]] : !fir.array<10xf32>, !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>
  ! CHECK:         fir.freemem %[[VAL_92]] : !fir.heap<!fir.array<10x!fir.logical<4>>>
  ! CHECK:         fir.freemem %[[VAL_50]] : !fir.heap<!fir.array<10x!fir.logical<4>>>
  ! CHECK:         return
  ! CHECK:       }

  real :: a(10), b(10)

  ! Statement
  where (a > 4.0) b = -a

  ! Construct
  where (a > 100.0)
     b = 2.0 * a
  elsewhere (a > 50.0)
     b = 3.0 + a
     a = a - 1.0
  elsewhere
     a = a / 2.0
  end where
end
