! RUN: bbc -o - --outline-intrinsics %s | FileCheck %s

! CHECK-LABEL: func @_QQmain() {
! CHECK-DAG:         %[[VAL_0:.*]] = arith.constant 10 : index
! CHECK-DAG:         %[[VAL_4:.*]] = arith.constant 2 : index
! CHECK-DAG:         %[[VAL_5:.*]] = arith.constant 1 : index
! CHECK-DAG:         %[[VAL_6:.*]] = arith.constant 0 : index
! CHECK-DAG:         %[[VAL_8:.*]] = arith.constant 8 : i64
! CHECK-DAG:         %[[VAL_11:.*]] = arith.constant 3 : index
! CHECK-DAG:         %[[VAL_13:.*]] = arith.constant 2 : i64
! CHECK-DAG:         %[[VAL_14:.*]] = arith.constant 7 : i64
! CHECK-DAG:         %[[VAL_16:.*]] = arith.constant 4 : i64
! CHECK-DAG:         %[[VAL_18:.*]] = arith.constant -1 : i32
! CHECK-DAG:         %[[VAL_19:.*]] = arith.constant 0 : i64
! CHECK-DAG:         %[[VAL_20:.*]] = arith.constant 1 : i64
! CHECK-DAG:         %[[VAL_21:.*]] = arith.constant 3 : i64
! CHECK-DAG:         %[[VAL_22:.*]] = arith.constant 4 : index
! CHECK-DAG:         %[[VAL_23:.*]] = arith.constant 1 : i32
! CHECK-DAG:         %[[VAL_24:.*]] = arith.constant 0 : i32
! CHECK-DAG:         %[[VAL_25:.*]] = fir.address_of(@_QFEa1) : !fir.ref<!fir.array<10x10xf32>>
! CHECK-DAG:         %[[VAL_26:.*]] = fir.address_of(@_QFEa2) : !fir.ref<!fir.array<3xf32>>
! CHECK-DAG:         %[[VAL_27:.*]] = fir.address_of(@_QFEa3) : !fir.ref<!fir.array<10xf32>>
! CHECK-DAG:         %[[VAL_28:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
! CHECK-DAG:         %[[VAL_29:.*]] = fir.address_of(@_QFEiv) : !fir.ref<!fir.array<3xi32>>
! CHECK-DAG:         %[[VAL_30:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFEj"}
! CHECK-DAG:         %[[VAL_31:.*]] = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFEk"}
! CHECK:         fir.store %[[VAL_24]] to %[[VAL_31]] : !fir.ref<i32>
! CHECK:         %[[STEP:.*]] = fir.convert %[[VAL_5]] : (index) -> i32
! CHECK:         br ^bb1(%[[STEP]], %[[VAL_0]] : i32, index)
! CHECK:       ^bb1(%[[VAL_32:.*]]: i32, %[[VAL_33:.*]]: index):
! CHECK:         %[[VAL_34:.*]] = arith.cmpi sgt, %[[VAL_33]], %[[VAL_6]] : index
! CHECK:         cond_br %[[VAL_34]], ^bb2, ^bb6
! CHECK:       ^bb2:
! CHECK:         fir.store %[[VAL_32]] to %[[VAL_30]] : !fir.ref<i32>
! CHECK:         br ^bb3(%[[STEP]], %[[VAL_0]] : i32, index)
! CHECK:       ^bb3(%[[VAL_36:.*]]: i32, %[[VAL_37:.*]]: index):
! CHECK:         %[[VAL_38:.*]] = arith.cmpi sgt, %[[VAL_37]], %[[VAL_6]] : index
! CHECK:         cond_br %[[VAL_38]], ^bb4, ^bb5
! CHECK:       ^bb4:
! CHECK:         fir.store %[[VAL_36]] to %[[VAL_28]] : !fir.ref<i32>
! CHECK:         %[[VAL_40:.*]] = fir.load %[[VAL_31]] : !fir.ref<i32>
! CHECK:         %[[VAL_41:.*]] = arith.addi %[[VAL_40]], %[[VAL_23]] : i32
! CHECK:         fir.store %[[VAL_41]] to %[[VAL_31]] : !fir.ref<i32>
! CHECK:         %[[VAL_42:.*]] = fir.load %[[VAL_31]] : !fir.ref<i32>
! CHECK:         %[[VAL_43:.*]] = fir.convert %[[VAL_42]] : (i32) -> f32
! CHECK:         %[[VAL_44:.*]] = fir.call @fir.cos.f32.f32(%[[VAL_43]]) : (f32) -> f32
! CHECK:         %[[VAL_45:.*]] = fir.load %[[VAL_28]] : !fir.ref<i32>
! CHECK:         %[[VAL_46:.*]] = fir.convert %[[VAL_45]] : (i32) -> i64
! CHECK:         %[[VAL_47:.*]] = arith.subi %[[VAL_46]], %[[VAL_20]] : i64
! CHECK:         %[[VAL_48:.*]] = fir.load %[[VAL_30]] : !fir.ref<i32>
! CHECK:         %[[VAL_49:.*]] = fir.convert %[[VAL_48]] : (i32) -> i64
! CHECK:         %[[VAL_50:.*]] = arith.subi %[[VAL_49]], %[[VAL_20]] : i64
! CHECK:         %[[VAL_51:.*]] = fir.coordinate_of %[[VAL_25]], %[[VAL_47]], %[[VAL_50]] : (!fir.ref<!fir.array<10x10xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK:         fir.store %[[VAL_44]] to %[[VAL_51]] : !fir.ref<f32>
! CHECK:         %[[LOADI:.*]] = fir.load %[[VAL_28]] : !fir.ref<i32>
! CHECK:         %[[VAL_52:.*]] = arith.addi %[[LOADI]], %[[STEP]] : i32
! CHECK:         %[[VAL_53:.*]] = arith.subi %[[VAL_37]], %[[VAL_5]] : index
! CHECK:         br ^bb3(%[[VAL_52]], %[[VAL_53]] : i32, index)
! CHECK:       ^bb5:
! CHECK:         fir.store %[[VAL_36]] to %[[VAL_28]] : !fir.ref<i32>
! CHECK:         %[[VAL_55:.*]] = fir.load %[[VAL_31]] : !fir.ref<i32>
! CHECK:         %[[VAL_56:.*]] = fir.convert %[[VAL_55]] : (i32) -> f32
! CHECK:         %[[VAL_57:.*]] = fir.call @fir.sin.f32.f32(%[[VAL_56]]) : (f32) -> f32
! CHECK:         %[[VAL_58:.*]] = fir.load %[[VAL_30]] : !fir.ref<i32>
! CHECK:         %[[VAL_59:.*]] = fir.convert %[[VAL_58]] : (i32) -> i64
! CHECK:         %[[VAL_60:.*]] = arith.subi %[[VAL_59]], %[[VAL_20]] : i64
! CHECK:         %[[VAL_61:.*]] = fir.coordinate_of %[[VAL_27]], %[[VAL_60]] : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
! CHECK:         fir.store %[[VAL_57]] to %[[VAL_61]] : !fir.ref<f32>
! CHECK:         %[[LOADJ:.*]] = fir.load %[[VAL_30]] : !fir.ref<i32>
! CHECK:         %[[VAL_62:.*]] = arith.addi %[[LOADJ]], %[[STEP]] : i32
! CHECK:         %[[VAL_63:.*]] = arith.subi %[[VAL_33]], %[[VAL_5]] : index
! CHECK:         br ^bb1(%[[VAL_62]], %[[VAL_63]] : i32, index)
! CHECK:       ^bb6:
! CHECK:         fir.store %[[VAL_32]] to %[[VAL_30]] : !fir.ref<i32>
! CHECK:         %[[VAL_65:.*]] = fir.shape %[[VAL_11]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_66:.*]] = fir.undefined index
! CHECK:         %[[VAL_67:.*]] = fir.shape %[[VAL_0]], %[[VAL_0]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_68:.*]] = fir.slice %[[VAL_16]], %[[VAL_66]], %[[VAL_66]], %[[VAL_4]], %[[VAL_0]], %[[VAL_11]] : (i64, index, index, index, index, index) -> !fir.slice<2>
! CHECK:         br ^bb7(%[[VAL_6]], %[[VAL_11]] : index, index)
! CHECK:       ^bb7(%[[VAL_69:.*]]: index, %[[VAL_70:.*]]: index):
! CHECK:         %[[VAL_71:.*]] = arith.cmpi sgt, %[[VAL_70]], %[[VAL_6]] : index
! CHECK:         cond_br %[[VAL_71]], ^bb8, ^bb9
! CHECK:       ^bb8:
! CHECK:         %[[VAL_72:.*]] = arith.addi %[[VAL_69]], %[[VAL_5]] : index
! CHECK:         %[[VAL_73:.*]] = fir.array_coor %[[VAL_25]](%[[VAL_67]]) {{\[}}%[[VAL_68]]] %[[VAL_22]], %[[VAL_72]] : (!fir.ref<!fir.array<10x10xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_74:.*]] = fir.load %[[VAL_73]] : !fir.ref<f32>
! CHECK:         %[[VAL_75:.*]] = fir.array_coor %[[VAL_26]](%[[VAL_65]]) %[[VAL_72]] : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:         fir.store %[[VAL_74]] to %[[VAL_75]] : !fir.ref<f32>
! CHECK:         %[[VAL_76:.*]] = arith.subi %[[VAL_70]], %[[VAL_5]] : index
! CHECK:         br ^bb7(%[[VAL_72]], %[[VAL_76]] : index, index)
! CHECK:       ^bb9:
! CHECK:         %[[VAL_77:.*]] = fir.coordinate_of %[[VAL_25]], %[[VAL_21]], %[[VAL_20]] : (!fir.ref<!fir.array<10x10xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK:         %[[VAL_78:.*]] = fir.load %[[VAL_77]] : !fir.ref<f32>
! CHECK:         %[[VAL_79:.*]] = fir.coordinate_of %[[VAL_26]], %[[VAL_19]] : (!fir.ref<!fir.array<3xf32>>, i64) -> !fir.ref<f32>
! CHECK:         %[[VAL_80:.*]] = fir.load %[[VAL_79]] : !fir.ref<f32>
! CHECK:         %[[VAL_81:.*]] = arith.cmpf une, %[[VAL_78]], %[[VAL_80]] : f32
! CHECK:         cond_br %[[VAL_81]], ^bb10, ^bb11
! CHECK:       ^bb10:
! CHECK:         %[[VAL_82:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,
! CHECK:         %[[VAL_83:.*]] = fir.convert %[[VAL_82]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_84:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_18]], %[[VAL_83]], %{{.*}}) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:         %[[VAL_85:.*]] = fir.address_of(@_QQcl.6D69736D617463682031) : !fir.ref<!fir.char<1,10>>
! CHECK:         %[[VAL_86:.*]] = fir.convert %[[VAL_85]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_87:.*]] = fir.convert %[[VAL_0]] : (index) -> i64
! CHECK:         %[[VAL_88:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_84]], %[[VAL_86]], %[[VAL_87]]) : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:         %[[VAL_89:.*]] = fir.load %[[VAL_79]] : !fir.ref<f32>
! CHECK:         %[[VAL_90:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_84]], %[[VAL_89]]) : (!fir.ref<i8>, f32) -> i1
! CHECK:         %[[VAL_91:.*]] = fir.load %[[VAL_77]] : !fir.ref<f32>
! CHECK:         %[[VAL_92:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_84]], %[[VAL_91]]) : (!fir.ref<i8>, f32) -> i1
! CHECK:         %[[VAL_93:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_84]]) : (!fir.ref<i8>) -> i32
! CHECK:         br ^bb11
! CHECK:       ^bb11:
! CHECK:         %[[VAL_94:.*]] = fir.coordinate_of %[[VAL_25]], %[[VAL_21]], %[[VAL_16]] : (!fir.ref<!fir.array<10x10xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK:         %[[VAL_95:.*]] = fir.load %[[VAL_94]] : !fir.ref<f32>
! CHECK:         %[[VAL_96:.*]] = fir.coordinate_of %[[VAL_26]], %[[VAL_20]] : (!fir.ref<!fir.array<3xf32>>, i64) -> !fir.ref<f32>
! CHECK:         %[[VAL_97:.*]] = fir.load %[[VAL_96]] : !fir.ref<f32>
! CHECK:         %[[VAL_98:.*]] = arith.cmpf une, %[[VAL_95]], %[[VAL_97]] : f32
! CHECK:         cond_br %[[VAL_98]], ^bb12, ^bb13
! CHECK:       ^bb12:
! CHECK:         %[[VAL_99:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,
! CHECK:         %[[VAL_100:.*]] = fir.convert %[[VAL_99]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_101:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_18]], %[[VAL_100]], %{{.*}}) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:         %[[VAL_102:.*]] = fir.address_of(@_QQcl.6D69736D617463682032) : !fir.ref<!fir.char<1,10>>
! CHECK:         %[[VAL_103:.*]] = fir.convert %[[VAL_102]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_104:.*]] = fir.convert %[[VAL_0]] : (index) -> i64
! CHECK:         %[[VAL_105:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_101]], %[[VAL_103]], %[[VAL_104]]) : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:         %[[VAL_106:.*]] = fir.load %[[VAL_96]] : !fir.ref<f32>
! CHECK:         %[[VAL_107:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_101]], %[[VAL_106]]) : (!fir.ref<i8>, f32) -> i1
! CHECK:         %[[VAL_108:.*]] = fir.load %[[VAL_94]] : !fir.ref<f32>
! CHECK:         %[[VAL_109:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_101]], %[[VAL_108]]) : (!fir.ref<i8>, f32) -> i1
! CHECK:         %[[VAL_110:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_101]]) : (!fir.ref<i8>) -> i32
! CHECK:         br ^bb13
! CHECK:       ^bb13:
! CHECK:         %[[VAL_111:.*]] = fir.coordinate_of %[[VAL_25]], %[[VAL_21]], %[[VAL_14]] : (!fir.ref<!fir.array<10x10xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK:         %[[VAL_112:.*]] = fir.load %[[VAL_111]] : !fir.ref<f32>
! CHECK:         %[[VAL_113:.*]] = fir.coordinate_of %[[VAL_26]], %[[VAL_13]] : (!fir.ref<!fir.array<3xf32>>, i64) -> !fir.ref<f32>
! CHECK:         %[[VAL_114:.*]] = fir.load %[[VAL_113]] : !fir.ref<f32>
! CHECK:         %[[VAL_115:.*]] = arith.cmpf une, %[[VAL_112]], %[[VAL_114]] : f32
! CHECK:         cond_br %[[VAL_115]], ^bb14, ^bb15
! CHECK:       ^bb14:
! CHECK:         %[[VAL_116:.*]] = fir.address_of(@_QQcl.{{.*}} : !fir.ref<!fir.char<1,
! CHECK:         %[[VAL_117:.*]] = fir.convert %[[VAL_116]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_118:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_18]], %[[VAL_117]], %{{.*}}) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:         %[[VAL_119:.*]] = fir.address_of(@_QQcl.6D69736D617463682033) : !fir.ref<!fir.char<1,10>>
! CHECK:         %[[VAL_120:.*]] = fir.convert %[[VAL_119]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_121:.*]] = fir.convert %[[VAL_0]] : (index) -> i64
! CHECK:         %[[VAL_122:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_118]], %[[VAL_120]], %[[VAL_121]]) : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:         %[[VAL_123:.*]] = fir.load %[[VAL_113]] : !fir.ref<f32>
! CHECK:         %[[VAL_124:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_118]], %[[VAL_123]]) : (!fir.ref<i8>, f32) -> i1
! CHECK:         %[[VAL_125:.*]] = fir.load %[[VAL_111]] : !fir.ref<f32>
! CHECK:         %[[VAL_126:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_118]], %[[VAL_125]]) : (!fir.ref<i8>, f32) -> i1
! CHECK:         %[[VAL_127:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_118]]) : (!fir.ref<i8>) -> i32
! CHECK:         br ^bb15
! CHECK:       ^bb15:
! CHECK:         %[[VAL_128:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_129:.*]] = fir.slice %[[VAL_5]], %[[VAL_0]], %[[VAL_22]] : (index, index, index) -> !fir.slice<1>
! CHECK:         br ^bb16(%[[VAL_6]], %[[VAL_11]] : index, index)
! CHECK:       ^bb16(%[[VAL_130:.*]]: index, %[[VAL_131:.*]]: index):
! CHECK:         %[[VAL_132:.*]] = arith.cmpi sgt, %[[VAL_131]], %[[VAL_6]] : index
! CHECK:         cond_br %[[VAL_132]], ^bb17, ^bb18
! CHECK:       ^bb17:
! CHECK:         %[[VAL_133:.*]] = arith.addi %[[VAL_130]], %[[VAL_5]] : index
! CHECK:         %[[VAL_134:.*]] = fir.array_coor %[[VAL_26]](%[[VAL_65]]) %[[VAL_133]] : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_135:.*]] = fir.load %[[VAL_134]] : !fir.ref<f32>
! CHECK:         %[[VAL_136:.*]] = fir.array_coor %[[VAL_27]](%[[VAL_128]]) {{\[}}%[[VAL_129]]] %[[VAL_133]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
! CHECK:         fir.store %[[VAL_135]] to %[[VAL_136]] : !fir.ref<f32>
! CHECK:         %[[VAL_137:.*]] = arith.subi %[[VAL_131]], %[[VAL_5]] : index
! CHECK:         br ^bb16(%[[VAL_133]], %[[VAL_137]] : index, index)
! CHECK:       ^bb18:
! CHECK:         %[[VAL_138:.*]] = fir.load %[[VAL_77]] : !fir.ref<f32>
! CHECK:         %[[VAL_139:.*]] = fir.coordinate_of %[[VAL_27]], %[[VAL_19]] : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
! CHECK:         %[[VAL_140:.*]] = fir.load %[[VAL_139]] : !fir.ref<f32>
! CHECK:         %[[VAL_141:.*]] = arith.cmpf une, %[[VAL_138]], %[[VAL_140]] : f32
! CHECK:         cond_br %[[VAL_141]], ^bb19, ^bb20
! CHECK:       ^bb19:
! CHECK:         %[[VAL_142:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,
! CHECK:         %[[VAL_143:.*]] = fir.convert %[[VAL_142]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_144:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_18]], %[[VAL_143]], %{{.*}}) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:         %[[VAL_145:.*]] = fir.address_of(@_QQcl.6D69736D617463682034) : !fir.ref<!fir.char<1,10>>
! CHECK:         %[[VAL_146:.*]] = fir.convert %[[VAL_145]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_147:.*]] = fir.convert %[[VAL_0]] : (index) -> i64
! CHECK:         %[[VAL_148:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_144]], %[[VAL_146]], %[[VAL_147]]) : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:         %[[VAL_149:.*]] = fir.load %[[VAL_77]] : !fir.ref<f32>
! CHECK:         %[[VAL_150:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_144]], %[[VAL_149]]) : (!fir.ref<i8>, f32) -> i1
! CHECK:         %[[VAL_151:.*]] = fir.load %[[VAL_139]] : !fir.ref<f32>
! CHECK:         %[[VAL_152:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_144]], %[[VAL_151]]) : (!fir.ref<i8>, f32) -> i1
! CHECK:         %[[VAL_153:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_144]]) : (!fir.ref<i8>) -> i32
! CHECK:         br ^bb20
! CHECK:       ^bb20:
! CHECK:         %[[VAL_154:.*]] = fir.load %[[VAL_94]] : !fir.ref<f32>
! CHECK:         %[[VAL_155:.*]] = fir.coordinate_of %[[VAL_27]], %[[VAL_16]] : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
! CHECK:         %[[VAL_156:.*]] = fir.load %[[VAL_155]] : !fir.ref<f32>
! CHECK:         %[[VAL_157:.*]] = arith.cmpf une, %[[VAL_154]], %[[VAL_156]] : f32
! CHECK:         cond_br %[[VAL_157]], ^bb21, ^bb22
! CHECK:       ^bb21:
! CHECK:         %[[VAL_158:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,
! CHECK:         %[[VAL_159:.*]] = fir.convert %[[VAL_158]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_160:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_18]], %[[VAL_159]], %{{.*}}) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:         %[[VAL_161:.*]] = fir.address_of(@_QQcl.6D69736D617463682035) : !fir.ref<!fir.char<1,10>>
! CHECK:         %[[VAL_162:.*]] = fir.convert %[[VAL_161]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_163:.*]] = fir.convert %[[VAL_0]] : (index) -> i64
! CHECK:         %[[VAL_164:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_160]], %[[VAL_162]], %[[VAL_163]]) : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:         %[[VAL_165:.*]] = fir.load %[[VAL_94]] : !fir.ref<f32>
! CHECK:         %[[VAL_166:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_160]], %[[VAL_165]]) : (!fir.ref<i8>, f32) -> i1
! CHECK:         %[[VAL_167:.*]] = fir.load %[[VAL_155]] : !fir.ref<f32>
! CHECK:         %[[VAL_168:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_160]], %[[VAL_167]]) : (!fir.ref<i8>, f32) -> i1
! CHECK:         %[[VAL_169:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_160]]) : (!fir.ref<i8>) -> i32
! CHECK:         br ^bb22
! CHECK:       ^bb22:
! CHECK:         %[[VAL_170:.*]] = fir.load %[[VAL_111]] : !fir.ref<f32>
! CHECK:         %[[VAL_171:.*]] = fir.coordinate_of %[[VAL_27]], %[[VAL_8]] : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
! CHECK:         %[[VAL_172:.*]] = fir.load %[[VAL_171]] : !fir.ref<f32>
! CHECK:         %[[VAL_173:.*]] = arith.cmpf une, %[[VAL_170]], %[[VAL_172]] : f32
! CHECK:         cond_br %[[VAL_173]], ^bb23, ^bb24
! CHECK:       ^bb23:
! CHECK:         %[[VAL_174:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,
! CHECK:         %[[VAL_175:.*]] = fir.convert %[[VAL_174]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_176:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_18]], %[[VAL_175]], %{{.*}}) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:         %[[VAL_177:.*]] = fir.address_of(@_QQcl.6D69736D617463682036) : !fir.ref<!fir.char<1,10>>
! CHECK:         %[[VAL_178:.*]] = fir.convert %[[VAL_177]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_179:.*]] = fir.convert %[[VAL_0]] : (index) -> i64
! CHECK:         %[[VAL_180:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_176]], %[[VAL_178]], %[[VAL_179]]) : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:         %[[VAL_181:.*]] = fir.load %[[VAL_111]] : !fir.ref<f32>
! CHECK:         %[[VAL_182:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_176]], %[[VAL_181]]) : (!fir.ref<i8>, f32) -> i1
! CHECK:         %[[VAL_183:.*]] = fir.load %[[VAL_171]] : !fir.ref<f32>
! CHECK:         %[[VAL_184:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_176]], %[[VAL_183]]) : (!fir.ref<i8>, f32) -> i1
! CHECK:         %[[VAL_185:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_176]]) : (!fir.ref<i8>) -> i32
! CHECK:         br ^bb24
! CHECK:       ^bb24:
! CHECK:         %[[VAL_186:.*]] = fir.address_of(@_QQro.3xi4.b7f1b733471804c07debf489e49d9c2f) : !fir.ref<!fir.array<3xi32>>
! CHECK:         br ^bb25(%[[VAL_6]], %[[VAL_11]] : index, index)
! CHECK:       ^bb25(%[[VAL_187:.*]]: index, %[[VAL_188:.*]]: index):
! CHECK:         %[[VAL_189:.*]] = arith.cmpi sgt, %[[VAL_188]], %[[VAL_6]] : index
! CHECK:         cond_br %[[VAL_189]], ^bb26, ^bb27
! CHECK:       ^bb26:
! CHECK:         %[[VAL_190:.*]] = arith.addi %[[VAL_187]], %[[VAL_5]] : index
! CHECK:         %[[VAL_191:.*]] = fir.array_coor %[[VAL_186]](%[[VAL_65]]) %[[VAL_190]] : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:         %[[VAL_192:.*]] = fir.load %[[VAL_191]] : !fir.ref<i32>
! CHECK:         %[[VAL_193:.*]] = fir.array_coor %[[VAL_29]](%[[VAL_65]]) %[[VAL_190]] : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:         fir.store %[[VAL_192]] to %[[VAL_193]] : !fir.ref<i32>
! CHECK:         %[[VAL_194:.*]] = arith.subi %[[VAL_188]], %[[VAL_5]] : index
! CHECK:         br ^bb25(%[[VAL_190]], %[[VAL_194]] : index, index)
! CHECK:       ^bb27:
! CHECK:         %[[VAL_195:.*]] = fir.allocmem !fir.array<3xf32>
! CHECK:         br ^bb28(%[[VAL_6]], %[[VAL_11]] : index, index)
! CHECK:       ^bb28(%[[VAL_196:.*]]: index, %[[VAL_197:.*]]: index):
! CHECK:         %[[VAL_198:.*]] = arith.cmpi sgt, %[[VAL_197]], %[[VAL_6]] : index
! CHECK:         cond_br %[[VAL_198]], ^bb29, ^bb30
! CHECK:       ^bb29:
! CHECK:         %[[VAL_199:.*]] = arith.addi %[[VAL_196]], %[[VAL_5]] : index
! CHECK:         %[[VAL_200:.*]] = fir.array_coor %[[VAL_26]](%[[VAL_65]]) %[[VAL_199]] : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_201:.*]] = fir.array_coor %[[VAL_195]](%[[VAL_65]]) %[[VAL_199]] : (!fir.heap<!fir.array<3xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_202:.*]] = fir.load %[[VAL_200]] : !fir.ref<f32>
! CHECK:         fir.store %[[VAL_202]] to %[[VAL_201]] : !fir.ref<f32>
! CHECK:         %[[VAL_203:.*]] = arith.subi %[[VAL_197]], %[[VAL_5]] : index
! CHECK:         br ^bb28(%[[VAL_199]], %[[VAL_203]] : index, index)
! CHECK:       ^bb30(%[[VAL_205:.*]]: index, %[[VAL_206:.*]]: index):
! CHECK:         %[[VAL_207:.*]] = arith.cmpi sgt, %[[VAL_206]], %[[VAL_6]] : index
! CHECK:         cond_br %[[VAL_207]], ^bb31, ^bb32(%[[VAL_6]], %[[VAL_11]] : index, index)
! CHECK:       ^bb31:
! CHECK:         %[[VAL_208:.*]] = arith.addi %[[VAL_205]], %[[VAL_5]] : index
! CHECK:         %[[VAL_209:.*]] = fir.array_coor %[[VAL_29]](%[[VAL_65]]) %[[VAL_208]] : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:         %[[VAL_210:.*]] = fir.load %[[VAL_209]] : !fir.ref<i32>
! CHECK:         %[[VAL_211:.*]] = fir.convert %[[VAL_210]] : (i32) -> index
! CHECK:         %[[VAL_212:.*]] = fir.array_coor %[[VAL_26]](%[[VAL_65]]) %[[VAL_211]] : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_213:.*]] = fir.load %[[VAL_212]] : !fir.ref<f32>
! CHECK:         %[[VAL_214:.*]] = fir.array_coor %[[VAL_195]](%[[VAL_65]]) %[[VAL_208]] : (!fir.heap<!fir.array<3xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:         fir.store %[[VAL_213]] to %[[VAL_214]] : !fir.ref<f32>
! CHECK:         %[[VAL_215:.*]] = arith.subi %[[VAL_206]], %[[VAL_5]] : index
! CHECK:         br ^bb30(%[[VAL_208]], %[[VAL_215]] : index, index)
! CHECK:       ^bb32(%[[VAL_216:.*]]: index, %[[VAL_217:.*]]: index):
! CHECK:         %[[VAL_218:.*]] = arith.cmpi sgt, %[[VAL_217]], %[[VAL_6]] : index
! CHECK:         cond_br %[[VAL_218]], ^bb33, ^bb34
! CHECK:       ^bb33:
! CHECK:         %[[VAL_219:.*]] = arith.addi %[[VAL_216]], %[[VAL_5]] : index
! CHECK:         %[[VAL_220:.*]] = fir.array_coor %[[VAL_195]](%[[VAL_65]]) %[[VAL_219]] : (!fir.heap<!fir.array<3xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_221:.*]] = fir.array_coor %[[VAL_26]](%[[VAL_65]]) %[[VAL_219]] : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_222:.*]] = fir.load %[[VAL_220]] : !fir.ref<f32>
! CHECK:         fir.store %[[VAL_222]] to %[[VAL_221]] : !fir.ref<f32>
! CHECK:         %[[VAL_223:.*]] = arith.subi %[[VAL_217]], %[[VAL_5]] : index
! CHECK:         br ^bb32(%[[VAL_219]], %[[VAL_223]] : index, index)
! CHECK:       ^bb34:
! CHECK:         fir.freemem %[[VAL_195]] : !fir.heap<!fir.array<3xf32>>
! CHECK:         %[[VAL_224:.*]] = fir.load %[[VAL_77]] : !fir.ref<f32>
! CHECK:         %[[VAL_225:.*]] = fir.load %[[VAL_96]] : !fir.ref<f32>
! CHECK:         %[[VAL_226:.*]] = arith.cmpf une, %[[VAL_224]], %[[VAL_225]] : f32
! CHECK:         cond_br %[[VAL_226]], ^bb35, ^bb36
! CHECK:       ^bb35:
! CHECK:         %[[VAL_227:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,
! CHECK:         %[[VAL_228:.*]] = fir.convert %[[VAL_227]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_229:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_18]], %[[VAL_228]], %{{.*}}) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:         %[[VAL_230:.*]] = fir.address_of(@_QQcl.6D69736D617463682037) : !fir.ref<!fir.char<1,10>>
! CHECK:         %[[VAL_231:.*]] = fir.convert %[[VAL_230]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_232:.*]] = fir.convert %[[VAL_0]] : (index) -> i64
! CHECK:         %[[VAL_233:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_229]], %[[VAL_231]], %[[VAL_232]]) : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:         %[[VAL_234:.*]] = fir.load %[[VAL_77]] : !fir.ref<f32>
! CHECK:         %[[VAL_235:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_229]], %[[VAL_234]]) : (!fir.ref<i8>, f32) -> i1
! CHECK:         %[[VAL_236:.*]] = fir.load %[[VAL_96]] : !fir.ref<f32>
! CHECK:         %[[VAL_237:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_229]], %[[VAL_236]]) : (!fir.ref<i8>, f32) -> i1
! CHECK:         %[[VAL_238:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_229]]) : (!fir.ref<i8>) -> i32
! CHECK:         br ^bb36
! CHECK:       ^bb36:
! CHECK:         %[[VAL_239:.*]] = fir.load %[[VAL_94]] : !fir.ref<f32>
! CHECK:         %[[VAL_240:.*]] = fir.load %[[VAL_113]] : !fir.ref<f32>
! CHECK:         %[[VAL_241:.*]] = arith.cmpf une, %[[VAL_239]], %[[VAL_240]] : f32
! CHECK:         cond_br %[[VAL_241]], ^bb37, ^bb38
! CHECK:       ^bb37:
! CHECK:         %[[VAL_242:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,
! CHECK:         %[[VAL_243:.*]] = fir.convert %[[VAL_242]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_244:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_18]], %[[VAL_243]], %{{.*}}) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:         %[[VAL_245:.*]] = fir.address_of(@_QQcl.6D69736D617463682038) : !fir.ref<!fir.char<1,10>>
! CHECK:         %[[VAL_246:.*]] = fir.convert %[[VAL_245]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_247:.*]] = fir.convert %[[VAL_0]] : (index) -> i64
! CHECK:         %[[VAL_248:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_244]], %[[VAL_246]], %[[VAL_247]]) : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:         %[[VAL_249:.*]] = fir.load %[[VAL_94]] : !fir.ref<f32>
! CHECK:         %[[VAL_250:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_244]], %[[VAL_249]]) : (!fir.ref<i8>, f32) -> i1
! CHECK:         %[[VAL_251:.*]] = fir.load %[[VAL_113]] : !fir.ref<f32>
! CHECK:         %[[VAL_252:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_244]], %[[VAL_251]]) : (!fir.ref<i8>, f32) -> i1
! CHECK:         %[[VAL_253:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_244]]) : (!fir.ref<i8>) -> i32
! CHECK:         br ^bb38
! CHECK:       ^bb38:
! CHECK:         %[[VAL_254:.*]] = fir.load %[[VAL_111]] : !fir.ref<f32>
! CHECK:         %[[VAL_255:.*]] = fir.load %[[VAL_79]] : !fir.ref<f32>
! CHECK:         %[[VAL_256:.*]] = arith.cmpf une, %[[VAL_254]], %[[VAL_255]] : f32
! CHECK:         cond_br %[[VAL_256]], ^bb39, ^bb40
! CHECK:       ^bb39:
! CHECK:         %[[VAL_257:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,
! CHECK:         %[[VAL_258:.*]] = fir.convert %[[VAL_257]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_259:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_18]], %[[VAL_258]], %{{.*}}) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:         %[[VAL_260:.*]] = fir.address_of(@_QQcl.6D69736D617463682039) : !fir.ref<!fir.char<1,10>>
! CHECK:         %[[VAL_261:.*]] = fir.convert %[[VAL_260]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_262:.*]] = fir.convert %[[VAL_0]] : (index) -> i64
! CHECK:         %[[VAL_263:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_259]], %[[VAL_261]], %[[VAL_262]]) : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:         %[[VAL_264:.*]] = fir.load %[[VAL_111]] : !fir.ref<f32>
! CHECK:         %[[VAL_265:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_259]], %[[VAL_264]]) : (!fir.ref<i8>, f32) -> i1
! CHECK:         %[[VAL_266:.*]] = fir.load %[[VAL_79]] : !fir.ref<f32>
! CHECK:         %[[VAL_267:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_259]], %[[VAL_266]]) : (!fir.ref<i8>, f32) -> i1
! CHECK:         %[[VAL_268:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_259]]) : (!fir.ref<i8>) -> i32
! CHECK:         br ^bb40
! CHECK:       ^bb40:
! CHECK:         return
! CHECK:       }

program p
  real :: a1(10,10)
  real :: a2(3)
  real :: a3(10)
  integer iv(3)
  integer k

  k = 0
  do j = 1, 10
     do i = 1, 10
        k = k + 1
        a1(i,j) = cos(real(k))
     end do
     a3(j) = sin(real(k))
  end do

  a2 = a1(4, 2:10:3)

  if (a1(4,2) .ne. a2(1)) print *, "mismatch 1", a2(1), a1(4,2)
  if (a1(4,5) .ne. a2(2)) print *, "mismatch 2", a2(2), a1(4,5)
  if (a1(4,8) .ne. a2(3)) print *, "mismatch 3", a2(3), a1(4,8)

  a3(1:10:4) = a2

  if (a1(4,2) .ne. a3(1)) print *, "mismatch 4", a1(4,2), a3(1)
  if (a1(4,5) .ne. a3(5)) print *, "mismatch 5", a1(4,5), a3(5)
  if (a1(4,8) .ne. a3(9)) print *, "mismatch 6", a1(4,8), a3(9)

  iv = (/ 3, 1, 2 /)

  a2 = a2(iv)

  if (a1(4,2) .ne. a2(2)) print *, "mismatch 7", a1(4,2), a2(2)
  if (a1(4,5) .ne. a2(3)) print *, "mismatch 8", a1(4,5), a2(3)
  if (a1(4,8) .ne. a2(1)) print *, "mismatch 9", a1(4,8), a2(1)

end program p

! CHECK-LABEL: func @_QPsub(
! CHECK-SAME:               %[[VAL_0:.*]]: !fir.boxchar<1>{{.*}}) {
! CHECK-DAG:     %[[VAL_1:.*]] = arith.constant 5 : index
! CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 2 : index
! CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 1 : index
! CHECK-DAG:     %[[VAL_4:.*]] = arith.constant 4 : index
! CHECK-DAG:     %[[VAL_6:.*]] = arith.constant -1 : i32
! CHECK-DAG:     %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_8:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1>>>
! CHECK:         %[[VAL_10:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_12:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_6]], %[[VAL_11]], %{{.*}}) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:         %[[VAL_13:.*]] = fir.address_of(@_QQcl.61203D20) : !fir.ref<!fir.char<1,4>>
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_4]] : (index) -> i64
! CHECK:         %[[VAL_16:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_12]], %[[VAL_14]], %[[VAL_15]]) : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:         %[[VAL_17:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_18:.*]] = fir.slice %[[VAL_3]], %[[VAL_1]], %[[VAL_2]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_19:.*]] = fir.embox %[[VAL_9]](%[[VAL_17]]) {{\[}}%[[VAL_18]]] : (!fir.ref<!fir.array<10x!fir.char<1>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<3x!fir.char<1>>>
! CHECK:         %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (!fir.box<!fir.array<3x!fir.char<1>>>) -> !fir.box<none>
! CHECK:         %[[VAL_21:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_12]], %[[VAL_20]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:         %[[VAL_22:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_12]]) : (!fir.ref<i8>) -> i32
! CHECK:         return
! CHECK:       }

! Slice operation on array of CHARACTER
subroutine sub(a)
  character :: a(10)
  print *, "a = ", a(1:5:2)
end subroutine sub
