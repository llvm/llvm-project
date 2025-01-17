! Test lowering of IO input items with vector subscripts
! RUN: bbc -hlfir=false %s -o - | FileCheck %s
! UNSUPPORTED: system-windows

! CHECK-LABEL: func @_QPsimple(
! CHECK-SAME: %[[VAL_20:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_16:.*]]: !fir.ref<!fir.array<3xi32>>{{.*}}) {
subroutine simple(x, y)
  integer :: y(3)
  integer :: x(10)
  read(*,*) x(y)
! CHECK-DAG: %[[VAL_0:.*]] = arith.constant 10 : index
! CHECK-DAG: %[[VAL_1:.*]] = arith.constant 5 : i32
! CHECK-DAG: %[[VAL_3:.*]] = arith.constant 4 : i32
! CHECK-DAG: %[[VAL_4:.*]] = arith.constant 3 : index
! CHECK-DAG: %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK-DAG: %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_7:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:   %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_9:.*]] = fir.call @_FortranAioBeginExternalListInput(%[[VAL_1]], %[[VAL_8]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_10:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
! CHECK:   %[[VAL_11:.*]] = fir.slice %[[VAL_6]], %[[VAL_4]], %[[VAL_6]] : (index, index, index) -> !fir.slice<1>
! CHECK:   cf.br ^bb1(%[[VAL_5]], %[[VAL_4]] : index, index)
! CHECK: ^bb1(%[[VAL_12:.*]]: index, %[[VAL_13:.*]]: index):
! CHECK:   %[[VAL_14:.*]] = arith.cmpi sgt, %[[VAL_13]], %[[VAL_5]] : index
! CHECK:   cf.cond_br %[[VAL_14]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_15:.*]] = fir.coordinate_of %[[VAL_16]], %[[VAL_12]] : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_17:.*]] = fir.load %[[VAL_15]] : !fir.ref<i32>
! CHECK:   %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> index
! CHECK:   %[[VAL_19:.*]] = fir.array_coor %[[VAL_20]](%[[VAL_10]]) {{\[}}%[[VAL_11]]] %[[VAL_18]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_21:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<i32>) -> !fir.ref<i64>
! CHECK:   %[[VAL_22:.*]] = fir.call @_FortranAioInputInteger(%[[VAL_9]], %[[VAL_21]], %[[VAL_3]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i64>, i32) -> i1
! CHECK:   %[[VAL_23:.*]] = arith.addi %[[VAL_12]], %[[VAL_6]] overflow<nsw> : index
! CHECK:   %[[VAL_24:.*]] = arith.subi %[[VAL_13]], %[[VAL_6]] : index
! CHECK:   cf.br ^bb1(%[[VAL_23]], %[[VAL_24]] : index, index)
! CHECK: ^bb3:
! CHECK:   %[[VAL_25:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_9]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPonly_once(
! CHECK-SAME: %[[VAL_51:.*]]: !fir.box<!fir.array<?x?xf32>>{{.*}}) {
subroutine only_once(x)
  interface
    function get_vector()
      integer, allocatable :: get_vector(:)
    end function
    integer function get_substcript()
    end function
  end interface
  real :: x(:, :)
  ! Test subscripts are only evaluated once.
  read(*,*) x(get_substcript(), get_vector())
! CHECK-DAG: %[[VAL_26:.*]] = arith.constant 5 : i32
! CHECK-DAG: %[[VAL_28:.*]] = arith.constant 0 : i64
! CHECK-DAG: %[[VAL_29:.*]] = arith.constant 0 : index
! CHECK-DAG: %[[VAL_30:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_31:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = ".result"}
! CHECK:   %[[VAL_32:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:   %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_34:.*]] = fir.call @_FortranAioBeginExternalListInput(%[[VAL_26]], %[[VAL_33]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_35:.*]] = fir.call @_QPget_substcript() {{.*}}: () -> i32
! CHECK:   %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i32) -> i64
! CHECK:   %[[VAL_37:.*]] = fir.call @_QPget_vector() {{.*}}: () -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:   fir.save_result %[[VAL_37]] to %[[VAL_31]] : !fir.box<!fir.heap<!fir.array<?xi32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:   %[[VAL_38:.*]] = fir.load %[[VAL_31]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:   %[[VAL_39:.*]]:3 = fir.box_dims %[[VAL_38]], %[[VAL_29]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:   %[[VAL_40:.*]] = fir.box_addr %[[VAL_38]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:   %[[VAL_41:.*]] = fir.undefined index
! CHECK:   %[[VAL_42:.*]] = fir.slice %[[VAL_36]], %[[VAL_41]], %[[VAL_41]], %[[VAL_30]], %[[VAL_39]]#1, %[[VAL_30]] : (i64, index, index, index, index, index) -> !fir.slice<2>
! CHECK:   cf.br ^bb1(%[[VAL_29]], %[[VAL_39]]#1 : index, index)
! CHECK: ^bb1(%[[VAL_43:.*]]: index, %[[VAL_44:.*]]: index):
! CHECK:   %[[VAL_45:.*]] = arith.cmpi sgt, %[[VAL_44]], %[[VAL_29]] : index
! CHECK:   cf.cond_br %[[VAL_45]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_46:.*]] = fir.convert %[[VAL_35]] : (i32) -> index
! CHECK:   %[[VAL_47:.*]] = fir.coordinate_of %[[VAL_40]], %[[VAL_43]] : (!fir.heap<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_48:.*]] = fir.load %[[VAL_47]] : !fir.ref<i32>
! CHECK:   %[[VAL_49:.*]] = fir.convert %[[VAL_48]] : (i32) -> index
! CHECK:   %[[VAL_50:.*]] = fir.array_coor %[[VAL_51]] {{\[}}%[[VAL_42]]] %[[VAL_46]], %[[VAL_49]] : (!fir.box<!fir.array<?x?xf32>>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:   %[[VAL_52:.*]] = fir.call @_FortranAioInputReal32(%[[VAL_34]], %[[VAL_50]]) {{.*}}: (!fir.ref<i8>, !fir.ref<f32>) -> i1
! CHECK:   %[[VAL_53:.*]] = arith.addi %[[VAL_43]], %[[VAL_30]] overflow<nsw> : index
! CHECK:   %[[VAL_54:.*]] = arith.subi %[[VAL_44]], %[[VAL_30]] : index
! CHECK:   cf.br ^bb1(%[[VAL_53]], %[[VAL_54]] : index, index)
! CHECK: ^bb3:
! CHECK:   %[[VAL_55:.*]] = fir.load %[[VAL_31]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:   %[[VAL_56:.*]] = fir.box_addr %[[VAL_55]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:   %[[VAL_57:.*]] = fir.convert %[[VAL_56]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK:   %[[VAL_58:.*]] = arith.cmpi ne, %[[VAL_57]], %[[VAL_28]] : i64
! CHECK:   cf.cond_br %[[VAL_58]], ^bb4, ^bb5
! CHECK: ^bb4:
! CHECK:   fir.freemem %[[VAL_56]] : !fir.heap<!fir.array<?xi32>>
! CHECK:   cf.br ^bb5
! CHECK: ^bb5:
! CHECK:   %[[VAL_59:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_34]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPwith_assumed_shapes(
! CHECK-SAME: %[[VAL_78:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[VAL_69:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) {
subroutine with_assumed_shapes(x, y)
  integer :: y(:)
  integer :: x(:)
  read(*,*) x(y)
! CHECK-DAG: %[[VAL_60:.*]] = arith.constant 5 : i32
! CHECK-DAG: %[[VAL_62:.*]] = arith.constant 4 : i32
! CHECK-DAG: %[[VAL_63:.*]] = arith.constant 0 : index
! CHECK-DAG: %[[VAL_64:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_65:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:   %[[VAL_66:.*]] = fir.convert %[[VAL_65]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_67:.*]] = fir.call @_FortranAioBeginExternalListInput(%[[VAL_60]], %[[VAL_66]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_68:.*]]:3 = fir.box_dims %[[VAL_69]], %[[VAL_63]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:   %[[VAL_70:.*]] = fir.slice %[[VAL_64]], %[[VAL_68]]#1, %[[VAL_64]] : (index, index, index) -> !fir.slice<1>
! CHECK:   cf.br ^bb1(%[[VAL_63]], %[[VAL_68]]#1 : index, index)
! CHECK: ^bb1(%[[VAL_71:.*]]: index, %[[VAL_72:.*]]: index):
! CHECK:   %[[VAL_73:.*]] = arith.cmpi sgt, %[[VAL_72]], %[[VAL_63]] : index
! CHECK:   cf.cond_br %[[VAL_73]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_74:.*]] = fir.coordinate_of %[[VAL_69]], %[[VAL_71]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_75:.*]] = fir.load %[[VAL_74]] : !fir.ref<i32>
! CHECK:   %[[VAL_76:.*]] = fir.convert %[[VAL_75]] : (i32) -> index
! CHECK:   %[[VAL_77:.*]] = fir.array_coor %[[VAL_78]] {{\[}}%[[VAL_70]]] %[[VAL_76]] : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_79:.*]] = fir.convert %[[VAL_77]] : (!fir.ref<i32>) -> !fir.ref<i64>
! CHECK:   %[[VAL_80:.*]] = fir.call @_FortranAioInputInteger(%[[VAL_67]], %[[VAL_79]], %[[VAL_62]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i64>, i32) -> i1
! CHECK:   %[[VAL_81:.*]] = arith.addi %[[VAL_71]], %[[VAL_64]] overflow<nsw> : index
! CHECK:   %[[VAL_82:.*]] = arith.subi %[[VAL_72]], %[[VAL_64]] : index
! CHECK:   cf.br ^bb1(%[[VAL_81]], %[[VAL_82]] : index, index)
! CHECK: ^bb3:
! CHECK:   %[[VAL_83:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_67]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPlower_bounds(
! CHECK-SAME: %[[VAL_108:.*]]: !fir.ref<!fir.array<4x6xi32>>{{.*}}, %[[VAL_104:.*]]: !fir.ref<!fir.array<3xi32>>{{.*}}) {
subroutine lower_bounds(x, y)
  integer :: y(3)
  integer :: x(2:5,3:8)
  read(*,*) x(3, y)
! CHECK-DAG: %[[VAL_84:.*]] = arith.constant 4 : index
! CHECK-DAG: %[[VAL_85:.*]] = arith.constant 6 : index
! CHECK-DAG: %[[VAL_86:.*]] = arith.constant 5 : i32
! CHECK-DAG: %[[VAL_88:.*]] = arith.constant 3 : i64
! CHECK-DAG: %[[VAL_89:.*]] = arith.constant 2 : index
! CHECK-DAG: %[[VAL_90:.*]] = arith.constant 4 : i32
! CHECK-DAG: %[[VAL_91:.*]] = arith.constant 3 : index
! CHECK-DAG: %[[VAL_92:.*]] = arith.constant 0 : index
! CHECK-DAG: %[[VAL_93:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_94:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:   %[[VAL_95:.*]] = fir.convert %[[VAL_94]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_96:.*]] = fir.call @_FortranAioBeginExternalListInput(%[[VAL_86]], %[[VAL_95]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_97:.*]] = fir.shape_shift %[[VAL_89]], %[[VAL_84]], %[[VAL_91]], %[[VAL_85]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:   %[[VAL_98:.*]] = fir.undefined index
! CHECK:   %[[VAL_99:.*]] = fir.slice %[[VAL_88]], %[[VAL_98]], %[[VAL_98]], %[[VAL_93]], %[[VAL_91]], %[[VAL_93]] : (i64, index, index, index, index, index) -> !fir.slice<2>
! CHECK:   cf.br ^bb1(%[[VAL_92]], %[[VAL_91]] : index, index)
! CHECK: ^bb1(%[[VAL_100:.*]]: index, %[[VAL_101:.*]]: index):
! CHECK:   %[[VAL_102:.*]] = arith.cmpi sgt, %[[VAL_101]], %[[VAL_92]] : index
! CHECK:   cf.cond_br %[[VAL_102]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_103:.*]] = fir.coordinate_of %[[VAL_104]], %[[VAL_100]] : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_105:.*]] = fir.load %[[VAL_103]] : !fir.ref<i32>
! CHECK:   %[[VAL_106:.*]] = fir.convert %[[VAL_105]] : (i32) -> index
! CHECK:   %[[VAL_107:.*]] = fir.array_coor %[[VAL_108]](%[[VAL_97]]) {{\[}}%[[VAL_99]]] %[[VAL_91]], %[[VAL_106]] : (!fir.ref<!fir.array<4x6xi32>>, !fir.shapeshift<2>, !fir.slice<2>, index, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_109:.*]] = fir.convert %[[VAL_107]] : (!fir.ref<i32>) -> !fir.ref<i64>
! CHECK:   %[[VAL_110:.*]] = fir.call @_FortranAioInputInteger(%[[VAL_96]], %[[VAL_109]], %[[VAL_90]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i64>, i32) -> i1
! CHECK:   %[[VAL_111:.*]] = arith.addi %[[VAL_100]], %[[VAL_93]] overflow<nsw> : index
! CHECK:   %[[VAL_112:.*]] = arith.subi %[[VAL_101]], %[[VAL_93]] : index
! CHECK:   cf.br ^bb1(%[[VAL_111]], %[[VAL_112]] : index, index)
! CHECK: ^bb3:
! CHECK:   %[[VAL_113:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_96]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPtwo_vectors(
! CHECK-SAME: %[[VAL_140:.*]]: !fir.ref<!fir.array<4x4xf32>>{{.*}}, %[[VAL_132:.*]]: !fir.ref<!fir.array<3xi32>>{{.*}}, %[[VAL_136:.*]]: !fir.ref<!fir.array<3xi32>>{{.*}}) {
subroutine two_vectors(x, y1, y2)
  integer :: y1(3), y2(3)
  real :: x(4, 4)
  read(*,*) x(y1, y2)
! CHECK-DAG: %[[VAL_114:.*]] = arith.constant 4 : index
! CHECK-DAG: %[[VAL_115:.*]] = arith.constant 5 : i32
! CHECK-DAG: %[[VAL_117:.*]] = arith.constant 3 : index
! CHECK-DAG: %[[VAL_118:.*]] = arith.constant 0 : index
! CHECK-DAG: %[[VAL_119:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_120:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:   %[[VAL_121:.*]] = fir.convert %[[VAL_120]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_122:.*]] = fir.call @_FortranAioBeginExternalListInput(%[[VAL_115]], %[[VAL_121]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_123:.*]] = fir.shape %[[VAL_114]], %[[VAL_114]] : (index, index) -> !fir.shape<2>
! CHECK:   %[[VAL_124:.*]] = fir.slice %[[VAL_119]], %[[VAL_117]], %[[VAL_119]], %[[VAL_119]], %[[VAL_117]], %[[VAL_119]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:   cf.br ^bb1(%[[VAL_118]], %[[VAL_117]] : index, index)
! CHECK: ^bb1(%[[VAL_125:.*]]: index, %[[VAL_126:.*]]: index):
! CHECK:   %[[VAL_127:.*]] = arith.cmpi sgt, %[[VAL_126]], %[[VAL_118]] : index
! CHECK:   cf.cond_br %[[VAL_127]], ^bb2(%[[VAL_118]], %[[VAL_117]] : index, index), ^bb5
! CHECK: ^bb2(%[[VAL_128:.*]]: index, %[[VAL_129:.*]]: index):
! CHECK:   %[[VAL_130:.*]] = arith.cmpi sgt, %[[VAL_129]], %[[VAL_118]] : index
! CHECK:   cf.cond_br %[[VAL_130]], ^bb3, ^bb4
! CHECK: ^bb3:
! CHECK:   %[[VAL_131:.*]] = fir.coordinate_of %[[VAL_132]], %[[VAL_128]] : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_133:.*]] = fir.load %[[VAL_131]] : !fir.ref<i32>
! CHECK:   %[[VAL_134:.*]] = fir.convert %[[VAL_133]] : (i32) -> index
! CHECK:   %[[VAL_135:.*]] = fir.coordinate_of %[[VAL_136]], %[[VAL_125]] : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_137:.*]] = fir.load %[[VAL_135]] : !fir.ref<i32>
! CHECK:   %[[VAL_138:.*]] = fir.convert %[[VAL_137]] : (i32) -> index
! CHECK:   %[[VAL_139:.*]] = fir.array_coor %[[VAL_140]](%[[VAL_123]]) {{\[}}%[[VAL_124]]] %[[VAL_134]], %[[VAL_138]] : (!fir.ref<!fir.array<4x4xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:   %[[VAL_141:.*]] = fir.call @_FortranAioInputReal32(%[[VAL_122]], %[[VAL_139]]) {{.*}}: (!fir.ref<i8>, !fir.ref<f32>) -> i1
! CHECK:   %[[VAL_142:.*]] = arith.addi %[[VAL_128]], %[[VAL_119]] overflow<nsw> : index
! CHECK:   %[[VAL_143:.*]] = arith.subi %[[VAL_129]], %[[VAL_119]] : index
! CHECK:   cf.br ^bb2(%[[VAL_142]], %[[VAL_143]] : index, index)
! CHECK: ^bb4:
! CHECK:   %[[VAL_144:.*]] = arith.addi %[[VAL_125]], %[[VAL_119]] overflow<nsw> : index
! CHECK:   %[[VAL_145:.*]] = arith.subi %[[VAL_126]], %[[VAL_119]] : index
! CHECK:   cf.br ^bb1(%[[VAL_144]], %[[VAL_145]] : index, index)
! CHECK: ^bb5:
! CHECK:   %[[VAL_146:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_122]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPtriplets_and_vector(
! CHECK-SAME:    %[[VAL_170:.*]]: !fir.ref<!fir.array<4x4xcomplex<f32>>>{{.*}}, %[[VAL_166:.*]]: !fir.ref<!fir.array<3xi32>>{{.*}}) {
subroutine triplets_and_vector(x, y)
  integer :: y(3)
  complex :: x(4, 4)
  read(*,*) x(1:4:2, y)
! CHECK-DAG: %[[VAL_147:.*]] = arith.constant 5 : i32
! CHECK-DAG: %[[VAL_149:.*]] = arith.constant 4 : index
! CHECK-DAG: %[[VAL_150:.*]] = arith.constant 3 : index
! CHECK-DAG: %[[VAL_151:.*]] = arith.constant 2 : index
! CHECK-DAG: %[[VAL_152:.*]] = arith.constant 0 : index
! CHECK-DAG: %[[VAL_153:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_154:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:   %[[VAL_155:.*]] = fir.convert %[[VAL_154]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_156:.*]] = fir.call @_FortranAioBeginExternalListInput(%[[VAL_147]], %[[VAL_155]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_157:.*]] = fir.shape %[[VAL_149]], %[[VAL_149]] : (index, index) -> !fir.shape<2>
! CHECK:   %[[VAL_158:.*]] = fir.slice %[[VAL_153]], %[[VAL_149]], %[[VAL_151]], %[[VAL_153]], %[[VAL_150]], %[[VAL_153]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:   cf.br ^bb1(%[[VAL_152]], %[[VAL_150]] : index, index)
! CHECK: ^bb1(%[[VAL_159:.*]]: index, %[[VAL_160:.*]]: index):
! CHECK:   %[[VAL_161:.*]] = arith.cmpi sgt, %[[VAL_160]], %[[VAL_152]] : index
! CHECK:   cf.cond_br %[[VAL_161]], ^bb2(%[[VAL_153]], %[[VAL_151]] : index, index), ^bb5
! CHECK: ^bb2(%[[VAL_162:.*]]: index, %[[VAL_163:.*]]: index):
! CHECK:   %[[VAL_164:.*]] = arith.cmpi sgt, %[[VAL_163]], %[[VAL_152]] : index
! CHECK:   cf.cond_br %[[VAL_164]], ^bb3, ^bb4
! CHECK: ^bb3:
! CHECK:   %[[VAL_165:.*]] = fir.coordinate_of %[[VAL_166]], %[[VAL_159]] : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_167:.*]] = fir.load %[[VAL_165]] : !fir.ref<i32>
! CHECK:   %[[VAL_168:.*]] = fir.convert %[[VAL_167]] : (i32) -> index
! CHECK:   %[[VAL_169:.*]] = fir.array_coor %[[VAL_170]](%[[VAL_157]]) {{\[}}%[[VAL_158]]] %[[VAL_162]], %[[VAL_168]] : (!fir.ref<!fir.array<4x4xcomplex<f32>>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<complex<f32>>
! CHECK:   %[[VAL_171:.*]] = fir.convert %[[VAL_169]] : (!fir.ref<complex<f32>>) -> !fir.ref<f32>
! CHECK:   %[[VAL_172:.*]] = fir.call @_FortranAioInputComplex32(%[[VAL_156]], %[[VAL_171]]) {{.*}}: (!fir.ref<i8>, !fir.ref<f32>) -> i1
! CHECK:   %[[VAL_173:.*]] = arith.addi %[[VAL_162]], %[[VAL_153]] overflow<nsw> : index
! CHECK:   %[[VAL_174:.*]] = arith.subi %[[VAL_163]], %[[VAL_153]] : index
! CHECK:   cf.br ^bb2(%[[VAL_173]], %[[VAL_174]] : index, index)
! CHECK: ^bb4:
! CHECK:   %[[VAL_175:.*]] = arith.addi %[[VAL_159]], %[[VAL_153]] overflow<nsw> : index
! CHECK:   %[[VAL_176:.*]] = arith.subi %[[VAL_160]], %[[VAL_153]] : index
! CHECK:   cf.br ^bb1(%[[VAL_175]], %[[VAL_176]] : index, index)
! CHECK: ^bb5:
! CHECK:   %[[VAL_177:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_156]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPsimple_char(
! CHECK-SAME: %[[VAL_185:.*]]: !fir.boxchar<1>{{.*}}, %[[VAL_196:.*]]: !fir.ref<!fir.array<3xi32>>{{.*}}) {
subroutine simple_char(x, y)
  integer :: y(3)
  character(*) :: x(3:8)
  read(*,*) x(y)
! CHECK-DAG: %[[VAL_178:.*]] = arith.constant 6 : index
! CHECK-DAG: %[[VAL_179:.*]] = arith.constant 5 : i32
! CHECK-DAG: %[[VAL_181:.*]] = arith.constant 3 : index
! CHECK-DAG: %[[VAL_182:.*]] = arith.constant 0 : index
! CHECK-DAG: %[[VAL_183:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_184:.*]]:2 = fir.unboxchar %[[VAL_185]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:   %[[VAL_186:.*]] = fir.convert %[[VAL_184]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<6x!fir.char<1,?>>>
! CHECK:   %[[VAL_187:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:   %[[VAL_188:.*]] = fir.convert %[[VAL_187]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_189:.*]] = fir.call @_FortranAioBeginExternalListInput(%[[VAL_179]], %[[VAL_188]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_190:.*]] = fir.shape_shift %[[VAL_181]], %[[VAL_178]] : (index, index) -> !fir.shapeshift<1>
! CHECK:   %[[VAL_191:.*]] = fir.slice %[[VAL_183]], %[[VAL_181]], %[[VAL_183]] : (index, index, index) -> !fir.slice<1>
! CHECK:   cf.br ^bb1(%[[VAL_182]], %[[VAL_181]] : index, index)
! CHECK: ^bb1(%[[VAL_192:.*]]: index, %[[VAL_193:.*]]: index):
! CHECK:   %[[VAL_194:.*]] = arith.cmpi sgt, %[[VAL_193]], %[[VAL_182]] : index
! CHECK:   cf.cond_br %[[VAL_194]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_195:.*]] = fir.coordinate_of %[[VAL_196]], %[[VAL_192]] : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_197:.*]] = fir.load %[[VAL_195]] : !fir.ref<i32>
! CHECK:   %[[VAL_198:.*]] = fir.convert %[[VAL_197]] : (i32) -> index
! CHECK:   %[[VAL_199:.*]] = fir.array_coor %[[VAL_186]](%[[VAL_190]]) {{\[}}%[[VAL_191]]] %[[VAL_198]] typeparams %[[VAL_184]]#1 : (!fir.ref<!fir.array<6x!fir.char<1,?>>>, !fir.shapeshift<1>, !fir.slice<1>, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:   %[[VAL_200:.*]] = fir.convert %[[VAL_199]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_201:.*]] = fir.convert %[[VAL_184]]#1 : (index) -> i64
! CHECK:   %[[VAL_202:.*]] = fir.call @_FortranAioInputAscii(%[[VAL_189]], %[[VAL_200]], %[[VAL_201]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:   %[[VAL_203:.*]] = arith.addi %[[VAL_192]], %[[VAL_183]] overflow<nsw> : index
! CHECK:   %[[VAL_204:.*]] = arith.subi %[[VAL_193]], %[[VAL_183]] : index
! CHECK:   cf.br ^bb1(%[[VAL_203]], %[[VAL_204]] : index, index)
! CHECK: ^bb3:
! CHECK:   %[[VAL_205:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_189]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPsubstring(
! CHECK-SAME: %[[VAL_229:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>{{.*}}, %[[VAL_225:.*]]: !fir.ref<!fir.array<3xi32>>{{.*}}, %[[VAL_215:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_218:.*]]: !fir.ref<i32>{{.*}}) {
subroutine substring(x, y, i, j)
  integer :: y(3), i, j
  character(*) :: x(:)
  read(*,*) x(y)(i:j)
! CHECK-DAG: %[[VAL_206:.*]] = arith.constant 5 : i32
! CHECK-DAG: %[[VAL_208:.*]] = arith.constant 3 : index
! CHECK-DAG: %[[VAL_209:.*]] = arith.constant 0 : index
! CHECK-DAG: %[[VAL_210:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_211:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:   %[[VAL_212:.*]] = fir.convert %[[VAL_211]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_213:.*]] = fir.call @_FortranAioBeginExternalListInput(%[[VAL_206]], %[[VAL_212]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_214:.*]] = fir.load %[[VAL_215]] : !fir.ref<i32>
! CHECK:   %[[VAL_216:.*]] = fir.convert %[[VAL_214]] : (i32) -> index
! CHECK:   %[[VAL_217:.*]] = fir.load %[[VAL_218]] : !fir.ref<i32>
! CHECK:   %[[VAL_219:.*]] = fir.convert %[[VAL_217]] : (i32) -> index
! CHECK:   %[[VAL_220:.*]] = fir.slice %[[VAL_210]], %[[VAL_208]], %[[VAL_210]] : (index, index, index) -> !fir.slice<1>
! CHECK:   cf.br ^bb1(%[[VAL_209]], %[[VAL_208]] : index, index)
! CHECK: ^bb1(%[[VAL_221:.*]]: index, %[[VAL_222:.*]]: index):
! CHECK:   %[[VAL_223:.*]] = arith.cmpi sgt, %[[VAL_222]], %[[VAL_209]] : index
! CHECK:   cf.cond_br %[[VAL_223]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_224:.*]] = fir.coordinate_of %[[VAL_225]], %[[VAL_221]] : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_226:.*]] = fir.load %[[VAL_224]] : !fir.ref<i32>
! CHECK:   %[[VAL_227:.*]] = fir.convert %[[VAL_226]] : (i32) -> index
! CHECK:   %[[VAL_228:.*]] = fir.array_coor %[[VAL_229]] {{\[}}%[[VAL_220]]] %[[VAL_227]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.slice<1>, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:   %[[VAL_230:.*]] = arith.subi %[[VAL_216]], %[[VAL_210]] : index
! CHECK:   %[[VAL_231:.*]] = fir.convert %[[VAL_228]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:   %[[VAL_232:.*]] = fir.coordinate_of %[[VAL_231]], %[[VAL_230]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:   %[[VAL_234:.*]] = arith.subi %[[VAL_219]], %[[VAL_216]] : index
! CHECK:   %[[VAL_235:.*]] = arith.addi %[[VAL_234]], %[[VAL_210]] : index
! CHECK:   %[[VAL_236:.*]] = arith.cmpi slt, %[[VAL_235]], %[[VAL_209]] : index
! CHECK:   %[[VAL_237:.*]] = arith.select %[[VAL_236]], %[[VAL_209]], %[[VAL_235]] : index
! CHECK:   %[[VAL_238:.*]] = fir.convert %[[VAL_232]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_239:.*]] = fir.convert %[[VAL_237]] : (index) -> i64
! CHECK:   %[[VAL_240:.*]] = fir.call @_FortranAioInputAscii(%[[VAL_213]], %[[VAL_238]], %[[VAL_239]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:   %[[VAL_241:.*]] = arith.addi %[[VAL_221]], %[[VAL_210]] overflow<nsw> : index
! CHECK:   %[[VAL_242:.*]] = arith.subi %[[VAL_222]], %[[VAL_210]] : index
! CHECK:   cf.br ^bb1(%[[VAL_241]], %[[VAL_242]] : index, index)
! CHECK: ^bb3:
! CHECK:   %[[VAL_243:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_213]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPcomplex_part(
! CHECK-SAME: %[[VAL_262:.*]]: !fir.box<!fir.array<?xcomplex<f32>>>{{.*}}, %[[VAL_253:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) {
subroutine complex_part(z, y)
  integer :: y(:)
  complex :: z(:)
  read(*,*) z(y)%IM
! CHECK-DAG: %[[VAL_244:.*]] = arith.constant 5 : i32
! CHECK-DAG: %[[VAL_246:.*]] = arith.constant 1 : i32
! CHECK-DAG: %[[VAL_247:.*]] = arith.constant 0 : index
! CHECK-DAG: %[[VAL_248:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_249:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:   %[[VAL_250:.*]] = fir.convert %[[VAL_249]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_251:.*]] = fir.call @_FortranAioBeginExternalListInput(%[[VAL_244]], %[[VAL_250]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_252:.*]]:3 = fir.box_dims %[[VAL_253]], %[[VAL_247]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:   %[[VAL_254:.*]] = fir.slice %[[VAL_248]], %[[VAL_252]]#1, %[[VAL_248]] path %[[VAL_246]] : (index, index, index, i32) -> !fir.slice<1>
! CHECK:   cf.br ^bb1(%[[VAL_247]], %[[VAL_252]]#1 : index, index)
! CHECK: ^bb1(%[[VAL_255:.*]]: index, %[[VAL_256:.*]]: index):
! CHECK:   %[[VAL_257:.*]] = arith.cmpi sgt, %[[VAL_256]], %[[VAL_247]] : index
! CHECK:   cf.cond_br %[[VAL_257]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_258:.*]] = fir.coordinate_of %[[VAL_253]], %[[VAL_255]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_259:.*]] = fir.load %[[VAL_258]] : !fir.ref<i32>
! CHECK:   %[[VAL_260:.*]] = fir.convert %[[VAL_259]] : (i32) -> index
! CHECK:   %[[VAL_261:.*]] = fir.array_coor %[[VAL_262]] {{\[}}%[[VAL_254]]] %[[VAL_260]] : (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.slice<1>, index) -> !fir.ref<f32>
! CHECK:   %[[VAL_263:.*]] = fir.call @_FortranAioInputReal32(%[[VAL_251]], %[[VAL_261]]) {{.*}}: (!fir.ref<i8>, !fir.ref<f32>) -> i1
! CHECK:   %[[VAL_264:.*]] = arith.addi %[[VAL_255]], %[[VAL_248]] overflow<nsw> : index
! CHECK:   %[[VAL_265:.*]] = arith.subi %[[VAL_256]], %[[VAL_248]] : index
! CHECK:   cf.br ^bb1(%[[VAL_264]], %[[VAL_265]] : index, index)
! CHECK: ^bb3:
! CHECK:   %[[VAL_266:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_251]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

module derived_types
  type t
    integer :: i
    character(2) :: c
  end type
  type t2
    type(t) :: a(5,5)
  end type
end module

! CHECK-LABEL: func @_QPsimple_derived(
! CHECK-SAME: %[[VAL_287:.*]]: !fir.ref<!fir.array<6x!fir.type<_QMderived_typesTt{i:i32,c:!fir.char<1,2>}>>>{{.*}}, %[[VAL_283:.*]]: !fir.ref<!fir.array<4xi32>>{{.*}}) {
subroutine simple_derived(x, y)
  use derived_types
  integer :: y(4)
  type(t) :: x(3:8)
  read(*,*) x(y)
! CHECK-DAG: %[[VAL_267:.*]] = arith.constant 6 : index
! CHECK-DAG: %[[VAL_268:.*]] = arith.constant 5 : i32
! CHECK-DAG: %[[VAL_270:.*]] = arith.constant 3 : index
! CHECK-DAG: %[[VAL_271:.*]] = arith.constant 4 : index
! CHECK-DAG: %[[VAL_272:.*]] = arith.constant 0 : index
! CHECK-DAG: %[[VAL_273:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_274:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:   %[[VAL_275:.*]] = fir.convert %[[VAL_274]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_276:.*]] = fir.call @_FortranAioBeginExternalListInput(%[[VAL_268]], %[[VAL_275]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_277:.*]] = fir.shape_shift %[[VAL_270]], %[[VAL_267]] : (index, index) -> !fir.shapeshift<1>
! CHECK:   %[[VAL_278:.*]] = fir.slice %[[VAL_273]], %[[VAL_271]], %[[VAL_273]] : (index, index, index) -> !fir.slice<1>
! CHECK:   cf.br ^bb1(%[[VAL_272]], %[[VAL_271]] : index, index)
! CHECK: ^bb1(%[[VAL_279:.*]]: index, %[[VAL_280:.*]]: index):
! CHECK:   %[[VAL_281:.*]] = arith.cmpi sgt, %[[VAL_280]], %[[VAL_272]] : index
! CHECK:   cf.cond_br %[[VAL_281]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_282:.*]] = fir.coordinate_of %[[VAL_283]], %[[VAL_279]] : (!fir.ref<!fir.array<4xi32>>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_284:.*]] = fir.load %[[VAL_282]] : !fir.ref<i32>
! CHECK:   %[[VAL_285:.*]] = fir.convert %[[VAL_284]] : (i32) -> index
! CHECK:   %[[VAL_286:.*]] = fir.array_coor %[[VAL_287]](%[[VAL_277]]) {{\[}}%[[VAL_278]]] %[[VAL_285]] : (!fir.ref<!fir.array<6x!fir.type<_QMderived_typesTt{i:i32,c:!fir.char<1,2>}>>>, !fir.shapeshift<1>, !fir.slice<1>, index) -> !fir.ref<!fir.type<_QMderived_typesTt{i:i32,c:!fir.char<1,2>}>>
! CHECK:   %[[VAL_288:.*]] = fir.embox %[[VAL_286]] : (!fir.ref<!fir.type<_QMderived_typesTt{i:i32,c:!fir.char<1,2>}>>) -> !fir.box<!fir.type<_QMderived_typesTt{i:i32,c:!fir.char<1,2>}>>
! CHECK:   %[[VAL_289:.*]] = fir.convert %[[VAL_288]] : (!fir.box<!fir.type<_QMderived_typesTt{i:i32,c:!fir.char<1,2>}>>) -> !fir.box<none>
! CHECK:   %[[VAL_290:.*]] = fir.call @_FortranAioInputDerivedType(%[[VAL_276]], %[[VAL_289]], {{.*}}) {{.*}}: (!fir.ref<i8>, !fir.box<none>, !fir.ref<none>) -> i1
! CHECK:   %[[VAL_291:.*]] = arith.addi %[[VAL_279]], %[[VAL_273]] overflow<nsw> : index
! CHECK:   %[[VAL_292:.*]] = arith.subi %[[VAL_280]], %[[VAL_273]] : index
! CHECK:   cf.br ^bb1(%[[VAL_291]], %[[VAL_292]] : index, index)
! CHECK: ^bb3:
! CHECK:   %[[VAL_293:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_276]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPwith_path(
! CHECK-SAME: [[VAL_326:.*]]: !fir.box<!fir.array<?x?x?x!fir.type<_QMderived_typesTt2{a:!fir.array<5x5x!fir.type<_QMderived_typesTt{i:i32,c:!fir.char<1,2>}>>}>>>{{.*}}, [[VAL_310:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) {
subroutine with_path(b, i)
  use derived_types
  type(t2) :: b(4:, 4:, 4:)
  integer :: i(:)
  read (*, *) b(5, i, 8:9:1)%a(4,5)%i
! CHECK-DAG: %[[VAL_294:.*]] = arith.constant 4 : index
! CHECK-DAG: %[[VAL_295:.*]] = arith.constant 5 : i32
! CHECK-DAG: %[[VAL_297:.*]] = arith.constant 8 : index
! CHECK-DAG: %[[VAL_298:.*]] = arith.constant 9 : index
! CHECK-DAG: %[[VAL_299:.*]] = arith.constant 4 : i64
! CHECK-DAG: %[[VAL_300:.*]] = arith.constant 5 : i64
! CHECK-DAG: %[[VAL_301:.*]] = arith.constant 5 : index
! CHECK-DAG: %[[VAL_302:.*]] = arith.constant 4 : i32
! CHECK-DAG: %[[VAL_303:.*]] = arith.constant 2 : index
! CHECK-DAG: %[[VAL_304:.*]] = arith.constant 0 : index
! CHECK-DAG: %[[VAL_305:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_306:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:   %[[VAL_307:.*]] = fir.convert %[[VAL_306]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_308:.*]] = fir.call @_FortranAioBeginExternalListInput(%[[VAL_295]], %[[VAL_307]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_309:.*]]:3 = fir.box_dims %[[VAL_310:.*]], %[[VAL_304]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:   %[[VAL_311:.*]] = fir.field_index a, !fir.type<_QMderived_typesTt2{a:!fir.array<5x5x!fir.type<_QMderived_typesTt{i:i32,c:!fir.char<1,2>}>>}>
! CHECK:   %[[VAL_312:.*]] = fir.field_index i, !fir.type<_QMderived_typesTt{i:i32,c:!fir.char<1,2>}>
! CHECK:   %[[VAL_313:.*]] = fir.shift %[[VAL_294]], %[[VAL_294]], %[[VAL_294]] : (index, index, index) -> !fir.shift<3>
! CHECK:   %[[VAL_314:.*]] = fir.undefined index
! CHECK:   %[[VAL_315:.*]] = fir.slice %[[VAL_300]], %[[VAL_314]], %[[VAL_314]], %[[VAL_305]], %[[VAL_309]]#1, %[[VAL_305]], %[[VAL_297]], %[[VAL_298]], %[[VAL_305]] path %[[VAL_311]], %[[VAL_299]], %[[VAL_300]], %[[VAL_312]] : (i64, index, index, index, index, index, index, index, index, !fir.field, i64, i64, !fir.field) -> !fir.slice<3>
! CHECK:   cf.br ^bb1(%[[VAL_294]], %[[VAL_303]] : index, index)
! CHECK: ^bb1(%[[VAL_316:.*]]: index, %[[VAL_317:.*]]: index):
! CHECK:   %[[VAL_318:.*]] = arith.cmpi sgt, %[[VAL_317]], %[[VAL_304]] : index
! CHECK:   cf.cond_br %[[VAL_318]], ^bb2(%[[VAL_304]], %[[VAL_309]]#1 : index, index), ^bb5
! CHECK: ^bb2(%[[VAL_319:.*]]: index, %[[VAL_320:.*]]: index):
! CHECK:   %[[VAL_321:.*]] = arith.cmpi sgt, %[[VAL_320]], %[[VAL_304]] : index
! CHECK:   cf.cond_br %[[VAL_321]], ^bb3, ^bb4
! CHECK: ^bb3:
! CHECK:   %[[VAL_322:.*]] = fir.coordinate_of %[[VAL_310]], %[[VAL_319]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_323:.*]] = fir.load %[[VAL_322]] : !fir.ref<i32>
! CHECK:   %[[VAL_324:.*]] = fir.convert %[[VAL_323]] : (i32) -> index
! CHECK:   %[[VAL_325:.*]] = fir.array_coor %[[VAL_326:.*]](%[[VAL_313]]) {{\[}}%[[VAL_315]]] %[[VAL_301]], %[[VAL_324]], %[[VAL_316]] : (!fir.box<!fir.array<?x?x?x!fir.type<_QMderived_typesTt2{a:!fir.array<5x5x!fir.type<_QMderived_typesTt{i:i32,c:!fir.char<1,2>}>>}>>>, !fir.shift<3>, !fir.slice<3>, index, index, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_327:.*]] = fir.convert %[[VAL_325]] : (!fir.ref<i32>) -> !fir.ref<i64>
! CHECK:   %[[VAL_328:.*]] = fir.call @_FortranAioInputInteger(%[[VAL_308]], %[[VAL_327]], %[[VAL_302]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i64>, i32) -> i1
! CHECK:   %[[VAL_329:.*]] = arith.addi %[[VAL_319]], %[[VAL_305]] overflow<nsw> : index
! CHECK:   %[[VAL_330:.*]] = arith.subi %[[VAL_320]], %[[VAL_305]] : index
! CHECK:   cf.br ^bb2(%[[VAL_329]], %[[VAL_330]] : index, index)
! CHECK: ^bb4:
! CHECK:   %[[VAL_331:.*]] = arith.addi %[[VAL_316]], %[[VAL_305]] overflow<nsw> : index
! CHECK:   %[[VAL_332:.*]] = arith.subi %[[VAL_317]], %[[VAL_305]] : index
! CHECK:   cf.br ^bb1(%[[VAL_331]], %[[VAL_332]] : index, index)
! CHECK: ^bb5:
! CHECK:   %[[VAL_333:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_308]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPsimple_iostat(
! CHECK-SAME: %[[VAL_357:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}, %[[VAL_346:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[VAL_361:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_364:.*]]: !fir.ref<i32>{{.*}}) {
subroutine simple_iostat(x, y, j, stat)
  integer :: j, y(:), stat
  real :: x(:)
  read(*, *, iostat=stat) x(y), j
! CHECK-DAG: %[[VAL_334:.*]] = arith.constant 5 : i32
! CHECK-DAG: %[[VAL_336:.*]] = arith.constant false
! CHECK-DAG: %[[VAL_337:.*]] = arith.constant true
! CHECK-DAG: %[[VAL_338:.*]] = arith.constant 1 : index
! CHECK-DAG: %[[VAL_339:.*]] = arith.constant 0 : index
! CHECK-DAG: %[[VAL_340:.*]] = arith.constant 4 : i32
! CHECK:   %[[VAL_341:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:   %[[VAL_342:.*]] = fir.convert %[[VAL_341]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_343:.*]] = fir.call @_FortranAioBeginExternalListInput(%[[VAL_334]], %[[VAL_342]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   fir.call @_FortranAioEnableHandlers(%[[VAL_343]], %[[VAL_337]], %[[VAL_336]], %[[VAL_336]], %[[VAL_336]], %[[VAL_336]]) {{.*}}: (!fir.ref<i8>, i1, i1, i1, i1, i1) -> ()
! CHECK:   %[[VAL_345:.*]]:3 = fir.box_dims %[[VAL_346]], %[[VAL_339]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:   %[[VAL_347:.*]] = fir.slice %[[VAL_338]], %[[VAL_345]]#1, %[[VAL_338]] : (index, index, index) -> !fir.slice<1>
! CHECK:   %[[VAL_348:.*]] = arith.subi %[[VAL_345]]#1, %[[VAL_338]] : index
! CHECK:   cf.br ^bb1(%[[VAL_339]], %[[VAL_337]] : index, i1)
! CHECK: ^bb1(%[[VAL_349:.*]]: index, %[[VAL_350:.*]]: i1):
! CHECK:   %[[VAL_351:.*]] = arith.cmpi sle, %[[VAL_349]], %[[VAL_348]] : index
! CHECK:   %[[VAL_352:.*]] = arith.andi %[[VAL_350]], %[[VAL_351]] : i1
! CHECK:   cf.cond_br %[[VAL_352]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_353:.*]] = fir.coordinate_of %[[VAL_346]], %[[VAL_349]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_354:.*]] = fir.load %[[VAL_353]] : !fir.ref<i32>
! CHECK:   %[[VAL_355:.*]] = fir.convert %[[VAL_354]] : (i32) -> index
! CHECK:   %[[VAL_356:.*]] = fir.array_coor %[[VAL_357]] {{\[}}%[[VAL_347]]] %[[VAL_355]] : (!fir.box<!fir.array<?xf32>>, !fir.slice<1>, index) -> !fir.ref<f32>
! CHECK:   %[[VAL_358:.*]] = fir.call @_FortranAioInputReal32(%[[VAL_343]], %[[VAL_356]]) {{.*}}: (!fir.ref<i8>, !fir.ref<f32>) -> i1
! CHECK:   %[[VAL_359:.*]] = arith.addi %[[VAL_349]], %[[VAL_338]] overflow<nsw> : index
! CHECK:   cf.br ^bb1(%[[VAL_359]], %[[VAL_358]] : index, i1)
! CHECK: ^bb3:
! CHECK:   cf.cond_br %[[VAL_350]], ^bb4, ^bb5
! CHECK: ^bb4:
! CHECK:   %[[VAL_360:.*]] = fir.convert %[[VAL_361]] : (!fir.ref<i32>) -> !fir.ref<i64>
! CHECK:   %[[VAL_362:.*]] = fir.call @_FortranAioInputInteger(%[[VAL_343]], %[[VAL_360]], %[[VAL_340]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i64>, i32) -> i1
! CHECK:   cf.br ^bb5
! CHECK: ^bb5:
! CHECK:   %[[VAL_363:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_343]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   fir.store %[[VAL_363]] to %[[VAL_364]] : !fir.ref<i32>
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPiostat_in_io_loop(
! CHECK-SAME: %[[VAL_400:.*]]: !fir.ref<!fir.array<3x5xi32>>{{.*}}, %[[VAL_396:.*]]: !fir.ref<!fir.array<3xi32>>{{.*}}, %[[VAL_408:.*]]: !fir.ref<i32>{{.*}}) {
subroutine iostat_in_io_loop(k, j, stat)
  integer :: k(3, 5)
  integer :: j(3)
  integer  :: stat
  read(*, *, iostat=stat) (k(i, j), i=1,3,1)
! CHECK-DAG: %[[VAL_365:.*]] = arith.constant 5 : index
! CHECK-DAG: %[[VAL_366:.*]] = arith.constant 5 : i32
! CHECK-DAG: %[[VAL_368:.*]] = arith.constant 3 : index
! CHECK-DAG: %[[VAL_369:.*]] = arith.constant true
! CHECK-DAG: %[[VAL_370:.*]] = arith.constant false
! CHECK-DAG: %[[VAL_371:.*]] = arith.constant 1 : index
! CHECK-DAG: %[[VAL_372:.*]] = arith.constant 0 : index
! CHECK-DAG: %[[VAL_373:.*]] = arith.constant 2 : index
! CHECK-DAG: %[[VAL_374:.*]] = arith.constant 4 : i32
! CHECK:   %[[VAL_375:.*]] = fir.alloca i32
! CHECK:   %[[VAL_376:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:   %[[VAL_377:.*]] = fir.convert %[[VAL_376]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_378:.*]] = fir.call @_FortranAioBeginExternalListInput(%[[VAL_366]], %[[VAL_377]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   fir.call @_FortranAioEnableHandlers(%[[VAL_378]], %[[VAL_369]], %[[VAL_370]], %[[VAL_370]], %[[VAL_370]], %[[VAL_370]]) {{.*}}: (!fir.ref<i8>, i1, i1, i1, i1, i1) -> ()
! CHECK:   cf.br ^bb1(%[[VAL_371]], %[[VAL_369]] : index, i1)
! CHECK: ^bb1(%[[VAL_380:.*]]: index, %[[VAL_381:.*]]: i1):
! CHECK:   %[[VAL_382:.*]] = arith.cmpi sle, %[[VAL_380]], %[[VAL_368]] : index
! CHECK:   %[[VAL_383:.*]] = arith.andi %[[VAL_381]], %[[VAL_382]] : i1
! CHECK:   cf.cond_br %[[VAL_383]], ^bb2, ^bb7
! CHECK: ^bb2:
! CHECK:   %[[VAL_384:.*]] = fir.convert %[[VAL_380]] : (index) -> i32
! CHECK:   fir.store %[[VAL_384]] to %[[VAL_375]] : !fir.ref<i32>
! CHECK:   cf.cond_br %[[VAL_381]], ^bb3, ^bb6(%[[VAL_370]] : i1)
! CHECK: ^bb3:
! CHECK:   %[[VAL_385:.*]] = fir.load %[[VAL_375]] : !fir.ref<i32>
! CHECK:   %[[VAL_386:.*]] = fir.convert %[[VAL_385]] : (i32) -> i64
! CHECK:   %[[VAL_387:.*]] = fir.shape %[[VAL_368]], %[[VAL_365]] : (index, index) -> !fir.shape<2>
! CHECK:   %[[VAL_388:.*]] = fir.undefined index
! CHECK:   %[[VAL_389:.*]] = fir.slice %[[VAL_386]], %[[VAL_388]], %[[VAL_388]], %[[VAL_371]], %[[VAL_368]], %[[VAL_371]] : (i64, index, index, index, index, index) -> !fir.slice<2>
! CHECK:   cf.br ^bb4(%[[VAL_372]], %[[VAL_369]] : index, i1)
! CHECK: ^bb4(%[[VAL_390:.*]]: index, %[[VAL_391:.*]]: i1):
! CHECK:   %[[VAL_392:.*]] = arith.cmpi sle, %[[VAL_390]], %[[VAL_373]] : index
! CHECK:   %[[VAL_393:.*]] = arith.andi %[[VAL_391]], %[[VAL_392]] : i1
! CHECK:   cf.cond_br %[[VAL_393]], ^bb5, ^bb6(%[[VAL_391]] : i1)
! CHECK: ^bb5:
! CHECK:   %[[VAL_394:.*]] = fir.convert %[[VAL_385]] : (i32) -> index
! CHECK:   %[[VAL_395:.*]] = fir.coordinate_of %[[VAL_396]], %[[VAL_390]] : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_397:.*]] = fir.load %[[VAL_395]] : !fir.ref<i32>
! CHECK:   %[[VAL_398:.*]] = fir.convert %[[VAL_397]] : (i32) -> index
! CHECK:   %[[VAL_399:.*]] = fir.array_coor %[[VAL_400]](%[[VAL_387]]) {{\[}}%[[VAL_389]]] %[[VAL_394]], %[[VAL_398]] : (!fir.ref<!fir.array<3x5xi32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_401:.*]] = fir.convert %[[VAL_399]] : (!fir.ref<i32>) -> !fir.ref<i64>
! CHECK:   %[[VAL_402:.*]] = fir.call @_FortranAioInputInteger(%[[VAL_378]], %[[VAL_401]], %[[VAL_374]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i64>, i32) -> i1
! CHECK:   %[[VAL_403:.*]] = arith.addi %[[VAL_390]], %[[VAL_371]] overflow<nsw> : index
! CHECK:   cf.br ^bb4(%[[VAL_403]], %[[VAL_402]] : index, i1)
! CHECK: ^bb6(%[[VAL_404:.*]]: i1):
! CHECK:   %[[VAL_405:.*]] = arith.addi %[[VAL_380]], %[[VAL_371]] overflow<nsw> : index
! CHECK:   cf.br ^bb1(%[[VAL_405]], %[[VAL_404]] : index, i1)
! CHECK: ^bb7:
! CHECK:   %[[VAL_406:.*]] = fir.convert %[[VAL_380]] : (index) -> i32
! CHECK:   fir.store %[[VAL_406]] to %[[VAL_375]] : !fir.ref<i32>
! CHECK:   %[[VAL_407:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_378]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   fir.store %[[VAL_407]] to %[[VAL_408]] : !fir.ref<i32>
! CHECK:   return
end subroutine
