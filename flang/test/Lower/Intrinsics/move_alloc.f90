  ! RUN: bbc --use-desc-for-alloc=false -emit-fir -hlfir=false %s -o - | FileCheck  %s
  ! RUN: %flang_fc1 -mllvm --use-desc-for-alloc=false -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

subroutine to_from_only
  integer, allocatable :: from(:), to(:)
  allocate(from(20))
  call move_alloc(from, to)
end subroutine to_from_only
! CHECK-LABEL:   func.func @_QPto_from_only() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "from", uniq_name = "_QFto_from_onlyEfrom"}
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.heap<!fir.array<?xi32>> {uniq_name = "_QFto_from_onlyEfrom.addr"}
! CHECK:           %[[VAL_2:.*]] = fir.alloca index {uniq_name = "_QFto_from_onlyEfrom.lb0"}
! CHECK:           %[[VAL_3:.*]] = fir.alloca index {uniq_name = "_QFto_from_onlyEfrom.ext0"}
! CHECK:           %[[VAL_4:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "to", uniq_name = "_QFto_from_onlyEto"}
! CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.heap<!fir.array<?xi32>> {uniq_name = "_QFto_from_onlyEto.addr"}
! CHECK:           %[[VAL_7:.*]] = fir.alloca index {uniq_name = "_QFto_from_onlyEto.lb0"}
! CHECK:           %[[VAL_8:.*]] = fir.alloca index {uniq_name = "_QFto_from_onlyEto.ext0"}
! CHECK:           %[[VAL_9:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.store %[[VAL_9]] to %[[VAL_6]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_10:.*]] = arith.constant false
! CHECK:           %[[VAL_11:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_12:.*]] = fir.address_of(@_QQclX8c4f3bfd49fac4527014c85e4d2f3268) : !fir.ref<!fir.char<1,74>>
! CHECK:           %[[VAL_13:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_14:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_17:.*]] = fir.shape_shift %[[VAL_14]], %[[VAL_15]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_18:.*]] = fir.embox %[[VAL_16]](%[[VAL_17]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_18]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_20:.*]] = arith.constant 20 : i32
! CHECK:           %[[VAL_21:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_19]] : (index) -> i64
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_20]] : (i32) -> i64
! CHECK:           fir.call @_FortranAAllocatableSetBounds(%[[VAL_22]], %[[VAL_21]], %[[VAL_23]], %[[VAL_24]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_12]] : (!fir.ref<!fir.char<1,74>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_27:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_25]], %[[VAL_10]], %[[VAL_11]], %[[VAL_26]], %[[VAL_13]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_28:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_29:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_30:.*]]:3 = fir.box_dims %[[VAL_28]], %[[VAL_29]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_31:.*]] = fir.box_addr %[[VAL_28]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.store %[[VAL_31]] to %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_30]]#1 to %[[VAL_3]] : !fir.ref<index>
! CHECK:           fir.store %[[VAL_30]]#0 to %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_32:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_33:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_34:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_35:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_36:.*]] = fir.shape_shift %[[VAL_33]], %[[VAL_34]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_37:.*]] = fir.embox %[[VAL_35]](%[[VAL_36]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_37]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_38:.*]] = fir.load %[[VAL_7]] : !fir.ref<index>
! CHECK:           %[[VAL_39:.*]] = fir.load %[[VAL_8]] : !fir.ref<index>
! CHECK:           %[[VAL_40:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_41:.*]] = fir.shape_shift %[[VAL_38]], %[[VAL_39]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_42:.*]] = fir.embox %[[VAL_40]](%[[VAL_41]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_42]] to %[[VAL_5]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_43:.*]] = arith.constant false
! CHECK:           %[[VAL_44:.*]] = fir.address_of(@_QQclX8c4f3bfd49fac4527014c85e4d2f3268) : !fir.ref<!fir.char<1,74>>
! CHECK:           %[[VAL_45:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_46:.*]] = fir.zero_bits !fir.ref<none>
! CHECK:           %[[VAL_47:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_48:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_49:.*]] = fir.convert %[[VAL_44]] : (!fir.ref<!fir.char<1,74>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_50:.*]] = fir.call @_FortranAMoveAlloc(%[[VAL_47]], %[[VAL_48]], %[[VAL_46]], %[[VAL_43]], %[[VAL_32]], %[[VAL_49]], %[[VAL_45]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_51:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_52:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_53:.*]]:3 = fir.box_dims %[[VAL_51]], %[[VAL_52]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_54:.*]] = fir.box_addr %[[VAL_51]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.store %[[VAL_54]] to %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_53]]#1 to %[[VAL_3]] : !fir.ref<index>
! CHECK:           fir.store %[[VAL_53]]#0 to %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_55:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_56:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_57:.*]]:3 = fir.box_dims %[[VAL_55]], %[[VAL_56]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_58:.*]] = fir.box_addr %[[VAL_55]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.store %[[VAL_58]] to %[[VAL_6]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_57]]#1 to %[[VAL_8]] : !fir.ref<index>
! CHECK:           fir.store %[[VAL_57]]#0 to %[[VAL_7]] : !fir.ref<index>
! CHECK:           %[[VAL_59:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_60:.*]] = fir.convert %[[VAL_59]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK:           %[[VAL_61:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_62:.*]] = arith.cmpi ne, %[[VAL_60]], %[[VAL_61]] : i64
! CHECK:           fir.if %[[VAL_62]] {
! CHECK:             %[[VAL_63:.*]] = arith.constant false
! CHECK:             %[[VAL_64:.*]] = fir.absent !fir.box<none>
! CHECK:             %[[VAL_65:.*]] = fir.address_of(@_QQclX8c4f3bfd49fac4527014c85e4d2f3268) : !fir.ref<!fir.char<1,74>>
! CHECK:             %[[VAL_66:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_67:.*]] = fir.load %[[VAL_7]] : !fir.ref<index>
! CHECK:             %[[VAL_68:.*]] = fir.load %[[VAL_8]] : !fir.ref<index>
! CHECK:             %[[VAL_69:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:             %[[VAL_70:.*]] = fir.shape_shift %[[VAL_67]], %[[VAL_68]] : (index, index) -> !fir.shapeshift<1>
! CHECK:             %[[VAL_71:.*]] = fir.embox %[[VAL_69]](%[[VAL_70]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:             fir.store %[[VAL_71]] to %[[VAL_5]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:             %[[VAL_72:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_73:.*]] = fir.convert %[[VAL_65]] : (!fir.ref<!fir.char<1,74>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_74:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_72]], %[[VAL_63]], %[[VAL_64]], %[[VAL_73]], %[[VAL_66]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:             %[[VAL_75:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:             %[[VAL_76:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_77:.*]]:3 = fir.box_dims %[[VAL_75]], %[[VAL_76]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_78:.*]] = fir.box_addr %[[VAL_75]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:             fir.store %[[VAL_78]] to %[[VAL_6]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:             fir.store %[[VAL_77]]#1 to %[[VAL_8]] : !fir.ref<index>
! CHECK:             fir.store %[[VAL_77]]#0 to %[[VAL_7]] : !fir.ref<index>
! CHECK:           }
! CHECK:           %[[VAL_79:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_80:.*]] = fir.convert %[[VAL_79]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK:           %[[VAL_81:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_82:.*]] = arith.cmpi ne, %[[VAL_80]], %[[VAL_81]] : i64
! CHECK:           fir.if %[[VAL_82]] {
! CHECK:             %[[VAL_83:.*]] = arith.constant false
! CHECK:             %[[VAL_84:.*]] = fir.absent !fir.box<none>
! CHECK:             %[[VAL_85:.*]] = fir.address_of(@_QQclX8c4f3bfd49fac4527014c85e4d2f3268) : !fir.ref<!fir.char<1,74>>
! CHECK:             %[[VAL_86:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_87:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
! CHECK:             %[[VAL_88:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:             %[[VAL_89:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:             %[[VAL_90:.*]] = fir.shape_shift %[[VAL_87]], %[[VAL_88]] : (index, index) -> !fir.shapeshift<1>
! CHECK:             %[[VAL_91:.*]] = fir.embox %[[VAL_89]](%[[VAL_90]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:             fir.store %[[VAL_91]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:             %[[VAL_92:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_93:.*]] = fir.convert %[[VAL_85]] : (!fir.ref<!fir.char<1,74>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_94:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_92]], %[[VAL_83]], %[[VAL_84]], %[[VAL_93]], %[[VAL_86]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:             %[[VAL_95:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:             %[[VAL_96:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_97:.*]]:3 = fir.box_dims %[[VAL_95]], %[[VAL_96]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_98:.*]] = fir.box_addr %[[VAL_95]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:             fir.store %[[VAL_98]] to %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:             fir.store %[[VAL_97]]#1 to %[[VAL_3]] : !fir.ref<index>
! CHECK:             fir.store %[[VAL_97]]#0 to %[[VAL_2]] : !fir.ref<index>
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine to_from_stat
  integer, allocatable :: from(:), to(:)
  integer :: stat
  allocate(from(20))
  call move_alloc(from, to, stat)
end subroutine to_from_stat
! CHECK-LABEL:   func.func @_QPto_from_stat() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "from", uniq_name = "_QFto_from_statEfrom"}
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.heap<!fir.array<?xi32>> {uniq_name = "_QFto_from_statEfrom.addr"}
! CHECK:           %[[VAL_2:.*]] = fir.alloca index {uniq_name = "_QFto_from_statEfrom.lb0"}
! CHECK:           %[[VAL_3:.*]] = fir.alloca index {uniq_name = "_QFto_from_statEfrom.ext0"}
! CHECK:           %[[VAL_4:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_5:.*]] = fir.alloca i32 {bindc_name = "stat", uniq_name = "_QFto_from_statEstat"}
! CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "to", uniq_name = "_QFto_from_statEto"}
! CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.heap<!fir.array<?xi32>> {uniq_name = "_QFto_from_statEto.addr"}
! CHECK:           %[[VAL_8:.*]] = fir.alloca index {uniq_name = "_QFto_from_statEto.lb0"}
! CHECK:           %[[VAL_9:.*]] = fir.alloca index {uniq_name = "_QFto_from_statEto.ext0"}
! CHECK:           %[[VAL_10:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.store %[[VAL_10]] to %[[VAL_7]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_11:.*]] = arith.constant false
! CHECK:           %[[VAL_12:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_13:.*]] = fir.address_of(@_QQclX8c4f3bfd49fac4527014c85e4d2f3268) : !fir.ref<!fir.char<1,74>>
! CHECK:           %[[VAL_14:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_18:.*]] = fir.shape_shift %[[VAL_15]], %[[VAL_16]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_19:.*]] = fir.embox %[[VAL_17]](%[[VAL_18]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_19]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_20:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_21:.*]] = arith.constant 20 : i32
! CHECK:           %[[VAL_22:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_20]] : (index) -> i64
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_21]] : (i32) -> i64
! CHECK:           fir.call @_FortranAAllocatableSetBounds(%[[VAL_23]], %[[VAL_22]], %[[VAL_24]], %[[VAL_25]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.char<1,74>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_28:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_26]], %[[VAL_11]], %[[VAL_12]], %[[VAL_27]], %[[VAL_14]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_29:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_30:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_31:.*]]:3 = fir.box_dims %[[VAL_29]], %[[VAL_30]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_32:.*]] = fir.box_addr %[[VAL_29]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.store %[[VAL_32]] to %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_31]]#1 to %[[VAL_3]] : !fir.ref<index>
! CHECK:           fir.store %[[VAL_31]]#0 to %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_33:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_34:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_35:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_36:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_37:.*]] = fir.shape_shift %[[VAL_34]], %[[VAL_35]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_38:.*]] = fir.embox %[[VAL_36]](%[[VAL_37]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_38]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_39:.*]] = fir.load %[[VAL_8]] : !fir.ref<index>
! CHECK:           %[[VAL_40:.*]] = fir.load %[[VAL_9]] : !fir.ref<index>
! CHECK:           %[[VAL_41:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_42:.*]] = fir.shape_shift %[[VAL_39]], %[[VAL_40]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_43:.*]] = fir.embox %[[VAL_41]](%[[VAL_42]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_43]] to %[[VAL_6]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_44:.*]] = arith.constant true
! CHECK:           %[[VAL_45:.*]] = fir.address_of(@_QQclX8c4f3bfd49fac4527014c85e4d2f3268) : !fir.ref<!fir.char<1,74>>
! CHECK:           %[[VAL_46:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_47:.*]] = fir.zero_bits !fir.ref<none>
! CHECK:           %[[VAL_48:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_49:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_50:.*]] = fir.convert %[[VAL_45]] : (!fir.ref<!fir.char<1,74>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_51:.*]] = fir.call @_FortranAMoveAlloc(%[[VAL_48]], %[[VAL_49]], %[[VAL_47]], %[[VAL_44]], %[[VAL_33]], %[[VAL_50]], %[[VAL_46]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_52:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_53:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_54:.*]]:3 = fir.box_dims %[[VAL_52]], %[[VAL_53]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_55:.*]] = fir.box_addr %[[VAL_52]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.store %[[VAL_55]] to %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_54]]#1 to %[[VAL_3]] : !fir.ref<index>
! CHECK:           fir.store %[[VAL_54]]#0 to %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_56:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_57:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_58:.*]]:3 = fir.box_dims %[[VAL_56]], %[[VAL_57]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_59:.*]] = fir.box_addr %[[VAL_56]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.store %[[VAL_59]] to %[[VAL_7]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_58]]#1 to %[[VAL_9]] : !fir.ref<index>
! CHECK:           fir.store %[[VAL_58]]#0 to %[[VAL_8]] : !fir.ref<index>
! CHECK:           %[[VAL_60:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<i32>) -> i64
! CHECK:           %[[VAL_61:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_62:.*]] = arith.cmpi ne, %[[VAL_60]], %[[VAL_61]] : i64
! CHECK:           fir.if %[[VAL_62]] {
! CHECK:             fir.store %[[VAL_51]] to %[[VAL_5]] : !fir.ref<i32>
! CHECK:           }
! CHECK:           %[[VAL_63:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_64:.*]] = fir.convert %[[VAL_63]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK:           %[[VAL_65:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_66:.*]] = arith.cmpi ne, %[[VAL_64]], %[[VAL_65]] : i64
! CHECK:           fir.if %[[VAL_66]] {
! CHECK:             %[[VAL_67:.*]] = arith.constant false
! CHECK:             %[[VAL_68:.*]] = fir.absent !fir.box<none>
! CHECK:             %[[VAL_69:.*]] = fir.address_of(@_QQclX8c4f3bfd49fac4527014c85e4d2f3268) : !fir.ref<!fir.char<1,74>>
! CHECK:             %[[VAL_70:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_71:.*]] = fir.load %[[VAL_8]] : !fir.ref<index>
! CHECK:             %[[VAL_72:.*]] = fir.load %[[VAL_9]] : !fir.ref<index>
! CHECK:             %[[VAL_73:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:             %[[VAL_74:.*]] = fir.shape_shift %[[VAL_71]], %[[VAL_72]] : (index, index) -> !fir.shapeshift<1>
! CHECK:             %[[VAL_75:.*]] = fir.embox %[[VAL_73]](%[[VAL_74]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:             fir.store %[[VAL_75]] to %[[VAL_6]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:             %[[VAL_76:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_77:.*]] = fir.convert %[[VAL_69]] : (!fir.ref<!fir.char<1,74>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_78:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_76]], %[[VAL_67]], %[[VAL_68]], %[[VAL_77]], %[[VAL_70]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:             %[[VAL_79:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:             %[[VAL_80:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_81:.*]]:3 = fir.box_dims %[[VAL_79]], %[[VAL_80]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_82:.*]] = fir.box_addr %[[VAL_79]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:             fir.store %[[VAL_82]] to %[[VAL_7]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:             fir.store %[[VAL_81]]#1 to %[[VAL_9]] : !fir.ref<index>
! CHECK:             fir.store %[[VAL_81]]#0 to %[[VAL_8]] : !fir.ref<index>
! CHECK:           }
! CHECK:           %[[VAL_83:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_84:.*]] = fir.convert %[[VAL_83]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK:           %[[VAL_85:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_86:.*]] = arith.cmpi ne, %[[VAL_84]], %[[VAL_85]] : i64
! CHECK:           fir.if %[[VAL_86]] {
! CHECK:             %[[VAL_87:.*]] = arith.constant false
! CHECK:             %[[VAL_88:.*]] = fir.absent !fir.box<none>
! CHECK:             %[[VAL_89:.*]] = fir.address_of(@_QQclX8c4f3bfd49fac4527014c85e4d2f3268) : !fir.ref<!fir.char<1,74>>
! CHECK:             %[[VAL_90:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_91:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
! CHECK:             %[[VAL_92:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:             %[[VAL_93:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:             %[[VAL_94:.*]] = fir.shape_shift %[[VAL_91]], %[[VAL_92]] : (index, index) -> !fir.shapeshift<1>
! CHECK:             %[[VAL_95:.*]] = fir.embox %[[VAL_93]](%[[VAL_94]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:             fir.store %[[VAL_95]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:             %[[VAL_96:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_97:.*]] = fir.convert %[[VAL_89]] : (!fir.ref<!fir.char<1,74>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_98:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_96]], %[[VAL_87]], %[[VAL_88]], %[[VAL_97]], %[[VAL_90]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:             %[[VAL_99:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:             %[[VAL_100:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_101:.*]]:3 = fir.box_dims %[[VAL_99]], %[[VAL_100]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_102:.*]] = fir.box_addr %[[VAL_99]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:             fir.store %[[VAL_102]] to %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:             fir.store %[[VAL_101]]#1 to %[[VAL_3]] : !fir.ref<index>
! CHECK:             fir.store %[[VAL_101]]#0 to %[[VAL_2]] : !fir.ref<index>
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine to_from_stat_errmsg
  integer, allocatable :: from(:), to(:)
  integer :: stat
  character :: errMsg*64
  allocate(from(20))
  call move_alloc(from, to, stat, errMsg)
end subroutine to_from_stat_errmsg
! CHECK-LABEL:   func.func @_QPto_from_stat_errmsg() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.char<1,64> {bindc_name = "errmsg", uniq_name = "_QFto_from_stat_errmsgEerrmsg"}
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "from", uniq_name = "_QFto_from_stat_errmsgEfrom"}
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.heap<!fir.array<?xi32>> {uniq_name = "_QFto_from_stat_errmsgEfrom.addr"}
! CHECK:           %[[VAL_3:.*]] = fir.alloca index {uniq_name = "_QFto_from_stat_errmsgEfrom.lb0"}
! CHECK:           %[[VAL_4:.*]] = fir.alloca index {uniq_name = "_QFto_from_stat_errmsgEfrom.ext0"}
! CHECK:           %[[VAL_5:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.store %[[VAL_5]] to %[[VAL_2]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_6:.*]] = fir.alloca i32 {bindc_name = "stat", uniq_name = "_QFto_from_stat_errmsgEstat"}
! CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "to", uniq_name = "_QFto_from_stat_errmsgEto"}
! CHECK:           %[[VAL_8:.*]] = fir.alloca !fir.heap<!fir.array<?xi32>> {uniq_name = "_QFto_from_stat_errmsgEto.addr"}
! CHECK:           %[[VAL_9:.*]] = fir.alloca index {uniq_name = "_QFto_from_stat_errmsgEto.lb0"}
! CHECK:           %[[VAL_10:.*]] = fir.alloca index {uniq_name = "_QFto_from_stat_errmsgEto.ext0"}
! CHECK:           %[[VAL_11:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.store %[[VAL_11]] to %[[VAL_8]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_12:.*]] = arith.constant false
! CHECK:           %[[VAL_13:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_14:.*]] = fir.address_of(@_QQclX8c4f3bfd49fac4527014c85e4d2f3268) : !fir.ref<!fir.char<1,74>>
! CHECK:           %[[VAL_15:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_4]] : !fir.ref<index>
! CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_19:.*]] = fir.shape_shift %[[VAL_16]], %[[VAL_17]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_20:.*]] = fir.embox %[[VAL_18]](%[[VAL_19]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_20]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_21:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_22:.*]] = arith.constant 20 : i32
! CHECK:           %[[VAL_23:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_21]] : (index) -> i64
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_22]] : (i32) -> i64
! CHECK:           fir.call @_FortranAAllocatableSetBounds(%[[VAL_24]], %[[VAL_23]], %[[VAL_25]], %[[VAL_26]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_14]] : (!fir.ref<!fir.char<1,74>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_29:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_27]], %[[VAL_12]], %[[VAL_13]], %[[VAL_28]], %[[VAL_15]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_30:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_31:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_32:.*]]:3 = fir.box_dims %[[VAL_30]], %[[VAL_31]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_33:.*]] = fir.box_addr %[[VAL_30]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.store %[[VAL_33]] to %[[VAL_2]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_32]]#1 to %[[VAL_4]] : !fir.ref<index>
! CHECK:           fir.store %[[VAL_32]]#0 to %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_34:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.char<1,64>>) -> !fir.box<!fir.char<1,64>>
! CHECK:           %[[VAL_35:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_36:.*]] = fir.load %[[VAL_4]] : !fir.ref<index>
! CHECK:           %[[VAL_37:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_38:.*]] = fir.shape_shift %[[VAL_35]], %[[VAL_36]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_39:.*]] = fir.embox %[[VAL_37]](%[[VAL_38]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_39]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_40:.*]] = fir.load %[[VAL_9]] : !fir.ref<index>
! CHECK:           %[[VAL_41:.*]] = fir.load %[[VAL_10]] : !fir.ref<index>
! CHECK:           %[[VAL_42:.*]] = fir.load %[[VAL_8]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_43:.*]] = fir.shape_shift %[[VAL_40]], %[[VAL_41]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_44:.*]] = fir.embox %[[VAL_42]](%[[VAL_43]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_44]] to %[[VAL_7]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_45:.*]] = arith.constant true
! CHECK:           %[[VAL_46:.*]] = fir.address_of(@_QQclX8c4f3bfd49fac4527014c85e4d2f3268) : !fir.ref<!fir.char<1,74>>
! CHECK:           %[[VAL_47:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_48:.*]] = fir.zero_bits !fir.ref<none>
! CHECK:           %[[VAL_49:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_50:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_51:.*]] = fir.convert %[[VAL_34]] : (!fir.box<!fir.char<1,64>>) -> !fir.box<none>
! CHECK:           %[[VAL_52:.*]] = fir.convert %[[VAL_46]] : (!fir.ref<!fir.char<1,74>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_53:.*]] = fir.call @_FortranAMoveAlloc(%[[VAL_49]], %[[VAL_50]], %[[VAL_48]], %[[VAL_45]], %[[VAL_51]], %[[VAL_52]], %[[VAL_47]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_54:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_55:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_56:.*]]:3 = fir.box_dims %[[VAL_54]], %[[VAL_55]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_57:.*]] = fir.box_addr %[[VAL_54]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.store %[[VAL_57]] to %[[VAL_2]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_56]]#1 to %[[VAL_4]] : !fir.ref<index>
! CHECK:           fir.store %[[VAL_56]]#0 to %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_58:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_59:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_60:.*]]:3 = fir.box_dims %[[VAL_58]], %[[VAL_59]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_61:.*]] = fir.box_addr %[[VAL_58]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.store %[[VAL_61]] to %[[VAL_8]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_60]]#1 to %[[VAL_10]] : !fir.ref<index>
! CHECK:           fir.store %[[VAL_60]]#0 to %[[VAL_9]] : !fir.ref<index>
! CHECK:           %[[VAL_62:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<i32>) -> i64
! CHECK:           %[[VAL_63:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_64:.*]] = arith.cmpi ne, %[[VAL_62]], %[[VAL_63]] : i64
! CHECK:           fir.if %[[VAL_64]] {
! CHECK:             fir.store %[[VAL_53]] to %[[VAL_6]] : !fir.ref<i32>
! CHECK:           }
! CHECK:           %[[VAL_65:.*]] = fir.load %[[VAL_8]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_66:.*]] = fir.convert %[[VAL_65]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK:           %[[VAL_67:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_68:.*]] = arith.cmpi ne, %[[VAL_66]], %[[VAL_67]] : i64
! CHECK:           fir.if %[[VAL_68]] {
! CHECK:             %[[VAL_69:.*]] = arith.constant false
! CHECK:             %[[VAL_70:.*]] = fir.absent !fir.box<none>
! CHECK:             %[[VAL_71:.*]] = fir.address_of(@_QQclX8c4f3bfd49fac4527014c85e4d2f3268) : !fir.ref<!fir.char<1,74>>
! CHECK:             %[[VAL_72:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_73:.*]] = fir.load %[[VAL_9]] : !fir.ref<index>
! CHECK:             %[[VAL_74:.*]] = fir.load %[[VAL_10]] : !fir.ref<index>
! CHECK:             %[[VAL_75:.*]] = fir.load %[[VAL_8]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:             %[[VAL_76:.*]] = fir.shape_shift %[[VAL_73]], %[[VAL_74]] : (index, index) -> !fir.shapeshift<1>
! CHECK:             %[[VAL_77:.*]] = fir.embox %[[VAL_75]](%[[VAL_76]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:             fir.store %[[VAL_77]] to %[[VAL_7]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:             %[[VAL_78:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_79:.*]] = fir.convert %[[VAL_71]] : (!fir.ref<!fir.char<1,74>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_80:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_78]], %[[VAL_69]], %[[VAL_70]], %[[VAL_79]], %[[VAL_72]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:             %[[VAL_81:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:             %[[VAL_82:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_83:.*]]:3 = fir.box_dims %[[VAL_81]], %[[VAL_82]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_84:.*]] = fir.box_addr %[[VAL_81]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:             fir.store %[[VAL_84]] to %[[VAL_8]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:             fir.store %[[VAL_83]]#1 to %[[VAL_10]] : !fir.ref<index>
! CHECK:             fir.store %[[VAL_83]]#0 to %[[VAL_9]] : !fir.ref<index>
! CHECK:           }
! CHECK:           %[[VAL_85:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_86:.*]] = fir.convert %[[VAL_85]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK:           %[[VAL_87:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_88:.*]] = arith.cmpi ne, %[[VAL_86]], %[[VAL_87]] : i64
! CHECK:           fir.if %[[VAL_88]] {
! CHECK:             %[[VAL_89:.*]] = arith.constant false
! CHECK:             %[[VAL_90:.*]] = fir.absent !fir.box<none>
! CHECK:             %[[VAL_91:.*]] = fir.address_of(@_QQclX8c4f3bfd49fac4527014c85e4d2f3268) : !fir.ref<!fir.char<1,74>>
! CHECK:             %[[VAL_92:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_93:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:             %[[VAL_94:.*]] = fir.load %[[VAL_4]] : !fir.ref<index>
! CHECK:             %[[VAL_95:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:             %[[VAL_96:.*]] = fir.shape_shift %[[VAL_93]], %[[VAL_94]] : (index, index) -> !fir.shapeshift<1>
! CHECK:             %[[VAL_97:.*]] = fir.embox %[[VAL_95]](%[[VAL_96]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:             fir.store %[[VAL_97]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:             %[[VAL_98:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_99:.*]] = fir.convert %[[VAL_91]] : (!fir.ref<!fir.char<1,74>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_100:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_98]], %[[VAL_89]], %[[VAL_90]], %[[VAL_99]], %[[VAL_92]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:             %[[VAL_101:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:             %[[VAL_102:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_103:.*]]:3 = fir.box_dims %[[VAL_101]], %[[VAL_102]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_104:.*]] = fir.box_addr %[[VAL_101]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:             fir.store %[[VAL_104]] to %[[VAL_2]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:             fir.store %[[VAL_103]]#1 to %[[VAL_4]] : !fir.ref<index>
! CHECK:             fir.store %[[VAL_103]]#0 to %[[VAL_3]] : !fir.ref<index>
! CHECK:           }
! CHECK:           return
! CHECK:         }
