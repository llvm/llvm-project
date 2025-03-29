! RUN: bbc --use-desc-for-alloc=false -emit-fir -hlfir=false %s -o - | FileCheck %s

! Test lowering of allocatables using runtime for allocate/deallcoate statements.
subroutine fooscalar()
  ! Test lowering of local allocatable specification
  real, allocatable :: x
  ! Test allocation of local allocatables
  allocate(x)
  ! Test reading allocatable bounds and extents
  print *, x
  ! Test deallocation
  deallocate(x)
end subroutine
! CHECK-LABEL:   func.func @_QPfooscalar() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "x", uniq_name = "_QFfooscalarEx"}
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.heap<f32> {uniq_name = "_QFfooscalarEx.addr"}
! CHECK:           %[[VAL_2:.*]] = fir.zero_bits !fir.heap<f32>
! CHECK:           fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<!fir.heap<f32>>
! CHECK:           %[[VAL_3:.*]] = arith.constant false
! CHECK:           %[[VAL_4:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_5:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:           %[[VAL_6:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<f32>>
! CHECK:           %[[VAL_8:.*]] = fir.embox %[[VAL_7]] : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
! CHECK:           fir.store %[[VAL_8]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_11:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_9]], %[[VAL_3]], %[[VAL_4]], %[[VAL_10]], %[[VAL_6]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_12:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:           %[[VAL_13:.*]] = fir.box_addr %[[VAL_12]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
! CHECK:           fir.store %[[VAL_13]] to %[[VAL_1]] : !fir.ref<!fir.heap<f32>>
! CHECK:           %[[VAL_14:.*]] = arith.constant 6 : i32
! CHECK:           %[[VAL_15:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_17:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_18:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_14]], %[[VAL_16]], %[[VAL_17]]) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<f32>>
! CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.heap<f32>
! CHECK:           %[[VAL_21:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_18]], %[[VAL_20]]) fastmath<contract> : (!fir.ref<i8>, f32) -> i1
! CHECK:           %[[VAL_22:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_18]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:           %[[VAL_23:.*]] = arith.constant false
! CHECK:           %[[VAL_24:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_25:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:           %[[VAL_26:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_27:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<f32>>
! CHECK:           %[[VAL_28:.*]] = fir.embox %[[VAL_27]] : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
! CHECK:           fir.store %[[VAL_28]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_25]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_31:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_29]], %[[VAL_23]], %[[VAL_24]], %[[VAL_30]], %[[VAL_26]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_32:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:           %[[VAL_33:.*]] = fir.box_addr %[[VAL_32]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
! CHECK:           fir.store %[[VAL_33]] to %[[VAL_1]] : !fir.ref<!fir.heap<f32>>
! CHECK:           %[[VAL_34:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<f32>>
! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (!fir.heap<f32>) -> i64
! CHECK:           %[[VAL_36:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_37:.*]] = arith.cmpi ne, %[[VAL_35]], %[[VAL_36]] : i64
! CHECK:           fir.if %[[VAL_37]] {
! CHECK:             %[[VAL_38:.*]] = arith.constant false
! CHECK:             %[[VAL_39:.*]] = fir.absent !fir.box<none>
! CHECK:             %[[VAL_40:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:             %[[VAL_41:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_42:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<f32>>
! CHECK:             %[[VAL_43:.*]] = fir.embox %[[VAL_42]] : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
! CHECK:             fir.store %[[VAL_43]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:             %[[VAL_44:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_40]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_46:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_44]], %[[VAL_38]], %[[VAL_39]], %[[VAL_45]], %[[VAL_41]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:             %[[VAL_47:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:             %[[VAL_48:.*]] = fir.box_addr %[[VAL_47]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
! CHECK:             fir.store %[[VAL_48]] to %[[VAL_1]] : !fir.ref<!fir.heap<f32>>
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine foodim1()
  ! Test lowering of local allocatable specification
  real, allocatable :: x(:)
  ! Test allocation of local allocatables
  allocate(x(42:100))
  ! Test reading allocatable bounds and extents
  print *, x(42)
  deallocate(x)
end subroutine
! CHECK-LABEL:   func.func @_QPfoodim1() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "x", uniq_name = "_QFfoodim1Ex"}
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.heap<!fir.array<?xf32>> {uniq_name = "_QFfoodim1Ex.addr"}
! CHECK:           %[[VAL_2:.*]] = fir.alloca index {uniq_name = "_QFfoodim1Ex.lb0"}
! CHECK:           %[[VAL_3:.*]] = fir.alloca index {uniq_name = "_QFfoodim1Ex.ext0"}
! CHECK:           %[[VAL_4:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
! CHECK:           %[[VAL_5:.*]] = arith.constant false
! CHECK:           %[[VAL_6:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_7:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:           %[[VAL_8:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
! CHECK:           %[[VAL_12:.*]] = fir.shape_shift %[[VAL_9]], %[[VAL_10]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_13:.*]] = fir.embox %[[VAL_11]](%[[VAL_12]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:           fir.store %[[VAL_13]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_14:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_15:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_16:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_14]] : (i32) -> i64
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
! CHECK:           fir.call @_FortranAAllocatableSetBounds(%[[VAL_17]], %[[VAL_16]], %[[VAL_18]], %[[VAL_19]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_22:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_20]], %[[VAL_5]], %[[VAL_6]], %[[VAL_21]], %[[VAL_8]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_23:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_24:.*]] = arith.constant {{.*}} : index
! CHECK:           %[[VAL_25:.*]]:3 = fir.box_dims %[[VAL_23]], %[[VAL_24]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_26:.*]] = fir.box_addr %[[VAL_23]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:           fir.store %[[VAL_26]] to %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
! CHECK:           fir.store %[[VAL_25]]#1 to %[[VAL_3]] : !fir.ref<index>
! CHECK:           fir.store %[[VAL_25]]#0 to %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_27:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_28:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_30:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_31:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_27]], %[[VAL_29]], %[[VAL_30]]) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_32:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_33:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
! CHECK:           %[[VAL_34:.*]] = arith.constant {{.*}} : i64
! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_32]] : (index) -> i64
! CHECK:           %[[VAL_36:.*]] = arith.subi %[[VAL_34]], %[[VAL_35]] : i64
! CHECK:           %[[VAL_37:.*]] = fir.coordinate_of %[[VAL_33]], %[[VAL_36]] : (!fir.heap<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! CHECK:           %[[VAL_38:.*]] = fir.load %[[VAL_37]] : !fir.ref<f32>
! CHECK:           %[[VAL_39:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_31]], %[[VAL_38]]) fastmath<contract> : (!fir.ref<i8>, f32) -> i1
! CHECK:           %[[VAL_40:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_31]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:           %[[VAL_41:.*]] = arith.constant false
! CHECK:           %[[VAL_42:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_43:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:           %[[VAL_44:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_45:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_46:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_47:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
! CHECK:           %[[VAL_48:.*]] = fir.shape_shift %[[VAL_45]], %[[VAL_46]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_49:.*]] = fir.embox %[[VAL_47]](%[[VAL_48]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:           fir.store %[[VAL_49]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_50:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_51:.*]] = fir.convert %[[VAL_43]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_52:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_50]], %[[VAL_41]], %[[VAL_42]], %[[VAL_51]], %[[VAL_44]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_53:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_54:.*]] = arith.constant {{.*}} : index
! CHECK:           %[[VAL_55:.*]]:3 = fir.box_dims %[[VAL_53]], %[[VAL_54]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_56:.*]] = fir.box_addr %[[VAL_53]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:           fir.store %[[VAL_56]] to %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
! CHECK:           fir.store %[[VAL_55]]#1 to %[[VAL_3]] : !fir.ref<index>
! CHECK:           fir.store %[[VAL_55]]#0 to %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_57:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
! CHECK:           %[[VAL_58:.*]] = fir.convert %[[VAL_57]] : (!fir.heap<!fir.array<?xf32>>) -> i64
! CHECK:           %[[VAL_59:.*]] = arith.constant {{.*}} : i64
! CHECK:           %[[VAL_60:.*]] = arith.cmpi ne, %[[VAL_58]], %[[VAL_59]] : i64
! CHECK:           fir.if %[[VAL_60]] {
! CHECK:             %[[VAL_61:.*]] = arith.constant false
! CHECK:             %[[VAL_62:.*]] = fir.absent !fir.box<none>
! CHECK:             %[[VAL_63:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:             %[[VAL_64:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_65:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
! CHECK:             %[[VAL_66:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:             %[[VAL_67:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
! CHECK:             %[[VAL_68:.*]] = fir.shape_shift %[[VAL_65]], %[[VAL_66]] : (index, index) -> !fir.shapeshift<1>
! CHECK:             %[[VAL_69:.*]] = fir.embox %[[VAL_67]](%[[VAL_68]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:             fir.store %[[VAL_69]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:             %[[VAL_70:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_71:.*]] = fir.convert %[[VAL_63]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_72:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_70]], %[[VAL_61]], %[[VAL_62]], %[[VAL_71]], %[[VAL_64]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:             %[[VAL_73:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:             %[[VAL_74:.*]] = arith.constant {{.*}} : index
! CHECK:             %[[VAL_75:.*]]:3 = fir.box_dims %[[VAL_73]], %[[VAL_74]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_76:.*]] = fir.box_addr %[[VAL_73]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:             fir.store %[[VAL_76]] to %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
! CHECK:             fir.store %[[VAL_75]]#1 to %[[VAL_3]] : !fir.ref<index>
! CHECK:             fir.store %[[VAL_75]]#0 to %[[VAL_2]] : !fir.ref<index>
! CHECK:           }
! CHECK:           return
! CHECK:         }

! CHECK-LABEL: _QPfoodim2
subroutine foodim2()
  ! Test lowering of local allocatable specification
  real, allocatable :: x(:, :)
  ! CHECK-DAG: fir.alloca !fir.heap<!fir.array<?x?xf32>> {{{.*}}uniq_name = "_QFfoodim2Ex.addr"}
  ! CHECK-DAG: fir.alloca index {{{.*}}uniq_name = "_QFfoodim2Ex.lb0"}
  ! CHECK-DAG: fir.alloca index {{{.*}}uniq_name = "_QFfoodim2Ex.ext0"}
  ! CHECK-DAG: fir.alloca index {{{.*}}uniq_name = "_QFfoodim2Ex.lb1"}
  ! CHECK-DAG: fir.alloca index {{{.*}}uniq_name = "_QFfoodim2Ex.ext1"}
end subroutine

! test lowering of character allocatables. Focus is placed on the length handling
subroutine char_deferred(n)
  integer :: n
  character(:), allocatable :: c
  allocate(character(10):: c)
  deallocate(c)
  allocate(character(n):: c)
  call bar(c)
end subroutine
! CHECK-LABEL:   func.func @_QPchar_deferred(
! CHECK-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {bindc_name = "c", uniq_name = "_QFchar_deferredEc"}
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.heap<!fir.char<1,?>> {uniq_name = "_QFchar_deferredEc.addr"}
! CHECK:           %[[VAL_3:.*]] = fir.alloca index {uniq_name = "_QFchar_deferredEc.len"}
! CHECK:           %[[VAL_4:.*]] = fir.zero_bits !fir.heap<!fir.char<1,?>>
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_2]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           %[[VAL_5:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_6:.*]] = arith.constant false
! CHECK:           %[[VAL_7:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_8:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:           %[[VAL_9:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           %[[VAL_12:.*]] = fir.embox %[[VAL_11]] typeparams %[[VAL_10]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:           fir.store %[[VAL_12]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_5]] : (i32) -> i64
! CHECK:           %[[VAL_15:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_16:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_17:.*]] = arith.constant 0 : i32
! CHECK:           fir.call @_FortranAAllocatableInitCharacterForAllocate(%[[VAL_13]], %[[VAL_14]], %[[VAL_15]], %[[VAL_16]], %[[VAL_17]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i64, i32, i32, i32) -> ()
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_20:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_18]], %[[VAL_6]], %[[VAL_7]], %[[VAL_19]], %[[VAL_9]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_21:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_22:.*]] = fir.box_elesize %[[VAL_21]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_23:.*]] = fir.box_addr %[[VAL_21]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:           fir.store %[[VAL_23]] to %[[VAL_2]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           fir.store %[[VAL_22]] to %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_24:.*]] = arith.constant false
! CHECK:           %[[VAL_25:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_26:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:           %[[VAL_27:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_28:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_29:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           %[[VAL_30:.*]] = fir.embox %[[VAL_29]] typeparams %[[VAL_28]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:           fir.store %[[VAL_30]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_33:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_31]], %[[VAL_24]], %[[VAL_25]], %[[VAL_32]], %[[VAL_27]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_34:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_35:.*]] = fir.box_elesize %[[VAL_34]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_36:.*]] = fir.box_addr %[[VAL_34]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:           fir.store %[[VAL_36]] to %[[VAL_2]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           fir.store %[[VAL_35]] to %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_37:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[VAL_38:.*]] = arith.constant false
! CHECK:           %[[VAL_39:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_40:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:           %[[VAL_41:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_42:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_43:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           %[[VAL_44:.*]] = fir.embox %[[VAL_43]] typeparams %[[VAL_42]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:           fir.store %[[VAL_44]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_45:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_46:.*]] = fir.convert %[[VAL_37]] : (i32) -> i64
! CHECK:           %[[VAL_47:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_48:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_49:.*]] = arith.constant 0 : i32
! CHECK:           fir.call @_FortranAAllocatableInitCharacterForAllocate(%[[VAL_45]], %[[VAL_46]], %[[VAL_47]], %[[VAL_48]], %[[VAL_49]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i64, i32, i32, i32) -> ()
! CHECK:           %[[VAL_50:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_51:.*]] = fir.convert %[[VAL_40]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_52:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_50]], %[[VAL_38]], %[[VAL_39]], %[[VAL_51]], %[[VAL_41]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_53:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_54:.*]] = fir.box_elesize %[[VAL_53]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_55:.*]] = fir.box_addr %[[VAL_53]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:           fir.store %[[VAL_55]] to %[[VAL_2]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           fir.store %[[VAL_54]] to %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_56:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_57:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           %[[VAL_58:.*]] = fir.emboxchar %[[VAL_57]], %[[VAL_56]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           fir.call @_QPbar(%[[VAL_58]]) fastmath<contract> : (!fir.boxchar<1>) -> ()
! CHECK:           %[[VAL_59:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           %[[VAL_60:.*]] = fir.convert %[[VAL_59]] : (!fir.heap<!fir.char<1,?>>) -> i64
! CHECK:           %[[VAL_61:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_62:.*]] = arith.cmpi ne, %[[VAL_60]], %[[VAL_61]] : i64
! CHECK:           fir.if %[[VAL_62]] {
! CHECK:             %[[VAL_63:.*]] = arith.constant false
! CHECK:             %[[VAL_64:.*]] = fir.absent !fir.box<none>
! CHECK:             %[[VAL_65:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:             %[[VAL_66:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_67:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:             %[[VAL_68:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:             %[[VAL_69:.*]] = fir.embox %[[VAL_68]] typeparams %[[VAL_67]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:             fir.store %[[VAL_69]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:             %[[VAL_70:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_71:.*]] = fir.convert %[[VAL_65]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_72:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_70]], %[[VAL_63]], %[[VAL_64]], %[[VAL_71]], %[[VAL_66]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:             %[[VAL_73:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:             %[[VAL_74:.*]] = fir.box_elesize %[[VAL_73]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
! CHECK:             %[[VAL_75:.*]] = fir.box_addr %[[VAL_73]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:             fir.store %[[VAL_75]] to %[[VAL_2]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:             fir.store %[[VAL_74]] to %[[VAL_3]] : !fir.ref<index>
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine char_explicit_cst(n)
  integer :: n
  character(10), allocatable :: c
  allocate(c)
  deallocate(c)
  allocate(character(n):: c)
  deallocate(c)
  allocate(character(10):: c)
  call bar(c)
end subroutine
! CHECK-LABEL:   func.func @_QPchar_explicit_cst(
! CHECK-SAME:                                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,10>>> {bindc_name = "c", uniq_name = "_QFchar_explicit_cstEc"}
! CHECK:           %[[VAL_2:.*]] = arith.constant {{.*}} : index
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.heap<!fir.char<1,10>> {uniq_name = "_QFchar_explicit_cstEc.addr"}
! CHECK:           %[[VAL_4:.*]] = fir.zero_bits !fir.heap<!fir.char<1,10>>
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_3]] : !fir.ref<!fir.heap<!fir.char<1,10>>>
! CHECK:           %[[VAL_5:.*]] = arith.constant false
! CHECK:           %[[VAL_6:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_7:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:           %[[VAL_8:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.heap<!fir.char<1,10>>>
! CHECK:           %[[VAL_10:.*]] = fir.embox %[[VAL_9]] : (!fir.heap<!fir.char<1,10>>) -> !fir.box<!fir.heap<!fir.char<1,10>>>
! CHECK:           fir.store %[[VAL_10]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_13:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_11]], %[[VAL_5]], %[[VAL_6]], %[[VAL_12]], %[[VAL_8]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_14:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
! CHECK:           %[[VAL_15:.*]] = fir.box_addr %[[VAL_14]] : (!fir.box<!fir.heap<!fir.char<1,10>>>) -> !fir.heap<!fir.char<1,10>>
! CHECK:           fir.store %[[VAL_15]] to %[[VAL_3]] : !fir.ref<!fir.heap<!fir.char<1,10>>>
! CHECK:           %[[VAL_16:.*]] = arith.constant false
! CHECK:           %[[VAL_17:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_18:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:           %[[VAL_19:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.heap<!fir.char<1,10>>>
! CHECK:           %[[VAL_21:.*]] = fir.embox %[[VAL_20]] : (!fir.heap<!fir.char<1,10>>) -> !fir.box<!fir.heap<!fir.char<1,10>>>
! CHECK:           fir.store %[[VAL_21]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_18]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_24:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_22]], %[[VAL_16]], %[[VAL_17]], %[[VAL_23]], %[[VAL_19]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_25:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
! CHECK:           %[[VAL_26:.*]] = fir.box_addr %[[VAL_25]] : (!fir.box<!fir.heap<!fir.char<1,10>>>) -> !fir.heap<!fir.char<1,10>>
! CHECK:           fir.store %[[VAL_26]] to %[[VAL_3]] : !fir.ref<!fir.heap<!fir.char<1,10>>>
! CHECK:           %[[VAL_27:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[VAL_28:.*]] = arith.constant false
! CHECK:           %[[VAL_29:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_30:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:           %[[VAL_31:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_32:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.heap<!fir.char<1,10>>>
! CHECK:           %[[VAL_33:.*]] = fir.embox %[[VAL_32]] : (!fir.heap<!fir.char<1,10>>) -> !fir.box<!fir.heap<!fir.char<1,10>>>
! CHECK:           fir.store %[[VAL_33]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
! CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_27]] : (i32) -> i64
! CHECK:           %[[VAL_36:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_37:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_38:.*]] = arith.constant 0 : i32
! CHECK:           fir.call @_FortranAAllocatableInitCharacterForAllocate(%[[VAL_34]], %[[VAL_35]], %[[VAL_36]], %[[VAL_37]], %[[VAL_38]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i64, i32, i32, i32) -> ()
! CHECK:           %[[VAL_39:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_40:.*]] = fir.convert %[[VAL_30]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_41:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_39]], %[[VAL_28]], %[[VAL_29]], %[[VAL_40]], %[[VAL_31]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_42:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
! CHECK:           %[[VAL_43:.*]] = fir.box_addr %[[VAL_42]] : (!fir.box<!fir.heap<!fir.char<1,10>>>) -> !fir.heap<!fir.char<1,10>>
! CHECK:           fir.store %[[VAL_43]] to %[[VAL_3]] : !fir.ref<!fir.heap<!fir.char<1,10>>>
! CHECK:           %[[VAL_44:.*]] = arith.constant false
! CHECK:           %[[VAL_45:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_46:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:           %[[VAL_47:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_48:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.heap<!fir.char<1,10>>>
! CHECK:           %[[VAL_49:.*]] = fir.embox %[[VAL_48]] : (!fir.heap<!fir.char<1,10>>) -> !fir.box<!fir.heap<!fir.char<1,10>>>
! CHECK:           fir.store %[[VAL_49]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
! CHECK:           %[[VAL_50:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_51:.*]] = fir.convert %[[VAL_46]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_52:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_50]], %[[VAL_44]], %[[VAL_45]], %[[VAL_51]], %[[VAL_47]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_53:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
! CHECK:           %[[VAL_54:.*]] = fir.box_addr %[[VAL_53]] : (!fir.box<!fir.heap<!fir.char<1,10>>>) -> !fir.heap<!fir.char<1,10>>
! CHECK:           fir.store %[[VAL_54]] to %[[VAL_3]] : !fir.ref<!fir.heap<!fir.char<1,10>>>
! CHECK:           %[[VAL_55:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_56:.*]] = arith.constant false
! CHECK:           %[[VAL_57:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_58:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:           %[[VAL_59:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_60:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.heap<!fir.char<1,10>>>
! CHECK:           %[[VAL_61:.*]] = fir.embox %[[VAL_60]] : (!fir.heap<!fir.char<1,10>>) -> !fir.box<!fir.heap<!fir.char<1,10>>>
! CHECK:           fir.store %[[VAL_61]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
! CHECK:           %[[VAL_62:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_63:.*]] = fir.convert %[[VAL_55]] : (i32) -> i64
! CHECK:           %[[VAL_64:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_65:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_66:.*]] = arith.constant 0 : i32
! CHECK:           fir.call @_FortranAAllocatableInitCharacterForAllocate(%[[VAL_62]], %[[VAL_63]], %[[VAL_64]], %[[VAL_65]], %[[VAL_66]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i64, i32, i32, i32) -> ()
! CHECK:           %[[VAL_67:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_68:.*]] = fir.convert %[[VAL_58]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_69:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_67]], %[[VAL_56]], %[[VAL_57]], %[[VAL_68]], %[[VAL_59]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_70:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
! CHECK:           %[[VAL_71:.*]] = fir.box_addr %[[VAL_70]] : (!fir.box<!fir.heap<!fir.char<1,10>>>) -> !fir.heap<!fir.char<1,10>>
! CHECK:           fir.store %[[VAL_71]] to %[[VAL_3]] : !fir.ref<!fir.heap<!fir.char<1,10>>>
! CHECK:           %[[VAL_72:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.heap<!fir.char<1,10>>>
! CHECK:           %[[VAL_73:.*]] = fir.emboxchar %[[VAL_72]], %[[VAL_2]] : (!fir.heap<!fir.char<1,10>>, index) -> !fir.boxchar<1>
! CHECK:           fir.call @_QPbar(%[[VAL_73]]) fastmath<contract> : (!fir.boxchar<1>) -> ()
! CHECK:           %[[VAL_74:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.heap<!fir.char<1,10>>>
! CHECK:           %[[VAL_75:.*]] = fir.convert %[[VAL_74]] : (!fir.heap<!fir.char<1,10>>) -> i64
! CHECK:           %[[VAL_76:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_77:.*]] = arith.cmpi ne, %[[VAL_75]], %[[VAL_76]] : i64
! CHECK:           fir.if %[[VAL_77]] {
! CHECK:             %[[VAL_78:.*]] = arith.constant false
! CHECK:             %[[VAL_79:.*]] = fir.absent !fir.box<none>
! CHECK:             %[[VAL_80:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:             %[[VAL_81:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_82:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.heap<!fir.char<1,10>>>
! CHECK:             %[[VAL_83:.*]] = fir.embox %[[VAL_82]] : (!fir.heap<!fir.char<1,10>>) -> !fir.box<!fir.heap<!fir.char<1,10>>>
! CHECK:             fir.store %[[VAL_83]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
! CHECK:             %[[VAL_84:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_85:.*]] = fir.convert %[[VAL_80]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_86:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_84]], %[[VAL_78]], %[[VAL_79]], %[[VAL_85]], %[[VAL_81]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:             %[[VAL_87:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
! CHECK:             %[[VAL_88:.*]] = fir.box_addr %[[VAL_87]] : (!fir.box<!fir.heap<!fir.char<1,10>>>) -> !fir.heap<!fir.char<1,10>>
! CHECK:             fir.store %[[VAL_88]] to %[[VAL_3]] : !fir.ref<!fir.heap<!fir.char<1,10>>>
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine char_explicit_dyn(l1, l2)
  integer :: l1, l2
  character(l1), allocatable :: c
  allocate(c)
  deallocate(c)
  allocate(character(l2):: c)
  deallocate(c)
  allocate(character(10):: c)
  call bar(c)
end subroutine
! CHECK-LABEL:   func.func @_QPchar_explicit_dyn(
! CHECK-SAME:                                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i32> {fir.bindc_name = "l1"},
! CHECK-SAME:                                    %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i32> {fir.bindc_name = "l2"}) {
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {bindc_name = "c", uniq_name = "_QFchar_explicit_dynEc"}
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_5:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[VAL_4]] : i32
! CHECK:           %[[VAL_6:.*]] = arith.select %[[VAL_5]], %[[VAL_3]], %[[VAL_4]] : i32
! CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.heap<!fir.char<1,?>> {uniq_name = "_QFchar_explicit_dynEc.addr"}
! CHECK:           %[[VAL_8:.*]] = fir.zero_bits !fir.heap<!fir.char<1,?>>
! CHECK:           fir.store %[[VAL_8]] to %[[VAL_7]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           %[[VAL_9:.*]] = arith.constant false
! CHECK:           %[[VAL_10:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_11:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:           %[[VAL_12:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_13:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           %[[VAL_14:.*]] = fir.embox %[[VAL_13]] typeparams %[[VAL_6]] : (!fir.heap<!fir.char<1,?>>, i32) -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:           fir.store %[[VAL_14]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_11]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_17:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_15]], %[[VAL_9]], %[[VAL_10]], %[[VAL_16]], %[[VAL_12]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_19:.*]] = fir.box_addr %[[VAL_18]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:           fir.store %[[VAL_19]] to %[[VAL_7]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           %[[VAL_20:.*]] = arith.constant false
! CHECK:           %[[VAL_21:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_22:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:           %[[VAL_23:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_24:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           %[[VAL_25:.*]] = fir.embox %[[VAL_24]] typeparams %[[VAL_6]] : (!fir.heap<!fir.char<1,?>>, i32) -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:           fir.store %[[VAL_25]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_28:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_26]], %[[VAL_20]], %[[VAL_21]], %[[VAL_27]], %[[VAL_23]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_29:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_30:.*]] = fir.box_addr %[[VAL_29]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:           fir.store %[[VAL_30]] to %[[VAL_7]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           %[[VAL_31:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_32:.*]] = arith.constant false
! CHECK:           %[[VAL_33:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_34:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:           %[[VAL_35:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_36:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           %[[VAL_37:.*]] = fir.embox %[[VAL_36]] typeparams %[[VAL_6]] : (!fir.heap<!fir.char<1,?>>, i32) -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:           fir.store %[[VAL_37]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_39:.*]] = fir.convert %[[VAL_31]] : (i32) -> i64
! CHECK:           %[[VAL_40:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_41:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_42:.*]] = arith.constant 0 : i32
! CHECK:           fir.call @_FortranAAllocatableInitCharacterForAllocate(%[[VAL_38]], %[[VAL_39]], %[[VAL_40]], %[[VAL_41]], %[[VAL_42]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i64, i32, i32, i32) -> ()
! CHECK:           %[[VAL_43:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_44:.*]] = fir.convert %[[VAL_34]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_45:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_43]], %[[VAL_32]], %[[VAL_33]], %[[VAL_44]], %[[VAL_35]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_46:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_47:.*]] = fir.box_addr %[[VAL_46]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:           fir.store %[[VAL_47]] to %[[VAL_7]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           %[[VAL_48:.*]] = arith.constant false
! CHECK:           %[[VAL_49:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_50:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:           %[[VAL_51:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_52:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           %[[VAL_53:.*]] = fir.embox %[[VAL_52]] typeparams %[[VAL_6]] : (!fir.heap<!fir.char<1,?>>, i32) -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:           fir.store %[[VAL_53]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_54:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_55:.*]] = fir.convert %[[VAL_50]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_56:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_54]], %[[VAL_48]], %[[VAL_49]], %[[VAL_55]], %[[VAL_51]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_57:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_58:.*]] = fir.box_addr %[[VAL_57]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:           fir.store %[[VAL_58]] to %[[VAL_7]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           %[[VAL_59:.*]] = arith.constant 10 : i32
! CHECK:           %[[VAL_60:.*]] = arith.constant false
! CHECK:           %[[VAL_61:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_62:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:           %[[VAL_63:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_64:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           %[[VAL_65:.*]] = fir.embox %[[VAL_64]] typeparams %[[VAL_6]] : (!fir.heap<!fir.char<1,?>>, i32) -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:           fir.store %[[VAL_65]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_66:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_67:.*]] = fir.convert %[[VAL_59]] : (i32) -> i64
! CHECK:           %[[VAL_68:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_69:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_70:.*]] = arith.constant 0 : i32
! CHECK:           fir.call @_FortranAAllocatableInitCharacterForAllocate(%[[VAL_66]], %[[VAL_67]], %[[VAL_68]], %[[VAL_69]], %[[VAL_70]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i64, i32, i32, i32) -> ()
! CHECK:           %[[VAL_71:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_72:.*]] = fir.convert %[[VAL_62]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_73:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_71]], %[[VAL_60]], %[[VAL_61]], %[[VAL_72]], %[[VAL_63]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_74:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_75:.*]] = fir.box_addr %[[VAL_74]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:           fir.store %[[VAL_75]] to %[[VAL_7]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           %[[VAL_76:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           %[[VAL_77:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:           %[[VAL_78:.*]] = fir.emboxchar %[[VAL_76]], %[[VAL_77]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           fir.call @_QPbar(%[[VAL_78]]) fastmath<contract> : (!fir.boxchar<1>) -> ()
! CHECK:           %[[VAL_79:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           %[[VAL_80:.*]] = fir.convert %[[VAL_79]] : (!fir.heap<!fir.char<1,?>>) -> i64
! CHECK:           %[[VAL_81:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_82:.*]] = arith.cmpi ne, %[[VAL_80]], %[[VAL_81]] : i64
! CHECK:           fir.if %[[VAL_82]] {
! CHECK:             %[[VAL_83:.*]] = arith.constant false
! CHECK:             %[[VAL_84:.*]] = fir.absent !fir.box<none>
! CHECK:             %[[VAL_85:.*]] = fir.address_of(@_QQclX8b6dcfbb2269977a79a6d83c61c3b19e) : !fir.ref<!fir.char<1,65>>
! CHECK:             %[[VAL_86:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_87:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:             %[[VAL_88:.*]] = fir.embox %[[VAL_87]] typeparams %[[VAL_6]] : (!fir.heap<!fir.char<1,?>>, i32) -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:             fir.store %[[VAL_88]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:             %[[VAL_89:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_90:.*]] = fir.convert %[[VAL_85]] : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_91:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_89]], %[[VAL_83]], %[[VAL_84]], %[[VAL_90]], %[[VAL_86]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:             %[[VAL_92:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:             %[[VAL_93:.*]] = fir.box_addr %[[VAL_92]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:             fir.store %[[VAL_93]] to %[[VAL_7]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:           }
! CHECK:           return
! CHECK:         }

! CHECK-LABEL: _QPspecifiers(
subroutine specifiers
  allocatable jj1(:), jj2(:,:), jj3(:)
  ! CHECK: [[STAT:%[0-9]+]] = fir.alloca i32 {{{.*}}uniq_name = "_QFspecifiersEsss"}
  integer sss
  character*30 :: mmm = "None"
  ! CHECK: fir.call @_FortranAAllocatableSetBounds
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableAllocate
  ! CHECK: fir.store [[RESULT]] to [[STAT]]
  ! CHECK: fir.if %{{[0-9]+}} {
  ! CHECK: fir.call @_FortranAAllocatableSetBounds
  ! CHECK: fir.call @_FortranAAllocatableSetBounds
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableAllocate
  ! CHECK: fir.store [[RESULT]] to [[STAT]]
  ! CHECK: fir.if %{{[0-9]+}} {
  ! CHECK: fir.call @_FortranAAllocatableSetBounds
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableAllocate
  ! CHECK: fir.store [[RESULT]] to [[STAT]]
  ! CHECK-NOT: fir.if %{{[0-9]+}} {
  ! CHECK-COUNT-2: }
  ! CHECK-NOT: }
  allocate(jj1(3), jj2(3,3), jj3(3), stat=sss, errmsg=mmm)
  ! CHECK: fir.call @_FortranAAllocatableSetBounds
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableAllocate
  ! CHECK: fir.call @_FortranAAllocatableSetBounds
  ! CHECK: fir.call @_FortranAAllocatableSetBounds
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableAllocate
  ! CHECK: fir.call @_FortranAAllocatableSetBounds
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableAllocate
  allocate(jj1(3), jj2(3,3), jj3(3), stat=sss, errmsg=mmm)
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableDeallocate
  ! CHECK: fir.store [[RESULT]] to [[STAT]]
  ! CHECK: fir.if %{{[0-9]+}} {
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableDeallocate
  ! CHECK: fir.store [[RESULT]] to [[STAT]]
  ! CHECK: fir.if %{{[0-9]+}} {
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableDeallocate
  ! CHECK: fir.store [[RESULT]] to [[STAT]]
  ! CHECK-NOT: fir.if %{{[0-9]+}} {
  ! CHECK-COUNT-2: }
  ! CHECK-NOT: }
  deallocate(jj1, jj2, jj3, stat=sss, errmsg=mmm)
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableDeallocate
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableDeallocate
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableDeallocate
  deallocate(jj1, jj2, jj3, stat=sss, errmsg=mmm)
end subroutine specifiers
