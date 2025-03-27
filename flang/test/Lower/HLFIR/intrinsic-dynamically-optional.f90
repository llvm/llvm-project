! RUN: bbc --emit-hlfir %s -o - | FileCheck %s

! mask argument is dynamically optional, lowered as a box
integer function test_optional_as_box(x, mask)
  integer :: x(:)
  logical, optional :: mask(:)
  test_optional_as_box = iall(x, mask=mask)
end function
! CHECK-LABEL:   func.func @_QPtest_optional_as_box(
! CHECK-SAME:                                       %[[X_ARG:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "x"},
! CHECK-SAME:                                       %[[MASK_ARG:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "mask", fir.optional}) -> i32 {
! CHECK:           %[[MASK_VAR:.*]]:2 = hlfir.declare %[[MASK_ARG]]
! CHECK:           %[[RET_ALLOC:.*]] = fir.alloca i32 {bindc_name = "test_optional_as_box", uniq_name = "_QFtest_optional_as_boxEtest_optional_as_box"}
! CHECK:           %[[RET_VAR:.*]]:2 = hlfir.declare %[[RET_ALLOC]]
! CHECK:           %[[X_VAR:.*]]:2 = hlfir.declare %[[X_ARG]]
! CHECK:           %[[C0:.*]] = arith.constant 0 : index
! CHECK:           %[[SRC_LINE:.*]] = fir.address_of({{.*}}) : !fir.ref<!fir.char<
! CHECK:           %[[C7:.*]] = arith.constant 7 : i32
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[X_VAR]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[SRC_LINE]] : (!fir.ref<!fir.char<{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[C0]] : (index) -> i32
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[MASK_VAR]]#1 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
! CHECK:           %[[RES:.*]] = fir.call @_FortranAIAll4(%[[VAL_6]], %[[VAL_7]], %[[C7]], %[[VAL_8]], %[[VAL_9]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
! CHECK:           hlfir.assign %[[RES]] to %[[RET_VAR]]#0 : i32, !fir.ref<i32>
! CHECK:           %[[RET:.*]] = fir.load %[[RET_VAR]]#0 : !fir.ref<i32>
! CHECK:           return %[[RET]] : i32
! CHECK:         }

! mask argument is dynamically optional, lowered as a box
integer function test_optional_as_box2(x, mask)
  integer :: x(:)
  logical, allocatable :: mask(:)
  test_optional_as_box2 = iall(x, mask=mask)
end function
! CHECK-LABEL:   func.func @_QPtest_optional_as_box2(
! CHECK-SAME:                                        %[[X_ARG:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "x"},
! CHECK-SAME:                                        %[[MASK_ARG:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>> {fir.bindc_name = "mask"}) -> i32 {
! CHECK:           %[[MASK_VAR:.*]]:2 = hlfir.declare %[[MASK_ARG]]
! CHECK:           %[[RET_ALLOC:.*]] = fir.alloca i32 {bindc_name = "test_optional_as_box2", uniq_name = "_QFtest_optional_as_box2Etest_optional_as_box2"}
! CHECK:           %[[RET_VAR:.*]]:2 = hlfir.declare %[[RET_ALLOC]]
! CHECK:           %[[X_VAR:.*]]:2 = hlfir.declare %[[X_ARG]]
! CHECK:           %[[MASK_LD:.*]] = fir.load %[[MASK_VAR]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
! CHECK:           %[[MASK_ADDR:.*]] = fir.box_addr %[[MASK_LD]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>) -> !fir.heap<!fir.array<?x!fir.logical<4>>>
! CHECK:           %[[MASK_INT:.*]] = fir.convert %[[MASK_ADDR]] : (!fir.heap<!fir.array<?x!fir.logical<4>>>) -> i64
! CHECK:           %[[C0_I64:.*]] = arith.constant 0 : i64
! CHECK:           %[[MASK_PRESENT:.*]] = arith.cmpi ne, %[[MASK_INT]], %[[C0_I64]] : i64
! CHECK:           %[[MASK_LD2:.*]] = fir.load %[[MASK_VAR]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
! CHECK:           %[[C0:.*]] = arith.constant 0 : index
! CHECK:           %[[MASK_DIMS:.*]]:3 = fir.box_dims %[[MASK_LD2]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>, index) -> (index, index, index)
! CHECK:           %[[MASK_ADDR2:.*]] = fir.box_addr %[[MASK_LD2]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>) -> !fir.heap<!fir.array<?x!fir.logical<4>>>
! CHECK:           %[[MASK_SHAPE:.*]] = fir.shape_shift %[[MASK_DIMS]]#0, %[[MASK_DIMS]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[MASK_REBOX:.*]] = fir.embox %[[MASK_ADDR2]](%[[MASK_SHAPE]]) : (!fir.heap<!fir.array<?x!fir.logical<4>>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:           %[[ABSENT:.*]] = fir.absent !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:           %[[MASK_SEL:.*]] = arith.select %[[MASK_PRESENT]], %[[MASK_REBOX]], %[[ABSENT]] : !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:           %[[VAL_16:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_17:.*]] = fir.address_of({{.*}}) : !fir.ref<!fir.char<{{.*}}>>
! CHECK:           %[[VAL_18:.*]] = arith.constant 33 : i32
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[X_VAR]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_17]] : (!fir.ref<!fir.char<{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_16]] : (index) -> i32
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[MASK_SEL]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
! CHECK:           %[[RES:.*]] = fir.call @_FortranAIAll4(%[[VAL_19]], %[[VAL_20]], %[[VAL_18]], %[[VAL_21]], %[[VAL_22]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
! CHECK:           hlfir.assign %[[RES]] to %[[RET_VAR]]#0 : i32, !fir.ref<i32>
! CHECK:           %[[RET:.*]] = fir.load %[[RET_VAR]]#0 : !fir.ref<i32>
! CHECK:           return %[[RET]] : i32
! CHECK:         }

! imaginary component is dyamically optional, lowered as a value
complex function test_optional_as_value(real, imaginary)
  real :: real
  real, optional :: imaginary
  test_optional_as_value = cmplx(real, imaginary)
end function
! CHECK-LABEL:   func.func @_QPtest_optional_as_value(
! CHECK-SAME:                                         %[[REAL_ARG:.*]]: !fir.ref<f32> {fir.bindc_name = "real"},
! CHECK-SAME:                                         %[[IMAG_ARG:.*]]: !fir.ref<f32> {fir.bindc_name = "imaginary", fir.optional}) -> complex<f32> {
! CHECK:           %[[IMAG_VAR:.*]]:2 = hlfir.declare %[[IMAG_ARG]]
! CHECK:           %[[REAL_VAR:.*]]:2 = hlfir.declare %[[REAL_ARG]]
! CHECK:           %[[RET_ALLOC:.*]] = fir.alloca complex<f32> {bindc_name = "test_optional_as_value", uniq_name = "_QFtest_optional_as_valueEtest_optional_as_value"}
! CHECK:           %[[RET_VAR:.*]]:2 = hlfir.declare %[[RET_ALLOC]]
! CHECK:           %[[IS_PRESENT:.*]] = fir.is_present %[[IMAG_VAR]]#0 : (!fir.ref<f32>) -> i1
! CHECK:           %[[REAL_LD:.*]] = fir.load %[[REAL_VAR]]#0 : !fir.ref<f32>
! CHECK:           %[[IMAG_LD:.*]] = fir.if %[[IS_PRESENT]] -> (f32) {
! CHECK:             %[[IMAG_PRESENT:.*]] = fir.load %[[IMAG_VAR]]#0 : !fir.ref<f32>
! CHECK:             fir.result %[[IMAG_PRESENT]] : f32
! CHECK:           } else {
! CHECK:             %[[IMAG_ABSENT:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:             fir.result %[[IMAG_ABSENT]] : f32
! CHECK:           }
! CHECK:           %[[UNDEF:.*]] = fir.undefined complex<f32>
! CHECK:           %[[INS_REAL:.*]] = fir.insert_value %[[UNDEF]], %[[REAL_LD]], [0 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK:           %[[INS_IMAG:.*]] = fir.insert_value %[[INS_REAL]], %[[IMAG_LD:.*]], [1 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK:           hlfir.assign %[[INS_IMAG]] to %[[RET_VAR]]#0
! CHECK:           %[[RET:.*]] = fir.load %[[RET_VAR]]#0 : !fir.ref<complex<f32>>
! CHECK:           return %[[RET]] : complex<f32>
! CHECK:         }

! stat argument is dynamically optional, lowered as an address
subroutine test_optional_as_addr
  integer, allocatable :: from(:), to(:)
  integer, allocatable :: stat
  allocate(from(20))
  call move_alloc(from, to, stat)
  deallocate(to)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_optional_as_addr() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "from", uniq_name = "_QFtest_optional_as_addrEfrom"}
! CHECK:           %[[VAL_1:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_4:.*]] = fir.embox %[[VAL_1]](%[[VAL_3]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFtest_optional_as_addrEfrom"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "stat", uniq_name = "_QFtest_optional_as_addrEstat"}
! CHECK:           %[[VAL_7:.*]] = fir.zero_bits !fir.heap<i32>
! CHECK:           %[[VAL_8:.*]] = fir.embox %[[VAL_7]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK:           fir.store %[[VAL_8]] to %[[VAL_6]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_6]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFtest_optional_as_addrEstat"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
! CHECK:           %[[VAL_10:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "to", uniq_name = "_QFtest_optional_as_addrEto"}
! CHECK:           %[[VAL_11:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:           %[[VAL_12:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_13:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_14:.*]] = fir.embox %[[VAL_11]](%[[VAL_13]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_14]] to %[[VAL_10]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_10]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFtest_optional_as_addrEto"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:           %[[VAL_16:.*]] = arith.constant false
! CHECK:           %[[VAL_17:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_18:.*]] = fir.address_of(@_QQclXa514fea0665eb481f11db615a3b4888a) : !fir.ref<!fir.char<1,89>>
! CHECK:           %[[VAL_19:.*]] = arith.constant 102 : i32
! CHECK:           %[[VAL_20:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_21:.*]] = arith.constant 20 : i32
! CHECK:           %[[VAL_22:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_5]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_20]] : (index) -> i64
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_21]] : (i32) -> i64
! CHECK:           fir.call @_FortranAAllocatableSetBounds(%[[VAL_23]], %[[VAL_22]], %[[VAL_24]], %[[VAL_25]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_5]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_18]] : (!fir.ref<!fir.char<1,89>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_28:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_26]], %[[VAL_16]], %[[VAL_17]], %[[VAL_27]], %[[VAL_19]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_29:.*]] = fir.load %[[VAL_9]]#1 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           %[[VAL_30:.*]] = fir.box_addr %[[VAL_29]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:           %[[VAL_31:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_32:.*]] = arith.constant true
! CHECK:           %[[VAL_33:.*]] = fir.address_of(@_QQclXa514fea0665eb481f11db615a3b4888a) : !fir.ref<!fir.char<1,89>>
! CHECK:           %[[VAL_34:.*]] = arith.constant 103 : i32
! CHECK:           %[[VAL_35:.*]] = fir.zero_bits !fir.ref<none>
! CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_15]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_37:.*]] = fir.convert %[[VAL_5]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_33]] : (!fir.ref<!fir.char<1,89>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_39:.*]] = fir.call @_FortranAMoveAlloc(%[[VAL_36]], %[[VAL_37]], %[[VAL_35]], %[[VAL_32]], %[[VAL_31]], %[[VAL_38]], %[[VAL_34]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_40:.*]] = fir.convert %[[VAL_30]] : (!fir.heap<i32>) -> i64
! CHECK:           %[[VAL_41:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_42:.*]] = arith.cmpi ne, %[[VAL_40]], %[[VAL_41]] : i64
! CHECK:           fir.if %[[VAL_42]] {
! CHECK:             fir.store %[[VAL_39]] to %[[VAL_30]] : !fir.heap<i32>
! CHECK:           }
! CHECK:           %[[VAL_43:.*]] = arith.constant false
! CHECK:           %[[VAL_44:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_45:.*]] = fir.address_of(@_QQclXa514fea0665eb481f11db615a3b4888a) : !fir.ref<!fir.char<1,89>>
! CHECK:           %[[VAL_46:.*]] = arith.constant 104 : i32
! CHECK:           %[[VAL_47:.*]] = fir.convert %[[VAL_15]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_48:.*]] = fir.convert %[[VAL_45]] : (!fir.ref<!fir.char<1,89>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_49:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_47]], %[[VAL_43]], %[[VAL_44]], %[[VAL_48]], %[[VAL_46]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_50:.*]] = fir.load %[[VAL_15]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_51:.*]] = fir.box_addr %[[VAL_50]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           %[[VAL_52:.*]] = fir.convert %[[VAL_51]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK:           %[[VAL_53:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_54:.*]] = arith.cmpi ne, %[[VAL_52]], %[[VAL_53]] : i64
! CHECK:           fir.if %[[VAL_54]] {
! CHECK:             %[[VAL_55:.*]] = arith.constant false
! CHECK:             %[[VAL_56:.*]] = fir.absent !fir.box<none>
! CHECK:             %[[VAL_57:.*]] = fir.address_of(@_QQclXa514fea0665eb481f11db615a3b4888a) : !fir.ref<!fir.char<1,89>>
! CHECK:             %[[VAL_58:.*]] = arith.constant 99 : i32
! CHECK:             %[[VAL_59:.*]] = fir.convert %[[VAL_15]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_60:.*]] = fir.convert %[[VAL_57]] : (!fir.ref<!fir.char<1,89>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_61:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_59]], %[[VAL_55]], %[[VAL_56]], %[[VAL_60]], %[[VAL_58]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           }
! CHECK:           %[[VAL_62:.*]] = fir.load %[[VAL_9]]#1 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           %[[VAL_63:.*]] = fir.box_addr %[[VAL_62]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:           %[[VAL_64:.*]] = fir.convert %[[VAL_63]] : (!fir.heap<i32>) -> i64
! CHECK:           %[[VAL_65:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_66:.*]] = arith.cmpi ne, %[[VAL_64]], %[[VAL_65]] : i64
! CHECK:           fir.if %[[VAL_66]] {
! CHECK:             %[[VAL_67:.*]] = arith.constant false
! CHECK:             %[[VAL_68:.*]] = fir.absent !fir.box<none>
! CHECK:             %[[VAL_69:.*]] = fir.address_of(@_QQclXa514fea0665eb481f11db615a3b4888a) : !fir.ref<!fir.char<1,89>>
! CHECK:             %[[VAL_70:.*]] = arith.constant 99 : i32
! CHECK:             %[[VAL_71:.*]] = fir.convert %[[VAL_9]]#1 : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_72:.*]] = fir.convert %[[VAL_69]] : (!fir.ref<!fir.char<1,89>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_73:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_71]], %[[VAL_67]], %[[VAL_68]], %[[VAL_72]], %[[VAL_70]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           }
! CHECK:           %[[VAL_74:.*]] = fir.load %[[VAL_5]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_75:.*]] = fir.box_addr %[[VAL_74]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           %[[VAL_76:.*]] = fir.convert %[[VAL_75]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK:           %[[VAL_77:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_78:.*]] = arith.cmpi ne, %[[VAL_76]], %[[VAL_77]] : i64
! CHECK:           fir.if %[[VAL_78]] {
! CHECK:             %[[VAL_79:.*]] = arith.constant false
! CHECK:             %[[VAL_80:.*]] = fir.absent !fir.box<none>
! CHECK:             %[[VAL_81:.*]] = fir.address_of(@_QQclXa514fea0665eb481f11db615a3b4888a) : !fir.ref<!fir.char<1,89>>
! CHECK:             %[[VAL_82:.*]] = arith.constant 99 : i32
! CHECK:             %[[VAL_83:.*]] = fir.convert %[[VAL_5]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_84:.*]] = fir.convert %[[VAL_81]] : (!fir.ref<!fir.char<1,89>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_85:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_83]], %[[VAL_79]], %[[VAL_80]], %[[VAL_84]], %[[VAL_82]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           }
! CHECK:           return
! CHECK:         }

! imaginary component is dyamically optional, lowered as a value
! Test placement of the designator under isPresent check.
function test_elemental_optional_as_value(real, imaginary)
  real :: real(3)
  real, optional :: imaginary(3)
  complex :: test_elemental_optional_as_value(3)
  test_elemental_optional_as_value = cmplx(real, imaginary)
end function
! CHECK-LABEL:   func.func @_QPtest_elemental_optional_as_value(
! CHECK-SAME:                                                   %[[VAL_0:.*]]: !fir.ref<!fir.array<3xf32>> {fir.bindc_name = "real"},
! CHECK-SAME:                                                   %[[VAL_1:.*]]: !fir.ref<!fir.array<3xf32>> {fir.bindc_name = "imaginary", fir.optional}) -> !fir.array<3xcomplex<f32>> {
! CHECK:           %[[VAL_2:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_3]]) dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFtest_elemental_optional_as_valueEimaginary"} : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf32>>, !fir.ref<!fir.array<3xf32>>)
! CHECK:           %[[VAL_5:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_6]]) dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest_elemental_optional_as_valueEreal"} : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf32>>, !fir.ref<!fir.array<3xf32>>)
! CHECK:           %[[VAL_8:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_9:.*]] = fir.alloca !fir.array<3xcomplex<f32>> {bindc_name = "test_elemental_optional_as_value", uniq_name = "_QFtest_elemental_optional_as_valueEtest_elemental_optional_as_value"}
! CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_9]](%[[VAL_10]]) {uniq_name = "_QFtest_elemental_optional_as_valueEtest_elemental_optional_as_value"} : (!fir.ref<!fir.array<3xcomplex<f32>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xcomplex<f32>>>, !fir.ref<!fir.array<3xcomplex<f32>>>)
! CHECK:           %[[VAL_12:.*]] = fir.is_present %[[VAL_4]]#0 : (!fir.ref<!fir.array<3xf32>>) -> i1
! CHECK:           %[[VAL_13:.*]] = hlfir.elemental %[[VAL_6]] unordered : (!fir.shape<1>) -> !hlfir.expr<3xcomplex<f32>> {
! CHECK:           ^bb0(%[[VAL_14:.*]]: index):
! CHECK:             %[[VAL_15:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_14]])  : (!fir.ref<!fir.array<3xf32>>, index) -> !fir.ref<f32>
! CHECK:             %[[VAL_16:.*]] = fir.load %[[VAL_15]] : !fir.ref<f32>
! CHECK:             %[[VAL_17:.*]] = fir.if %[[VAL_12]] -> (f32) {
! CHECK:               %[[VAL_18:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_14]])  : (!fir.ref<!fir.array<3xf32>>, index) -> !fir.ref<f32>
! CHECK:               %[[VAL_19:.*]] = fir.load %[[VAL_18]] : !fir.ref<f32>
! CHECK:               fir.result %[[VAL_19]] : f32
! CHECK:             } else {
! CHECK:               %[[VAL_20:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:               fir.result %[[VAL_20]] : f32
! CHECK:             }
! CHECK:             %[[VAL_21:.*]] = fir.undefined complex<f32>
! CHECK:             %[[VAL_22:.*]] = fir.insert_value %[[VAL_21]], %[[VAL_16]], [0 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK:             %[[VAL_23:.*]] = fir.insert_value %[[VAL_22]], %[[VAL_17]], [1 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK:             hlfir.yield_element %[[VAL_23]] : complex<f32>
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_13]] to %[[VAL_11]]#0 : !hlfir.expr<3xcomplex<f32>>, !fir.ref<!fir.array<3xcomplex<f32>>>
! CHECK:           hlfir.destroy %[[VAL_13]] : !hlfir.expr<3xcomplex<f32>>
! CHECK:           %[[VAL_24:.*]] = fir.load %[[VAL_11]]#0 : !fir.ref<!fir.array<3xcomplex<f32>>>
! CHECK:           return %[[VAL_24]] : !fir.array<3xcomplex<f32>>
! CHECK:         }

! TODO: there seem to be no intrinsics with dynamically optional arguments lowered asInquired

