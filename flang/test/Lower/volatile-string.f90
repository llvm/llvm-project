! RUN: bbc --strict-fir-volatile-verifier %s -o - | FileCheck %s
program p
  character(3), volatile :: string = 'foo'
  character(3)           :: nonvolatile_string
  integer                :: i
  call assign_same_length(string)
  call assign_different_length(string)
  i = index(string, 'o')
  i = len(string)
  string = adjustl(string)
  nonvolatile_string = trim(string)
  nonvolatile_string = string
contains
  subroutine assign_same_length(x)
    character(3), intent(inout), volatile :: x
    x = 'bar'
  end subroutine
  subroutine assign_different_length(string)
    character(3), intent(inout), volatile :: string
    string = 'bo'
  end subroutine
end program

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "P"} {
! CHECK:           %[[VAL_0:.*]] = arith.constant true
! CHECK:           %[[VAL_1:.*]] = arith.constant 10 : i32
! CHECK:           %[[VAL_2:.*]] = arith.constant 3 : i32
! CHECK:           %[[VAL_3:.*]] = arith.constant false
! CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_5:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,3>>>
! CHECK:           %[[VAL_7:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_8:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_10:.*]] = fir.alloca !fir.char<1,3> {bindc_name = "nonvolatile_string", uniq_name = "_QFEnonvolatile_string"}
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]] typeparams %[[VAL_5]] {uniq_name = "_QFEnonvolatile_string"} : (!fir.ref<!fir.char<1,3>>, index) -> (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>)
! CHECK:           %[[VAL_12:.*]] = fir.address_of(@_QFEstring) : !fir.ref<!fir.char<1,3>>
! CHECK:           %[[VAL_13:.*]] = fir.volatile_cast %[[VAL_12]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<!fir.char<1,3>, volatile>
! CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_13]] typeparams %[[VAL_5]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEstring"} : (!fir.ref<!fir.char<1,3>, volatile>, index) -> (!fir.ref<!fir.char<1,3>, volatile>, !fir.ref<!fir.char<1,3>, volatile>)
! CHECK:           %[[VAL_15:.*]] = fir.volatile_cast %[[VAL_14]]#0 : (!fir.ref<!fir.char<1,3>, volatile>) -> !fir.ref<!fir.char<1,3>>
! CHECK:           %[[VAL_16:.*]] = fir.emboxchar %[[VAL_15]], %[[VAL_5]] : (!fir.ref<!fir.char<1,3>>, index) -> !fir.boxchar<1>
! CHECK:           fir.call @_QFPassign_same_length(%[[VAL_16]]) fastmath<contract> : (!fir.boxchar<1>) -> ()
! CHECK:           fir.call @_QFPassign_different_length(%[[VAL_16]]) fastmath<contract> : (!fir.boxchar<1>) -> ()
! CHECK:           %[[VAL_17:.*]] = fir.address_of(@_QQclX6F) : !fir.ref<!fir.char<1>>
! CHECK:           %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_17]] typeparams %[[VAL_4]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX6F"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_15]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_5]] : (index) -> i64
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_18]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_4]] : (index) -> i64
! CHECK:           %[[VAL_23:.*]] = fir.call @_FortranAIndex1(%[[VAL_19]], %[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_3]]) fastmath<contract> : (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i64) -> i32
! CHECK:           hlfir.assign %[[VAL_24]] to %[[VAL_9]]#0 : i32, !fir.ref<i32>
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_9]]#0 : i32, !fir.ref<i32>
! CHECK:           %[[VAL_25:.*]] = fir.embox %[[VAL_14]]#0 : (!fir.ref<!fir.char<1,3>, volatile>) -> !fir.box<!fir.char<1,3>, volatile>
! CHECK:           %[[VAL_26:.*]] = fir.zero_bits !fir.heap<!fir.char<1,3>>
! CHECK:           %[[VAL_27:.*]] = fir.embox %[[VAL_26]] : (!fir.heap<!fir.char<1,3>>) -> !fir.box<!fir.heap<!fir.char<1,3>>>
! CHECK:           fir.store %[[VAL_27]] to %[[VAL_6]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,3>>>>
! CHECK:           %[[VAL_28:.*]] = fir.address_of(
! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,3>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_30:.*]] = fir.volatile_cast %[[VAL_25]] : (!fir.box<!fir.char<1,3>, volatile>) -> !fir.box<!fir.char<1,3>>
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (!fir.box<!fir.char<1,3>>) -> !fir.box<none>
! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_28]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           fir.call @_FortranAAdjustl(%[[VAL_29]], %[[VAL_31]], %[[VAL_32]], %[[VAL_1]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
! CHECK:           %[[VAL_33:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,3>>>>
! CHECK:           %[[VAL_34:.*]] = fir.box_elesize %[[VAL_33]] : (!fir.box<!fir.heap<!fir.char<1,3>>>) -> index
! CHECK:           %[[VAL_35:.*]] = fir.box_addr %[[VAL_33]] : (!fir.box<!fir.heap<!fir.char<1,3>>>) -> !fir.heap<!fir.char<1,3>>
! CHECK:           %[[VAL_36:.*]]:2 = hlfir.declare %[[VAL_35]] typeparams %[[VAL_34]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.char<1,3>>, index) -> (!fir.heap<!fir.char<1,3>>, !fir.heap<!fir.char<1,3>>)
! CHECK:           %[[VAL_37:.*]] = hlfir.as_expr %[[VAL_36]]#0 move %[[VAL_0]] : (!fir.heap<!fir.char<1,3>>, i1) -> !hlfir.expr<!fir.char<1,3>>
! CHECK:           hlfir.assign %[[VAL_37]] to %[[VAL_14]]#0 : !hlfir.expr<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>, volatile>
! CHECK:           hlfir.destroy %[[VAL_37]] : !hlfir.expr<!fir.char<1,3>>
! CHECK:           %[[VAL_38:.*]] = hlfir.char_trim %[[VAL_14]]#0 : (!fir.ref<!fir.char<1,3>, volatile>) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:           hlfir.assign %[[VAL_38]] to %[[VAL_11]]#0 : !hlfir.expr<!fir.char<1,?>>, !fir.ref<!fir.char<1,3>>
! CHECK:           hlfir.destroy %[[VAL_38]] : !hlfir.expr<!fir.char<1,?>>
! CHECK:           hlfir.assign %[[VAL_14]]#0 to %[[VAL_11]]#0 : !fir.ref<!fir.char<1,3>, volatile>, !fir.ref<!fir.char<1,3>>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPassign_same_length(
! CHECK-SAME:                                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "x"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
! CHECK:           %[[VAL_1:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,3>>
! CHECK:           %[[VAL_5:.*]] = fir.volatile_cast %[[VAL_4]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<!fir.char<1,3>, volatile>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] typeparams %[[VAL_1]] dummy_scope %[[VAL_2]] {fortran_attrs = #fir.var_attrs<intent_inout, volatile>, uniq_name = "_QFFassign_same_lengthEx"} : (!fir.ref<!fir.char<1,3>, volatile>, index, !fir.dscope) -> (!fir.ref<!fir.char<1,3>, volatile>, !fir.ref<!fir.char<1,3>, volatile>)
! CHECK:           %[[VAL_7:.*]] = fir.address_of(@_QQclX626172) : !fir.ref<!fir.char<1,3>>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] typeparams %[[VAL_1]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX626172"} : (!fir.ref<!fir.char<1,3>>, index) -> (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>)
! CHECK:           hlfir.assign %[[VAL_8]]#0 to %[[VAL_6]]#0 : !fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>, volatile>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPassign_different_length(
! CHECK-SAME:                                                   %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "string"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
! CHECK:           %[[VAL_1:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_2:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,3>>
! CHECK:           %[[VAL_6:.*]] = fir.volatile_cast %[[VAL_5]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<!fir.char<1,3>, volatile>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] typeparams %[[VAL_2]] dummy_scope %[[VAL_3]] {fortran_attrs = #fir.var_attrs<intent_inout, volatile>, uniq_name = "_QFFassign_different_lengthEstring"} : (!fir.ref<!fir.char<1,3>, volatile>, index, !fir.dscope) -> (!fir.ref<!fir.char<1,3>, volatile>, !fir.ref<!fir.char<1,3>, volatile>)
! CHECK:           %[[VAL_8:.*]] = fir.address_of(@_QQclX626F) : !fir.ref<!fir.char<1,2>>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] typeparams %[[VAL_1]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX626F"} : (!fir.ref<!fir.char<1,2>>, index) -> (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>)
! CHECK:           hlfir.assign %[[VAL_9]]#0 to %[[VAL_7]]#0 : !fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,3>, volatile>
! CHECK:           return
! CHECK:         }
