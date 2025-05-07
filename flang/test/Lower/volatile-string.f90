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

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "p"} {
! CHECK:           %[[VAL_0:.*]] = arith.constant 11 : i32
! CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_2:.*]] = arith.constant true
! CHECK:           %[[VAL_3:.*]] = arith.constant 10 : i32
! CHECK:           %[[VAL_4:.*]] = arith.constant 3 : i32
! CHECK:           %[[VAL_5:.*]] = arith.constant false
! CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_7:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_8:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:           %[[VAL_9:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,3>>>
! CHECK:           %[[VAL_10:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_12:.*]] = fir.alloca !fir.char<1,3> {bindc_name = "nonvolatile_string", uniq_name = "_QFEnonvolatile_string"}
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_12]] typeparams %[[VAL_7]] {uniq_name = "_QFEnonvolatile_string"} : (!fir.ref<!fir.char<1,3>>, index) -> (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>)
! CHECK:           %[[VAL_14:.*]] = fir.address_of(@_QFEstring) : !fir.ref<!fir.char<1,3>>
! CHECK:           %[[VAL_15:.*]] = fir.volatile_cast %[[VAL_14]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<!fir.char<1,3>, volatile>
! CHECK:           %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_15]] typeparams %[[VAL_7]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEstring"} : (!fir.ref<!fir.char<1,3>, volatile>, index) -> (!fir.ref<!fir.char<1,3>, volatile>, !fir.ref<!fir.char<1,3>, volatile>)
! CHECK:           %[[VAL_17:.*]] = fir.volatile_cast %[[VAL_16]]#0 : (!fir.ref<!fir.char<1,3>, volatile>) -> !fir.ref<!fir.char<1,3>>
! CHECK:           %[[VAL_18:.*]] = fir.emboxchar %[[VAL_17]], %[[VAL_7]] : (!fir.ref<!fir.char<1,3>>, index) -> !fir.boxchar<1>
! CHECK:           fir.call @_QFPassign_same_length(%[[VAL_18]]) fastmath<contract> : (!fir.boxchar<1>) -> ()
! CHECK:           fir.call @_QFPassign_different_length(%[[VAL_18]]) fastmath<contract> : (!fir.boxchar<1>) -> ()
! CHECK:           %[[VAL_19:.*]] = fir.address_of(@_QQclX6F) : !fir.ref<!fir.char<1>>
! CHECK:           %[[VAL_20:.*]]:2 = hlfir.declare %[[VAL_19]] typeparams %[[VAL_6]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX6F"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_17]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_7]] : (index) -> i64
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_20]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_6]] : (index) -> i64
! CHECK:           %[[VAL_25:.*]] = fir.call @_FortranAIndex1(%[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_5]]) fastmath<contract> : (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i64) -> i32
! CHECK:           hlfir.assign %[[VAL_26]] to %[[VAL_11]]#0 : i32, !fir.ref<i32>
! CHECK:           hlfir.assign %[[VAL_4]] to %[[VAL_11]]#0 : i32, !fir.ref<i32>
! CHECK:           %[[VAL_27:.*]] = fir.embox %[[VAL_16]]#0 : (!fir.ref<!fir.char<1,3>, volatile>) -> !fir.box<!fir.char<1,3>, volatile>
! CHECK:           %[[VAL_28:.*]] = fir.zero_bits !fir.heap<!fir.char<1,3>>
! CHECK:           %[[VAL_29:.*]] = fir.embox %[[VAL_28]] : (!fir.heap<!fir.char<1,3>>) -> !fir.box<!fir.heap<!fir.char<1,3>>>
! CHECK:           fir.store %[[VAL_29]] to %[[VAL_9]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,3>>>>
! CHECK:           %[[VAL_30:.*]] = fir.address_of(
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_9]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,3>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_32:.*]] = fir.volatile_cast %[[VAL_27]] : (!fir.box<!fir.char<1,3>, volatile>) -> !fir.box<!fir.char<1,3>>
! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (!fir.box<!fir.char<1,3>>) -> !fir.box<none>
! CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_30]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           fir.call @_FortranAAdjustl(%[[VAL_31]], %[[VAL_33]], %[[VAL_34]], %[[VAL_3]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
! CHECK:           %[[VAL_35:.*]] = fir.load %[[VAL_9]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,3>>>>
! CHECK:           %[[VAL_36:.*]] = fir.box_elesize %[[VAL_35]] : (!fir.box<!fir.heap<!fir.char<1,3>>>) -> index
! CHECK:           %[[VAL_37:.*]] = fir.box_addr %[[VAL_35]] : (!fir.box<!fir.heap<!fir.char<1,3>>>) -> !fir.heap<!fir.char<1,3>>
! CHECK:           %[[VAL_38:.*]]:2 = hlfir.declare %[[VAL_37]] typeparams %[[VAL_36]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.char<1,3>>, index) -> (!fir.heap<!fir.char<1,3>>, !fir.heap<!fir.char<1,3>>)
! CHECK:           %[[VAL_39:.*]] = hlfir.as_expr %[[VAL_38]]#0 move %[[VAL_2]] : (!fir.heap<!fir.char<1,3>>, i1) -> !hlfir.expr<!fir.char<1,3>>
! CHECK:           hlfir.assign %[[VAL_39]] to %[[VAL_16]]#0 : !hlfir.expr<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>, volatile>
! CHECK:           hlfir.destroy %[[VAL_39]] : !hlfir.expr<!fir.char<1,3>>
! CHECK:           %[[VAL_40:.*]] = fir.zero_bits !fir.heap<!fir.char<1,?>>
! CHECK:           %[[VAL_41:.*]] = fir.embox %[[VAL_40]] typeparams %[[VAL_1]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:           fir.store %[[VAL_41]] to %[[VAL_8]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_42:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           fir.call @_FortranATrim(%[[VAL_42]], %[[VAL_33]], %[[VAL_34]], %[[VAL_0]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
! CHECK:           %[[VAL_43:.*]] = fir.load %[[VAL_8]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_44:.*]] = fir.box_elesize %[[VAL_43]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_45:.*]] = fir.box_addr %[[VAL_43]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:           %[[VAL_46:.*]]:2 = hlfir.declare %[[VAL_45]] typeparams %[[VAL_44]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.heap<!fir.char<1,?>>)
! CHECK:           %[[VAL_47:.*]] = hlfir.as_expr %[[VAL_46]]#0 move %[[VAL_2]] : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:           hlfir.assign %[[VAL_47]] to %[[VAL_13]]#0 : !hlfir.expr<!fir.char<1,?>>, !fir.ref<!fir.char<1,3>>
! CHECK:           hlfir.destroy %[[VAL_47]] : !hlfir.expr<!fir.char<1,?>>
! CHECK:           hlfir.assign %[[VAL_16]]#0 to %[[VAL_13]]#0 : !fir.ref<!fir.char<1,3>, volatile>, !fir.ref<!fir.char<1,3>>
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
