! RUN: bbc -emit-hlfir -fwrapv %s -o - | FileCheck %s

subroutine issue(c1, c2)

  character(4) :: c1(3)
  character(*) :: c2(3)
  c1 = c2
end subroutine
! CHECK-LABEL:   func.func @_QPissue(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "c1"},
! CHECK-SAME:                        %[[VAL_1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c2"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<3x!fir.char<1,4>>>
! CHECK:           %[[VAL_5:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_6:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_7]]) typeparams %[[VAL_5]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QFissueEc1"} : (!fir.ref<!fir.array<3x!fir.char<1,4>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.ref<!fir.array<3x!fir.char<1,4>>>, !fir.ref<!fir.array<3x!fir.char<1,4>>>)
! CHECK:           %[[VAL_9:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<3x!fir.char<1,?>>>
! CHECK:           %[[VAL_11:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_12:.*]] = fir.shape %[[VAL_11]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_10]](%[[VAL_12]]) typeparams %[[VAL_9]]#1 dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QFissueEc2"} : (!fir.ref<!fir.array<3x!fir.char<1,?>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<3x!fir.char<1,?>>>, !fir.ref<!fir.array<3x!fir.char<1,?>>>)
! CHECK:           hlfir.assign %[[VAL_13]]#0 to %[[VAL_8]]#0 : !fir.box<!fir.array<3x!fir.char<1,?>>>, !fir.ref<!fir.array<3x!fir.char<1,4>>>

program p
  character(4) :: c1(3)
  character(4) :: c2(3) = ["abcd", "    ", "    "]
  print *, c2
  call issue(c1, c2)
  print *, c1
  call charlit
end program p

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "P"} {
! CHECK:           %[[VAL_0:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_1:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.array<3x!fir.char<1,4>> {bindc_name = "c1", uniq_name = "_QFEc1"}
! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_3]]) typeparams %[[VAL_0]] {uniq_name = "_QFEc1"} : (!fir.ref<!fir.array<3x!fir.char<1,4>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<3x!fir.char<1,4>>>, !fir.ref<!fir.array<3x!fir.char<1,4>>>)
! CHECK:           %[[VAL_5:.*]] = fir.address_of(@_QFEc2) : !fir.ref<!fir.array<3x!fir.char<1,4>>>
! CHECK:           %[[VAL_6:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_7:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_8]]) typeparams %[[VAL_6]] {uniq_name = "_QFEc2"} : (!fir.ref<!fir.array<3x!fir.char<1,4>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<3x!fir.char<1,4>>>, !fir.ref<!fir.array<3x!fir.char<1,4>>>)
! CHECK:           %[[VAL_10:.*]] = arith.constant 6 : i32
! CHECK:           %[[VAL_14:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_10]], %{{.*}}, %{{.*}}) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_15:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_16:.*]] = fir.embox %[[VAL_9]]#0(%[[VAL_15]]) : (!fir.ref<!fir.array<3x!fir.char<1,4>>>, !fir.shape<1>) -> !fir.box<!fir.array<3x!fir.char<1,4>>>
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (!fir.box<!fir.array<3x!fir.char<1,4>>>) -> !fir.box<none>
! CHECK:           %[[VAL_18:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_14]], %[[VAL_17]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_19:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_14]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.array<3x!fir.char<1,4>>>) -> !fir.ref<!fir.char<1,4>>
! CHECK:           %[[VAL_21:.*]] = fir.emboxchar %[[VAL_20]], %[[VAL_0]] : (!fir.ref<!fir.char<1,4>>, index) -> !fir.boxchar<1>
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_9]]#0 : (!fir.ref<!fir.array<3x!fir.char<1,4>>>) -> !fir.ref<!fir.char<1,4>>
! CHECK:           %[[VAL_23:.*]] = fir.emboxchar %[[VAL_22]], %[[VAL_6]] : (!fir.ref<!fir.char<1,4>>, index) -> !fir.boxchar<1>
! CHECK:           fir.call @_QPissue(%[[VAL_21]], %[[VAL_23]]) fastmath<contract> : (!fir.boxchar<1>, !fir.boxchar<1>) -> ()
! CHECK:           %[[VAL_24:.*]] = arith.constant 6 : i32
! CHECK:           %[[VAL_28:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_24]], %{{.*}}, %{{.*}}) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_29:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_30:.*]] = fir.embox %[[VAL_4]]#0(%[[VAL_29]]) : (!fir.ref<!fir.array<3x!fir.char<1,4>>>, !fir.shape<1>) -> !fir.box<!fir.array<3x!fir.char<1,4>>>
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (!fir.box<!fir.array<3x!fir.char<1,4>>>) -> !fir.box<none>
! CHECK:           %[[VAL_32:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_28]], %[[VAL_31]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_33:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_28]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:           fir.call @_QPcharlit() fastmath<contract> : () -> ()

subroutine charlit
  print*, ['AA ', 'MM ', 'MM ', 'ZZ ']
  print*, ['AA ', 'MM ', 'MM ', 'ZZ ']
  print*, ['AA ', 'MM ', 'MM ', 'ZZ ']
end
! CHECK-LABEL:   func.func @_QPcharlit() {
! CHECK:           %[[VAL_0:.*]] = arith.constant 6 : i32
! CHECK:           %[[VAL_4:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_0]], %{{.*}}, %{{.*}}) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_5:.*]] = fir.address_of(@_QQro.4x3xc1.0) : !fir.ref<!fir.array<4x!fir.char<1,3>>>
! CHECK:           %[[VAL_6:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_7:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_8]]) typeparams %[[VAL_7]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.4x3xc1.0"} : (!fir.ref<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<4x!fir.char<1,3>>>, !fir.ref<!fir.array<4x!fir.char<1,3>>>)
! CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_11:.*]] = fir.embox %[[VAL_9]]#0(%[[VAL_10]]) : (!fir.ref<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>) -> !fir.box<!fir.array<4x!fir.char<1,3>>>
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.box<!fir.array<4x!fir.char<1,3>>>) -> !fir.box<none>
! CHECK:           %[[VAL_13:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_4]], %[[VAL_12]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_14:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_4]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:           %[[VAL_15:.*]] = arith.constant 6 : i32
! CHECK:           %[[VAL_19:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_15]], %{{.*}}, %{{.*}}) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_20:.*]] = fir.address_of(@_QQro.4x3xc1.0) : !fir.ref<!fir.array<4x!fir.char<1,3>>>
! CHECK:           %[[VAL_21:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_22:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_23:.*]] = fir.shape %[[VAL_21]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_24:.*]]:2 = hlfir.declare %[[VAL_20]](%[[VAL_23]]) typeparams %[[VAL_22]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.4x3xc1.0"} : (!fir.ref<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<4x!fir.char<1,3>>>, !fir.ref<!fir.array<4x!fir.char<1,3>>>)
! CHECK:           %[[VAL_25:.*]] = fir.shape %[[VAL_21]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_26:.*]] = fir.embox %[[VAL_24]]#0(%[[VAL_25]]) : (!fir.ref<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>) -> !fir.box<!fir.array<4x!fir.char<1,3>>>
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (!fir.box<!fir.array<4x!fir.char<1,3>>>) -> !fir.box<none>
! CHECK:           %[[VAL_28:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_19]], %[[VAL_27]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_29:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_19]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:           %[[VAL_30:.*]] = arith.constant 6 : i32
! CHECK:           %[[VAL_34:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_30]], %{{.*}}, %{{.*}}) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_35:.*]] = fir.address_of(@_QQro.4x3xc1.0) : !fir.ref<!fir.array<4x!fir.char<1,3>>>
! CHECK:           %[[VAL_36:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_37:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_38:.*]] = fir.shape %[[VAL_36]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_39:.*]]:2 = hlfir.declare %[[VAL_35]](%[[VAL_38]]) typeparams %[[VAL_37]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.4x3xc1.0"} : (!fir.ref<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<4x!fir.char<1,3>>>, !fir.ref<!fir.array<4x!fir.char<1,3>>>)
! CHECK:           %[[VAL_40:.*]] = fir.shape %[[VAL_36]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_41:.*]] = fir.embox %[[VAL_39]]#0(%[[VAL_40]]) : (!fir.ref<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>) -> !fir.box<!fir.array<4x!fir.char<1,3>>>
! CHECK:           %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (!fir.box<!fir.array<4x!fir.char<1,3>>>) -> !fir.box<none>
! CHECK:           %[[VAL_43:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_34]], %[[VAL_42]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_44:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_34]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:           return
! CHECK:         }

! CHECK: fir.global internal @_QQro.4x3xc1.0 constant : !fir.array<4x!fir.char<1,3>>
! CHECK: AA
! CHECK: MM
! CHECK: ZZ
! CHECK-NOT: fir.global internal @_QQro.4x3xc1
! CHECK-NOT: AA
! CHECK-NOT: MM
! CHECK-NOT: ZZ
