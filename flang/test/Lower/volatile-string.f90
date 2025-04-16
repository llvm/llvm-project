! RUN: bbc %s -o - | FileCheck %s
program p
character(3), volatile :: string = 'foo'
call bar(string)
contains
  subroutine bar(x)
    character(3), volatile :: x
    x = 'bar'
  end subroutine
end program

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "p"} {
! CHECK:           %[[VAL_0:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_1:.*]] = fir.address_of(@_QFEstring) : !fir.ref<!fir.char<1,3>>
! CHECK:           %[[VAL_2:.*]] = fir.volatile_cast %[[VAL_1]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<!fir.char<1,3>, volatile>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] typeparams %[[VAL_0]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEstring"} : (!fir.ref<!fir.char<1,3>, volatile>, index) -> (!fir.ref<!fir.char<1,3>, volatile>, !fir.ref<!fir.char<1,3>, volatile>)
! CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,3>, volatile>) -> !fir.ref<!fir.char<1,3>>
! CHECK:           %[[VAL_5:.*]] = fir.emboxchar %[[VAL_4]], %[[VAL_0]] : (!fir.ref<!fir.char<1,3>>, index) -> !fir.boxchar<1>
! CHECK:           fir.call @_QFPbar(%[[VAL_5]]) fastmath<contract> : (!fir.boxchar<1>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPbar(
! CHECK-SAME:                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "x"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
! CHECK:           %[[VAL_1:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,3>>
! CHECK:           %[[VAL_5:.*]] = fir.volatile_cast %[[VAL_4]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<!fir.char<1,3>, volatile>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] typeparams %[[VAL_1]] dummy_scope %[[VAL_2]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFbarEx"} : (!fir.ref<!fir.char<1,3>, volatile>, index, !fir.dscope) -> (!fir.ref<!fir.char<1,3>, volatile>, !fir.ref<!fir.char<1,3>, volatile>)
! CHECK:           %[[VAL_7:.*]] = fir.address_of(@_QQclX626172) : !fir.ref<!fir.char<1,3>>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] typeparams %[[VAL_1]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX626172"} : (!fir.ref<!fir.char<1,3>>, index) -> (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>)
! CHECK:           hlfir.assign %[[VAL_8]]#0 to %[[VAL_6]]#0 : !fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>, volatile>
! CHECK:           return
! CHECK:         }
