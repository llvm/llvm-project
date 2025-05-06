! RUN: bbc --strict-fir-volatile-verifier %s -o - | FileCheck %s
! Ensure member access of a volatile derived type is volatile.
  type t
    integer :: e(4)=2
  end type t
  type(t), volatile :: f
  call test (f%e(::2))
contains
  subroutine test(v)
    integer, asynchronous :: v(:)
  end subroutine
end
! CHECK-LABEL:   func.func @_QQmain() {
! CHECK:           %[[VAL_0:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_5:.*]] = fir.address_of(@_QFE.b.t.e) : !fir.ref<!fir.array<2x1x!fir.type<{{.+}}>>>
! CHECK:           %[[VAL_6:.*]] = fir.shape_shift %[[VAL_3]], %[[VAL_2]], %[[VAL_3]], %[[VAL_1]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_6]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFE.b.t.e"} :
! CHECK:           %[[VAL_8:.*]] = fir.address_of(@_QFE.n.e) : !fir.ref<!fir.char<1>>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] typeparams %[[VAL_1]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFE.n.e"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
! CHECK:           %[[VAL_10:.*]] = fir.address_of(@_QFE.di.t.e) : !fir.ref<!fir.array<4xi32>>
! CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_10]](%[[VAL_11]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFE.di.t.e"} : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xi32>>, !fir.ref<!fir.array<4xi32>>)
! CHECK:           %[[VAL_13:.*]] = fir.address_of(@_QFE.n.t) : !fir.ref<!fir.char<1>>
! CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_13]] typeparams %[[VAL_1]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFE.n.t"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
! CHECK:           %[[VAL_15:.*]] = fir.alloca !fir.type<_QFTt{e:!fir.array<4xi32>}> {bindc_name = "f", uniq_name = "_QFEf"}
! CHECK:           %[[VAL_16:.*]] = fir.volatile_cast %[[VAL_15]] : (!fir.ref<!fir.type<_QFTt{e:!fir.array<4xi32>}>>) -> !fir.ref<!fir.type<_QFTt{e:!fir.array<4xi32>}>, volatile>
! CHECK:           %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_16]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEf"} : (!fir.ref<!fir.type<_QFTt{e:!fir.array<4xi32>}>, volatile>) -> (!fir.ref<!fir.type<_QFTt{e:!fir.array<4xi32>}>, volatile>, !fir.ref<!fir.type<_QFTt{e:!fir.array<4xi32>}>, volatile>)
! CHECK:           %[[VAL_18:.*]] = fir.address_of(@_QQ_QFTt.DerivedInit) : !fir.ref<!fir.type<_QFTt{e:!fir.array<4xi32>}>>
! CHECK:           fir.copy %[[VAL_18]] to %[[VAL_17]]#0 no_overlap : !fir.ref<!fir.type<_QFTt{e:!fir.array<4xi32>}>>, !fir.ref<!fir.type<_QFTt{e:!fir.array<4xi32>}>, volatile>
! CHECK:           %[[VAL_20:.*]] = fir.shape_shift %[[VAL_3]], %[[VAL_1]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_21:.*]]:2 = hlfir.declare %{{.+}}(%[[VAL_20]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFE.c.t"} :
! CHECK:           %[[VAL_24:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_25:.*]] = hlfir.designate %[[VAL_17]]#0{"e"} <%[[VAL_11]]> (%[[VAL_1]]:%[[VAL_0]]:%[[VAL_2]])  shape %[[VAL_24]] : (!fir.ref<!fir.type<_QFTt{e:!fir.array<4xi32>}>, volatile>, !fir.shape<1>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>, volatile>
! CHECK:           %[[VAL_26:.*]] = fir.volatile_cast %[[VAL_25]] : (!fir.box<!fir.array<2xi32>, volatile>) -> !fir.box<!fir.array<2xi32>>
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (!fir.box<!fir.array<2xi32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           fir.call @_QFPtest(%[[VAL_27]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
! CHECK:           return
! CHECK:         }
! CHECK-LABEL:   func.func private @_QFPtest(
! CHECK-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xi32>> {fir.asynchronous, fir.bindc_name = "v"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<asynchronous>, uniq_name = "_QFFtestEv"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:           return
! CHECK:         }
