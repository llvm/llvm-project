! RUN: bbc -polymorphic-type -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL:   func.func @_QPtest1(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.class<none> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest1Ex"} : (!fir.class<none>) -> (!fir.class<none>, !fir.class<none>)
! CHECK:           fir.select_type %[[VAL_1]]#1 : !fir.class<none> [#fir.type_is<!fir.char<1,?>>, ^bb1, unit, ^bb2]
! CHECK:         ^bb1:
! CHECK:           %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]]#1 : (!fir.class<none>) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_3:.*]] = fir.box_elesize %[[VAL_1]]#1 : (!fir.class<none>) -> index
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_2]] typeparams %[[VAL_3]] {uniq_name = "_QFtest1Ex"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:           %[[VAL_5:.*]]:2 = fir.unboxchar %[[VAL_4]]#0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_6:.*]] = fir.embox %[[VAL_5]]#0 typeparams %[[VAL_3]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:           %[[VAL_7:.*]] = fir.rebox %[[VAL_6]] : (!fir.box<!fir.char<1,?>>) -> !fir.class<none>
! CHECK:           fir.call @_QPprint(%[[VAL_7]]) fastmath<contract> : (!fir.class<none>) -> ()
! CHECK:           cf.br ^bb3
! CHECK:         ^bb2:
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_1]]#1 {uniq_name = "_QFtest1Ex"} : (!fir.class<none>) -> (!fir.class<none>, !fir.class<none>)
! CHECK:           %[[VAL_9:.*]] = fir.address_of(@_QQcl.4641494C) : !fir.ref<!fir.char<1,4>>
! CHECK:           %[[VAL_10:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_9]] typeparams %[[VAL_10]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQcl.4641494C"} : (!fir.ref<!fir.char<1,4>>, index) -> (!fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>)
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]]#1 : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_10]] : (index) -> i64
! CHECK:           %[[VAL_14:.*]] = arith.constant false
! CHECK:           %[[VAL_15:.*]] = arith.constant false
! CHECK:           %[[VAL_16:.*]] = fir.call @_FortranAStopStatementText(%[[VAL_12]], %[[VAL_13]], %[[VAL_14]], %[[VAL_15]]) fastmath<contract> : (!fir.ref<i8>, i64, i1, i1) -> none
! CHECK:           fir.unreachable
! CHECK:         ^bb3:
! CHECK:           return
! CHECK:         }
subroutine test1(x)
  interface
     subroutine print(x)
       class(*) x
     end subroutine print
  end interface
  class(*) x
  select type(x)
  type is (Character(*))
     call print(x)
  class Default
     stop 'FAIL'
  end select
end subroutine test1

! CHECK-LABEL:   func.func @_QPtest2(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.class<!fir.array<10xnone>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest2Ex"} : (!fir.class<!fir.array<10xnone>>) -> (!fir.class<!fir.array<10xnone>>, !fir.class<!fir.array<10xnone>>)
! CHECK:           fir.select_type %[[VAL_1]]#1 : !fir.class<!fir.array<10xnone>> [#fir.type_is<!fir.char<1,?>>, ^bb1, unit, ^bb2]
! CHECK:         ^bb1:
! CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#1 : (!fir.class<!fir.array<10xnone>>) -> !fir.box<!fir.array<10x!fir.char<1,?>>>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFtest2Ex"} : (!fir.box<!fir.array<10x!fir.char<1,?>>>) -> (!fir.box<!fir.array<10x!fir.char<1,?>>>, !fir.box<!fir.array<10x!fir.char<1,?>>>)
! CHECK:           %[[VAL_4:.*]] = fir.box_elesize %[[VAL_3]]#1 : (!fir.box<!fir.array<10x!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_6:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_5]])  typeparams %[[VAL_4]] : (!fir.box<!fir.array<10x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:           %[[VAL_7:.*]]:2 = fir.unboxchar %[[VAL_6]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_8:.*]] = fir.embox %[[VAL_7]]#0 typeparams %[[VAL_4]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:           %[[VAL_9:.*]] = fir.rebox %[[VAL_8]] : (!fir.box<!fir.char<1,?>>) -> !fir.class<none>
! CHECK:           fir.call @_QPprint(%[[VAL_9]]) fastmath<contract> : (!fir.class<none>) -> ()
! CHECK:           cf.br ^bb3
! CHECK:         ^bb2:
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_1]]#1 {uniq_name = "_QFtest2Ex"} : (!fir.class<!fir.array<10xnone>>) -> (!fir.class<!fir.array<10xnone>>, !fir.class<!fir.array<10xnone>>)
! CHECK:           %[[VAL_11:.*]] = fir.address_of(@_QQcl.4641494C) : !fir.ref<!fir.char<1,4>>
! CHECK:           %[[VAL_12:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_11]] typeparams %[[VAL_12]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQcl.4641494C"} : (!fir.ref<!fir.char<1,4>>, index) -> (!fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>)
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_13]]#1 : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_12]] : (index) -> i64
! CHECK:           %[[VAL_16:.*]] = arith.constant false
! CHECK:           %[[VAL_17:.*]] = arith.constant false
! CHECK:           %[[VAL_18:.*]] = fir.call @_FortranAStopStatementText(%[[VAL_14]], %[[VAL_15]], %[[VAL_16]], %[[VAL_17]]) fastmath<contract> : (!fir.ref<i8>, i64, i1, i1) -> none
! CHECK:           fir.unreachable
! CHECK:         ^bb3:
! CHECK:           return
! CHECK:         }
subroutine test2(x)
  interface
     subroutine print(x)
       class(*) x
     end subroutine print
  end interface
  class(*) x(10)
  select type(x)
  type is (Character(*))
     call print(x(1))
  class Default
     stop 'FAIL'
  end select
end subroutine test2
