! Test lowering of elemental calls with character argument
! with the VALUE attribute.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s


module char_elem_byval

interface
elemental integer function elem(c, j)
  character(*), value :: c
  integer, intent(in) :: j
end function
end interface

contains
subroutine foo1(i, j, c)
  integer :: i(10), j(10)
  character(*) :: c(10)
  i = elem(c, j)
end subroutine
! CHECK-LABEL:   func.func @_QMchar_elem_byvalPfoo1(
! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "i"},
! CHECK-SAME:                                       %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "j"},
! CHECK-SAME:                                       %[[VAL_2:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1,?>>>
! CHECK:           %[[VAL_6:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_7]]) typeparams %[[VAL_4]]#1 dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QMchar_elem_byvalFfoo1Ec"} : (!fir.ref<!fir.array<10x!fir.char<1,?>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<10x!fir.char<1,?>>>, !fir.ref<!fir.array<10x!fir.char<1,?>>>)
! CHECK:           %[[VAL_9:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_10]]) dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QMchar_elem_byvalFfoo1Ei"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_12:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_13:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_13]]) dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QMchar_elem_byvalFfoo1Ej"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_15:.*]] = hlfir.elemental %[[VAL_7]] unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:           ^bb0(%[[VAL_16:.*]]: index):
! CHECK:             %[[VAL_17:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_16]])  typeparams %[[VAL_4]]#1 : (!fir.box<!fir.array<10x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:             %[[VAL_18:.*]] = hlfir.as_expr %[[VAL_17]] : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:             %[[VAL_19:.*]]:3 = hlfir.associate %[[VAL_18]] typeparams %[[VAL_4]]#1 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
! CHECK:             %[[VAL_20:.*]] = hlfir.designate %[[VAL_14]]#0 (%[[VAL_16]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_21:.*]] = fir.call @_QPelem(%[[VAL_19]]#0, %[[VAL_20]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.boxchar<1>, !fir.ref<i32>) -> i32
! CHECK:             hlfir.end_associate %[[VAL_19]]#1, %[[VAL_19]]#2 : !fir.ref<!fir.char<1,?>>, i1
! CHECK:             hlfir.yield_element %[[VAL_21]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_15]] to %[[VAL_11]]#0 : !hlfir.expr<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:           hlfir.destroy %[[VAL_15]] : !hlfir.expr<10xi32>
! CHECK:           return
! CHECK:         }

subroutine foo2(i, j, c)
  integer :: i(10), j(10)
  character(*) :: c
  i = elem(c, j)
end subroutine
! CHECK-LABEL:   func.func @_QMchar_elem_byvalPfoo2(
! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "i"},
! CHECK-SAME:                                       %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "j"},
! CHECK-SAME:                                       %[[VAL_2:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]]#0 typeparams %[[VAL_4]]#1 dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QMchar_elem_byvalFfoo2Ec"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:           %[[VAL_6:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_7]]) dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QMchar_elem_byvalFfoo2Ei"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_9:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_10]]) dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QMchar_elem_byvalFfoo2Ej"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_12:.*]] = hlfir.elemental %[[VAL_10]] unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:           ^bb0(%[[VAL_13:.*]]: index):
! CHECK:             %[[VAL_14:.*]] = hlfir.as_expr %[[VAL_5]]#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:             %[[VAL_15:.*]]:3 = hlfir.associate %[[VAL_14]] typeparams %[[VAL_4]]#1 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
! CHECK:             %[[VAL_16:.*]] = hlfir.designate %[[VAL_11]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_17:.*]] = fir.call @_QPelem(%[[VAL_15]]#0, %[[VAL_16]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.boxchar<1>, !fir.ref<i32>) -> i32
! CHECK:             hlfir.end_associate %[[VAL_15]]#1, %[[VAL_15]]#2 : !fir.ref<!fir.char<1,?>>, i1
! CHECK:             hlfir.yield_element %[[VAL_17]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_12]] to %[[VAL_8]]#0 : !hlfir.expr<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:           hlfir.destroy %[[VAL_12]] : !hlfir.expr<10xi32>
! CHECK:           return
! CHECK:         }

subroutine foo3(i, j)
  integer :: i(10), j(10)
  i = elem(char(j), j)
end subroutine
! CHECK-LABEL:   func.func @_QMchar_elem_byvalPfoo3(
! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "i"},
! CHECK-SAME:                                       %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "j"}) {
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.char<1>
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_5]]) dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QMchar_elem_byvalFfoo3Ei"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_8]]) dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QMchar_elem_byvalFfoo3Ej"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_10:.*]] = hlfir.elemental %[[VAL_8]] unordered : (!fir.shape<1>) -> !hlfir.expr<10xi64> {
! CHECK:           ^bb0(%[[VAL_11:.*]]: index):
! CHECK:             %[[VAL_12:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_11]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_13:.*]] = fir.load %[[VAL_12]] : !fir.ref<i32>
! CHECK:             %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> i64
! CHECK:             hlfir.yield_element %[[VAL_14]] : i64
! CHECK:           }
! CHECK:           %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_16:.*]] = hlfir.elemental %[[VAL_8]] typeparams %[[VAL_15]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1>> {
! CHECK:           ^bb0(%[[VAL_17:.*]]: index):
! CHECK:             %[[VAL_18:.*]] = hlfir.apply %[[VAL_10]], %[[VAL_17]] : (!hlfir.expr<10xi64>, index) -> i64
! CHECK:             %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i64) -> i8
! CHECK:             %[[VAL_20:.*]] = fir.undefined !fir.char<1>
! CHECK:             %[[VAL_21:.*]] = fir.insert_value %[[VAL_20]], %[[VAL_19]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:             fir.store %[[VAL_21]] to %[[VAL_2]] : !fir.ref<!fir.char<1>>
! CHECK:             %[[VAL_22:.*]] = arith.constant false
! CHECK:             %[[VAL_23:.*]] = hlfir.as_expr %[[VAL_2]] move %[[VAL_22]] : (!fir.ref<!fir.char<1>>, i1) -> !hlfir.expr<!fir.char<1>>
! CHECK:             hlfir.yield_element %[[VAL_23]] : !hlfir.expr<!fir.char<1>>
! CHECK:           }
! CHECK:           %[[VAL_24:.*]] = hlfir.elemental %[[VAL_8]] unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:           ^bb0(%[[VAL_25:.*]]: index):
! CHECK:             %[[VAL_26:.*]] = hlfir.apply %[[VAL_16]], %[[VAL_25]] typeparams %[[VAL_15]] : (!hlfir.expr<10x!fir.char<1>>, index, index) -> !hlfir.expr<!fir.char<1>>
! CHECK:             %[[VAL_27:.*]]:3 = hlfir.associate %[[VAL_26]] typeparams %[[VAL_15]] {adapt.valuebyref} : (!hlfir.expr<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>, i1)
! CHECK:             %[[VAL_28:.*]] = fir.emboxchar %[[VAL_27]]#0, %[[VAL_15]] : (!fir.ref<!fir.char<1>>, index) -> !fir.boxchar<1>
! CHECK:             %[[VAL_29:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_25]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_30:.*]] = fir.call @_QPelem(%[[VAL_28]], %[[VAL_29]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.boxchar<1>, !fir.ref<i32>) -> i32
! CHECK:             hlfir.end_associate %[[VAL_27]]#1, %[[VAL_27]]#2 : !fir.ref<!fir.char<1>>, i1
! CHECK:             hlfir.yield_element %[[VAL_30]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_24]] to %[[VAL_6]]#0 : !hlfir.expr<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:           hlfir.destroy %[[VAL_24]] : !hlfir.expr<10xi32>
! CHECK:           hlfir.destroy %[[VAL_16]] : !hlfir.expr<10x!fir.char<1>>
! CHECK:           hlfir.destroy %[[VAL_10]] : !hlfir.expr<10xi64>
! CHECK:           return
! CHECK:         }

subroutine foo4(i, j)
  integer :: i(10), j(10)
  i = elem(char(j(1)), j)
end subroutine
! CHECK-LABEL:   func.func @_QMchar_elem_byvalPfoo4(
! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "i"},
! CHECK-SAME:                                       %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "j"}) {
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.char<1>
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_5]]) dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QMchar_elem_byvalFfoo4Ei"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_8]]) dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QMchar_elem_byvalFfoo4Ej"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_11:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_10]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:           %[[VAL_12:.*]] = fir.load %[[VAL_11]] : !fir.ref<i32>
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> i64
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> i8
! CHECK:           %[[VAL_15:.*]] = fir.undefined !fir.char<1>
! CHECK:           %[[VAL_16:.*]] = fir.insert_value %[[VAL_15]], %[[VAL_14]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:           fir.store %[[VAL_16]] to %[[VAL_2]] : !fir.ref<!fir.char<1>>
! CHECK:           %[[VAL_17:.*]] = arith.constant false
! CHECK:           %[[VAL_18:.*]] = hlfir.as_expr %[[VAL_2]] move %[[VAL_17]] : (!fir.ref<!fir.char<1>>, i1) -> !hlfir.expr<!fir.char<1>>
! CHECK:           %[[VAL_19:.*]] = hlfir.elemental %[[VAL_8]] unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:           ^bb0(%[[VAL_20:.*]]: index):
! CHECK:             %[[VAL_21:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_22:.*]]:3 = hlfir.associate %[[VAL_18]] typeparams %[[VAL_21]] {adapt.valuebyref} : (!hlfir.expr<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>, i1)
! CHECK:             %[[VAL_23:.*]] = fir.emboxchar %[[VAL_22]]#0, %[[VAL_21]] : (!fir.ref<!fir.char<1>>, index) -> !fir.boxchar<1>
! CHECK:             %[[VAL_24:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_20]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_25:.*]] = fir.call @_QPelem(%[[VAL_23]], %[[VAL_24]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.boxchar<1>, !fir.ref<i32>) -> i32
! CHECK:             hlfir.end_associate %[[VAL_22]]#1, %[[VAL_22]]#2 : !fir.ref<!fir.char<1>>, i1
! CHECK:             hlfir.yield_element %[[VAL_25]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_19]] to %[[VAL_6]]#0 : !hlfir.expr<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:           hlfir.destroy %[[VAL_19]] : !hlfir.expr<10xi32>
! CHECK:           hlfir.destroy %[[VAL_18]] : !hlfir.expr<!fir.char<1>>
! CHECK:           return
! CHECK:         }

! Note: the copy of the constant is important because VALUE argument can be
! modified on the caller side.

subroutine foo5(i, j)
  integer :: i(10), j(10)
  i = elem("hello", j)
end subroutine
! CHECK-LABEL:   func.func @_QMchar_elem_byvalPfoo5(
! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "i"},
! CHECK-SAME:                                       %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "j"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_4]]) dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QMchar_elem_byvalFfoo5Ei"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_6:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_7]]) dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QMchar_elem_byvalFfoo5Ej"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_9:.*]] = fir.address_of(@_QQclX68656C6C6F) : !fir.ref<!fir.char<1,5>>
! CHECK:           %[[VAL_10:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_9]] typeparams %[[VAL_10]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX68656C6C6F"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! CHECK:           %[[VAL_12:.*]] = hlfir.elemental %[[VAL_7]] unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:           ^bb0(%[[VAL_13:.*]]: index):
! CHECK:             %[[VAL_14:.*]] = hlfir.as_expr %[[VAL_11]]#0 : (!fir.ref<!fir.char<1,5>>) -> !hlfir.expr<!fir.char<1,5>>
! CHECK:             %[[VAL_15:.*]]:3 = hlfir.associate %[[VAL_14]] typeparams %[[VAL_10]] {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>, i1)
! CHECK:             %[[VAL_16:.*]] = fir.emboxchar %[[VAL_15]]#0, %[[VAL_10]] : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
! CHECK:             %[[VAL_17:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_18:.*]] = fir.call @_QPelem(%[[VAL_16]], %[[VAL_17]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.boxchar<1>, !fir.ref<i32>) -> i32
! CHECK:             hlfir.end_associate %[[VAL_15]]#1, %[[VAL_15]]#2 : !fir.ref<!fir.char<1,5>>, i1
! CHECK:             hlfir.yield_element %[[VAL_18]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_12]] to %[[VAL_5]]#0 : !hlfir.expr<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:           hlfir.destroy %[[VAL_12]] : !hlfir.expr<10xi32>
! CHECK:           return
! CHECK:         }

end module
