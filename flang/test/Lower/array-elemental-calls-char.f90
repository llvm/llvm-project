! Test lowering of elemental calls with character argument
! without the VALUE attribute.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

module char_elem

interface
elemental integer function elem(c)
  character(*), intent(in) :: c
end function

elemental integer function elem2(c, j)
  character(*), intent(in) :: c
  integer, intent(in) :: j
end function

end interface

contains

subroutine foo1(i, c)
  integer :: i(10)
  character(*) :: c(10)
  i = elem(c)
end subroutine
! CHECK-LABEL:   func.func @_QMchar_elemPfoo1(
! CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "i"},
! CHECK-SAME:                                 %[[VAL_1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1,?>>>
! CHECK:           %[[VAL_5:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_6]]) typeparams %[[VAL_3]]#1 dummy_scope %[[VAL_2]] {uniq_name = "_QMchar_elemFfoo1Ec"} : (!fir.ref<!fir.array<10x!fir.char<1,?>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<10x!fir.char<1,?>>>, !fir.ref<!fir.array<10x!fir.char<1,?>>>)
! CHECK:           %[[VAL_8:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_9]]) dummy_scope %[[VAL_2]] {uniq_name = "_QMchar_elemFfoo1Ei"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_11:.*]] = hlfir.elemental %[[VAL_6]] unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:           ^bb0(%[[VAL_12:.*]]: index):
! CHECK:             %[[VAL_13:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_12]])  typeparams %[[VAL_3]]#1 : (!fir.box<!fir.array<10x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:             %[[VAL_14:.*]] = fir.call @_QPelem(%[[VAL_13]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.boxchar<1>) -> i32
! CHECK:             hlfir.yield_element %[[VAL_14]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_11]] to %[[VAL_10]]#0 : !hlfir.expr<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:           hlfir.destroy %[[VAL_11]] : !hlfir.expr<10xi32>
! CHECK:           return
! CHECK:         }

subroutine foo1b(i, c)
  integer :: i(10)
  character(10) :: c(10)
  i = elem(c)
end subroutine
! CHECK-LABEL:   func.func @_QMchar_elemPfoo1b(
! CHECK-SAME:                                  %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "i"},
! CHECK-SAME:                                  %[[VAL_1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1,10>>>
! CHECK:           %[[VAL_5:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_6:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_7]]) typeparams %[[VAL_5]] dummy_scope %[[VAL_2]] {uniq_name = "_QMchar_elemFfoo1bEc"} : (!fir.ref<!fir.array<10x!fir.char<1,10>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.ref<!fir.array<10x!fir.char<1,10>>>, !fir.ref<!fir.array<10x!fir.char<1,10>>>)
! CHECK:           %[[VAL_9:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_10]]) dummy_scope %[[VAL_2]] {uniq_name = "_QMchar_elemFfoo1bEi"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_12:.*]] = hlfir.elemental %[[VAL_7]] unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:           ^bb0(%[[VAL_13:.*]]: index):
! CHECK:             %[[VAL_14:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_13]])  typeparams %[[VAL_5]] : (!fir.ref<!fir.array<10x!fir.char<1,10>>>, index, index) -> !fir.ref<!fir.char<1,10>>
! CHECK:             %[[VAL_15:.*]] = fir.emboxchar %[[VAL_14]], %[[VAL_5]] : (!fir.ref<!fir.char<1,10>>, index) -> !fir.boxchar<1>
! CHECK:             %[[VAL_16:.*]] = fir.call @_QPelem(%[[VAL_15]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.boxchar<1>) -> i32
! CHECK:             hlfir.yield_element %[[VAL_16]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_12]] to %[[VAL_11]]#0 : !hlfir.expr<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:           hlfir.destroy %[[VAL_12]] : !hlfir.expr<10xi32>
! CHECK:           return
! CHECK:         }

subroutine foo2(i, j, c)
  integer :: i(10), j(10)
  character(*) :: c
  i = elem2(c, j)
end subroutine
! CHECK-LABEL:   func.func @_QMchar_elemPfoo2(
! CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "i"},
! CHECK-SAME:                                 %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "j"},
! CHECK-SAME:                                 %[[VAL_2:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]]#0 typeparams %[[VAL_4]]#1 dummy_scope %[[VAL_3]] {uniq_name = "_QMchar_elemFfoo2Ec"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:           %[[VAL_6:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_7]]) dummy_scope %[[VAL_3]] {uniq_name = "_QMchar_elemFfoo2Ei"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_9:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_10]]) dummy_scope %[[VAL_3]] {uniq_name = "_QMchar_elemFfoo2Ej"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_12:.*]] = hlfir.elemental %[[VAL_10]] unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:           ^bb0(%[[VAL_13:.*]]: index):
! CHECK:             %[[VAL_14:.*]] = hlfir.designate %[[VAL_11]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_15:.*]] = fir.call @_QPelem2(%[[VAL_5]]#0, %[[VAL_14]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.boxchar<1>, !fir.ref<i32>) -> i32
! CHECK:             hlfir.yield_element %[[VAL_15]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_12]] to %[[VAL_8]]#0 : !hlfir.expr<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:           hlfir.destroy %[[VAL_12]] : !hlfir.expr<10xi32>
! CHECK:           return
! CHECK:         }

subroutine foo2b(i, j, c)
  integer :: i(10), j(10)
  character(10) :: c
  i = elem2(c, j)
end subroutine
! CHECK-LABEL:   func.func @_QMchar_elemPfoo2b(
! CHECK-SAME:                                  %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "i"},
! CHECK-SAME:                                  %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "j"},
! CHECK-SAME:                                  %[[VAL_2:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,10>>
! CHECK:           %[[VAL_6:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_5]] typeparams %[[VAL_6]] dummy_scope %[[VAL_3]] {uniq_name = "_QMchar_elemFfoo2bEc"} : (!fir.ref<!fir.char<1,10>>, index, !fir.dscope) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
! CHECK:           %[[VAL_8:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_9]]) dummy_scope %[[VAL_3]] {uniq_name = "_QMchar_elemFfoo2bEi"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_11:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_12:.*]] = fir.shape %[[VAL_11]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_12]]) dummy_scope %[[VAL_3]] {uniq_name = "_QMchar_elemFfoo2bEj"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_14:.*]] = hlfir.elemental %[[VAL_12]] unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:           ^bb0(%[[VAL_15:.*]]: index):
! CHECK:             %[[VAL_16:.*]] = fir.emboxchar %[[VAL_7]]#0, %[[VAL_6]] : (!fir.ref<!fir.char<1,10>>, index) -> !fir.boxchar<1>
! CHECK:             %[[VAL_17:.*]] = hlfir.designate %[[VAL_13]]#0 (%[[VAL_15]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_18:.*]] = fir.call @_QPelem2(%[[VAL_16]], %[[VAL_17]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.boxchar<1>, !fir.ref<i32>) -> i32
! CHECK:             hlfir.yield_element %[[VAL_18]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_14]] to %[[VAL_10]]#0 : !hlfir.expr<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:           hlfir.destroy %[[VAL_14]] : !hlfir.expr<10xi32>
! CHECK:           return
! CHECK:         }

subroutine foo3(i, j)
  integer :: i(10), j(10)
  i = elem(char(j))
end subroutine
! CHECK-LABEL:   func.func @_QMchar_elemPfoo3(
! CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "i"},
! CHECK-SAME:                                 %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "j"}) {
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.char<1>
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_5]]) dummy_scope %[[VAL_3]] {uniq_name = "_QMchar_elemFfoo3Ei"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_8]]) dummy_scope %[[VAL_3]] {uniq_name = "_QMchar_elemFfoo3Ej"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
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
! CHECK:             %[[VAL_29:.*]] = fir.call @_QPelem(%[[VAL_28]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.boxchar<1>) -> i32
! CHECK:             hlfir.end_associate %[[VAL_27]]#1, %[[VAL_27]]#2 : !fir.ref<!fir.char<1>>, i1
! CHECK:             hlfir.yield_element %[[VAL_29]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_24]] to %[[VAL_6]]#0 : !hlfir.expr<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:           hlfir.destroy %[[VAL_24]] : !hlfir.expr<10xi32>
! CHECK:           hlfir.destroy %[[VAL_16]] : !hlfir.expr<10x!fir.char<1>>
! CHECK:           hlfir.destroy %[[VAL_10]] : !hlfir.expr<10xi64>
! CHECK:           return
! CHECK:         }

subroutine foo4(i, j)
  integer :: i(10), j(10)
  i = elem2("hello", j)
end subroutine
! CHECK-LABEL:   func.func @_QMchar_elemPfoo4(
! CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "i"},
! CHECK-SAME:                                 %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "j"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_4]]) dummy_scope %[[VAL_2]] {uniq_name = "_QMchar_elemFfoo4Ei"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_6:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_7]]) dummy_scope %[[VAL_2]] {uniq_name = "_QMchar_elemFfoo4Ej"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_9:.*]] = fir.address_of(@_QQclX68656C6C6F) : !fir.ref<!fir.char<1,5>>
! CHECK:           %[[VAL_10:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_9]] typeparams %[[VAL_10]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX68656C6C6F"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! CHECK:           %[[VAL_12:.*]] = hlfir.elemental %[[VAL_7]] unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:           ^bb0(%[[VAL_13:.*]]: index):
! CHECK:             %[[VAL_14:.*]] = hlfir.as_expr %[[VAL_11]]#0 : (!fir.ref<!fir.char<1,5>>) -> !hlfir.expr<!fir.char<1,5>>
! CHECK:             %[[VAL_15:.*]]:3 = hlfir.associate %[[VAL_14]] typeparams %[[VAL_10]] {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>, i1)
! CHECK:             %[[VAL_16:.*]] = fir.emboxchar %[[VAL_15]]#0, %[[VAL_10]] : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
! CHECK:             %[[VAL_17:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_18:.*]] = fir.call @_QPelem2(%[[VAL_16]], %[[VAL_17]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.boxchar<1>, !fir.ref<i32>) -> i32
! CHECK:             hlfir.end_associate %[[VAL_15]]#1, %[[VAL_15]]#2 : !fir.ref<!fir.char<1,5>>, i1
! CHECK:             hlfir.yield_element %[[VAL_18]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_12]] to %[[VAL_5]]#0 : !hlfir.expr<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:           hlfir.destroy %[[VAL_12]] : !hlfir.expr<10xi32>
! CHECK:           return
! CHECK:         }

! Test character return for elemental functions.

elemental function elem_return_char(c)
 character(*), intent(in) :: c
 character(len(c)) :: elem_return_char
 elem_return_char = "ab" // c
end function

subroutine foo6(c)
  implicit none
  character(*) :: c(10)
  c = elem_return_char(c)
end subroutine
! CHECK-LABEL:   func.func @_QMchar_elemPfoo6(
! CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1,?>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_5]]) typeparams %[[VAL_2]]#1 dummy_scope %[[VAL_1]] {uniq_name = "_QMchar_elemFfoo6Ec"} : (!fir.ref<!fir.array<10x!fir.char<1,?>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<10x!fir.char<1,?>>>, !fir.ref<!fir.array<10x!fir.char<1,?>>>)
! CHECK:           %[[VAL_7:.*]] = fir.convert %c1_i64 : (i64) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] typeparams %[[VAL_2]]#1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMchar_elemFelem_return_charEc"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_2]]#1 : (index) -> i64
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> i32
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> i64
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:           %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_14:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[VAL_13]] : index
! CHECK:           %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[VAL_12]], %[[VAL_13]] : index
! CHECK:           %[[VAL_16:.*]] = hlfir.elemental %[[VAL_5]] typeparams %[[VAL_15]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1,?>> {
! CHECK:           ^bb0(%[[VAL_17:.*]]: index):
! CHECK:             %[[VAL_18:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_17]])  typeparams %[[VAL_2]]#1 : (!fir.box<!fir.array<10x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:             %[[VAL_28:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_15]] : index) {bindc_name = ".result"}
! CHECK:             %[[VAL_29:.*]] = fir.call @_QMchar_elemPelem_return_char(%[[VAL_28]], %[[VAL_15]], %[[VAL_18]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.ref<!fir.char<1,?>>, index, !fir.boxchar<1>) -> !fir.boxchar<1>
! CHECK:             %[[VAL_30:.*]]:2 = hlfir.declare %[[VAL_28]] typeparams %[[VAL_15]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:             %[[VAL_31:.*]] = arith.constant false
! CHECK:             %[[VAL_32:.*]] = hlfir.as_expr %[[VAL_30]]#0 move %[[VAL_31]] : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:             hlfir.yield_element %[[VAL_32]] : !hlfir.expr<!fir.char<1,?>>
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_16]] to %[[VAL_6]]#0 : !hlfir.expr<10x!fir.char<1,?>>, !fir.box<!fir.array<10x!fir.char<1,?>>>
! CHECK:           hlfir.destroy %[[VAL_16]] : !hlfir.expr<10x!fir.char<1,?>>
! CHECK:           return
! CHECK:         }

end module
