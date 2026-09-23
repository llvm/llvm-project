! Test character substring lowering
! RUN: bbc %s -o - -emit-hlfir | FileCheck %s

! Test substring lower where the parent is a scalar-char-literal-constant
subroutine scalar_substring_embox(i, j)
  integer(8) :: i, j
  call bar("abcHello World!dfg"(i:j))
end subroutine scalar_substring_embox
! CHECK-LABEL:   func.func @_QPscalar_substring_embox(
! CHECK-SAME:                                         %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "i"},
! CHECK-SAME:                                         %[[VAL_1:.*]]: !fir.ref<i64> {fir.bindc_name = "j"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QFscalar_substring_emboxEi"} : (!fir.ref<i64>, !fir.dscope) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QFscalar_substring_emboxEj"} : (!fir.ref<i64>, !fir.dscope) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_5:.*]] = fir.address_of(@_QQclX61626348656C6C6F20576F726C6421646667) : !fir.ref<!fir.char<1,18>>
! CHECK:           %[[VAL_6:.*]] = arith.constant 18 : index
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_5]] typeparams %[[VAL_6]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = ".stringlit"} : (!fir.ref<!fir.char<1,18>>, index) -> (!fir.ref<!fir.char<1,18>>, !fir.ref<!fir.char<1,18>>)
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
! CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_13:.*]] = arith.subi %[[VAL_11]], %[[VAL_10]] : index
! CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_12]] : index
! CHECK:           %[[VAL_15:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_16:.*]] = arith.cmpi sgt, %[[VAL_14]], %[[VAL_15]] : index
! CHECK:           %[[VAL_17:.*]] = arith.select %[[VAL_16]], %[[VAL_14]], %[[VAL_15]] : index
! CHECK:           %[[VAL_18:.*]] = hlfir.designate %[[VAL_7]]#0  substr %[[VAL_10]], %[[VAL_11]]  typeparams %[[VAL_17]] : (!fir.ref<!fir.char<1,18>>, index, index, index) -> !fir.boxchar<1>
! CHECK:           %[[VAL_19:.*]] = hlfir.as_expr %[[VAL_18]] : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:           %[[VAL_20:.*]]:3 = hlfir.associate %[[VAL_19]] typeparams %[[VAL_17]] {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
! CHECK:           fir.call @_QPbar(%[[VAL_20]]#0) fastmath<contract> : (!fir.boxchar<1>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_20]]#1, %[[VAL_20]]#2 : !fir.ref<!fir.char<1,?>>, i1
! CHECK:           return
! CHECK:         }


subroutine array_substring_embox(arr)
  interface
    subroutine s(a)
     character(1) :: a(:)
    end subroutine s
  end interface

  character(7) arr(4)

  call s(arr(:)(5:5))
end subroutine array_substring_embox
! CHECK-LABEL:   func.func @_QParray_substring_embox(
! CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "arr"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<4x!fir.char<1,7>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 7 : index
! CHECK:           %[[VAL_5:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_6]]) typeparams %[[VAL_4]] dummy_scope %[[VAL_1]] arg {{[0-9]+}} {uniq_name = "_QFarray_substring_emboxEarr"} : (!fir.ref<!fir.array<4x!fir.char<1,7>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.ref<!fir.array<4x!fir.char<1,7>>>, !fir.ref<!fir.array<4x!fir.char<1,7>>>)
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_9:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_10:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_12:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_13:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_14:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_15:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_8]]:%[[VAL_5]]:%[[VAL_9]]) substr %[[VAL_12]], %[[VAL_13]]  shape %[[VAL_11]] typeparams %[[VAL_14]] : (!fir.ref<!fir.array<4x!fir.char<1,7>>>, index, index, index, index, index, !fir.shape<1>, index) -> !fir.box<!fir.array<4x!fir.char<1>>>
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (!fir.box<!fir.array<4x!fir.char<1>>>) -> !fir.box<!fir.array<?x!fir.char<1>>>
! CHECK:           fir.call @_QPs(%[[VAL_16]]) fastmath<contract> : (!fir.box<!fir.array<?x!fir.char<1>>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine substring_assignment(a,b)

  character(4) :: a, b
  a(1:2) = b(3:4)
end subroutine substring_assignment
! CHECK-LABEL:   func.func @_QPsubstring_assignment(
! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "a"},
! CHECK-SAME:                                       %[[VAL_1:.*]]: !fir.boxchar<1> {fir.bindc_name = "b"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,4>>
! CHECK:           %[[VAL_5:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_4]] typeparams %[[VAL_5]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QFsubstring_assignmentEa"} : (!fir.ref<!fir.char<1,4>>, index, !fir.dscope) -> (!fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>)
! CHECK:           %[[VAL_7:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,4>>
! CHECK:           %[[VAL_9:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_8]] typeparams %[[VAL_9]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QFsubstring_assignmentEb"} : (!fir.ref<!fir.char<1,4>>, index, !fir.dscope) -> (!fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>)
! CHECK:           %[[VAL_11:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_12:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_13:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_14:.*]] = hlfir.designate %[[VAL_10]]#0  substr %[[VAL_11]], %[[VAL_12]]  typeparams %[[VAL_13]] : (!fir.ref<!fir.char<1,4>>, index, index, index) -> !fir.ref<!fir.char<1,2>>
! CHECK:           %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_16:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_17:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_18:.*]] = hlfir.designate %[[VAL_6]]#0  substr %[[VAL_15]], %[[VAL_16]]  typeparams %[[VAL_17]] : (!fir.ref<!fir.char<1,4>>, index, index, index) -> !fir.ref<!fir.char<1,2>>
! CHECK:           hlfir.assign %[[VAL_14]] to %[[VAL_18]] : !fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>
! CHECK:           return
! CHECK:         }


subroutine array_substring_assignment(a)
  character(5) :: a(6)
  a(:)(3:5) = "BAD"
end subroutine array_substring_assignment
! CHECK-LABEL:   func.func @_QParray_substring_assignment(
! CHECK-SAME:                                             %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "a"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<6x!fir.char<1,5>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_5:.*]] = arith.constant 6 : index
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_6]]) typeparams %[[VAL_4]] dummy_scope %[[VAL_1]] arg {{[0-9]+}} {uniq_name = "_QFarray_substring_assignmentEa"} : (!fir.ref<!fir.array<6x!fir.char<1,5>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.ref<!fir.array<6x!fir.char<1,5>>>, !fir.ref<!fir.array<6x!fir.char<1,5>>>)
! CHECK:           %[[VAL_8:.*]] = fir.address_of(@_QQclX424144) : !fir.ref<!fir.char<1,3>>
! CHECK:           %[[VAL_9:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_8]] typeparams %[[VAL_9]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX424144"} : (!fir.ref<!fir.char<1,3>>, index) -> (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>)
! CHECK:           %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_13:.*]] = arith.constant 6 : index
! CHECK:           %[[VAL_14:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_15:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_16:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_17:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_18:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_11]]:%[[VAL_5]]:%[[VAL_12]]) substr %[[VAL_15]], %[[VAL_16]]  shape %[[VAL_14]] typeparams %[[VAL_17]] : (!fir.ref<!fir.array<6x!fir.char<1,5>>>, index, index, index, index, index, !fir.shape<1>, index) -> !fir.box<!fir.array<6x!fir.char<1,3>>>
! CHECK:           hlfir.assign %[[VAL_10]]#0 to %[[VAL_18]] : !fir.ref<!fir.char<1,3>>, !fir.box<!fir.array<6x!fir.char<1,3>>>
! CHECK:           return
! CHECK:         }


subroutine array_substring_assignment2(a)
  type t
     character(7) :: ch
  end type t
  type(t) :: a(8)
  a%ch(4:7) = "nice"
end subroutine array_substring_assignment2
! CHECK-LABEL:   func.func @_QParray_substring_assignment2(
! CHECK-SAME:                                              %[[VAL_0:.*]]: !fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment2Tt{ch:!fir.char<1,7>}>>> {fir.bindc_name = "a"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_8:.*]] = arith.constant 8 : index
! CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_9]]) dummy_scope %[[VAL_1]] arg {{[0-9]+}} {uniq_name = "_QFarray_substring_assignment2Ea"} : (!fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment2Tt{ch:!fir.char<1,7>}>>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment2Tt{ch:!fir.char<1,7>}>>>, !fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment2Tt{ch:!fir.char<1,7>}>>>)
! CHECK:           %[[VAL_18:.*]] = fir.address_of(@_QQclX6E696365) : !fir.ref<!fir.char<1,4>>
! CHECK:           %[[VAL_19:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_20:.*]]:2 = hlfir.declare %[[VAL_18]] typeparams %[[VAL_19]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX6E696365"} : (!fir.ref<!fir.char<1,4>>, index) -> (!fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>)
! CHECK:           %[[VAL_21:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_22:.*]] = arith.constant 7 : index
! CHECK:           %[[VAL_23:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_24:.*]] = hlfir.designate %[[VAL_10]]#0{"ch"}  substr %[[VAL_21]], %[[VAL_22]]  shape %[[VAL_9]] typeparams %[[VAL_23]] : (!fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment2Tt{ch:!fir.char<1,7>}>>>, index, index, !fir.shape<1>, index) -> !fir.box<!fir.array<8x!fir.char<1,4>>>
! CHECK:           hlfir.assign %[[VAL_20]]#0 to %[[VAL_24]] : !fir.ref<!fir.char<1,4>>, !fir.box<!fir.array<8x!fir.char<1,4>>>
! CHECK:           return
! CHECK:         }


subroutine array_substring_assignment3(a,b)
  type t
     character(7) :: ch
  end type t
  type(t) :: a(8), b(8)
  a%ch(4:7) = b%ch(2:5)
end subroutine array_substring_assignment3
! CHECK-LABEL:   func.func @_QParray_substring_assignment3(
! CHECK-SAME:                                              %[[VAL_0:.*]]: !fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment3Tt{ch:!fir.char<1,7>}>>> {fir.bindc_name = "a"},
! CHECK-SAME:                                              %[[VAL_1:.*]]: !fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment3Tt{ch:!fir.char<1,7>}>>> {fir.bindc_name = "b"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_9:.*]] = arith.constant 8 : index
! CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_10]]) dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QFarray_substring_assignment3Ea"} : (!fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment3Tt{ch:!fir.char<1,7>}>>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment3Tt{ch:!fir.char<1,7>}>>>, !fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment3Tt{ch:!fir.char<1,7>}>>>)
! CHECK:           %[[VAL_12:.*]] = arith.constant 8 : index
! CHECK:           %[[VAL_13:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_13]]) dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QFarray_substring_assignment3Eb"} : (!fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment3Tt{ch:!fir.char<1,7>}>>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment3Tt{ch:!fir.char<1,7>}>>>, !fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment3Tt{ch:!fir.char<1,7>}>>>)
! CHECK:           %[[VAL_22:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_23:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_24:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_25:.*]] = hlfir.designate %[[VAL_14]]#0{"ch"}  substr %[[VAL_22]], %[[VAL_23]]  shape %[[VAL_13]] typeparams %[[VAL_24]] : (!fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment3Tt{ch:!fir.char<1,7>}>>>, index, index, !fir.shape<1>, index) -> !fir.box<!fir.array<8x!fir.char<1,4>>>
! CHECK:           %[[VAL_26:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_27:.*]] = arith.constant 7 : index
! CHECK:           %[[VAL_28:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_29:.*]] = hlfir.designate %[[VAL_11]]#0{"ch"}  substr %[[VAL_26]], %[[VAL_27]]  shape %[[VAL_10]] typeparams %[[VAL_28]] : (!fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment3Tt{ch:!fir.char<1,7>}>>>, index, index, !fir.shape<1>, index) -> !fir.box<!fir.array<8x!fir.char<1,4>>>
! CHECK:           hlfir.assign %[[VAL_25]] to %[[VAL_29]] : !fir.box<!fir.array<8x!fir.char<1,4>>>, !fir.box<!fir.array<8x!fir.char<1,4>>>
! CHECK:           return
! CHECK:         }
