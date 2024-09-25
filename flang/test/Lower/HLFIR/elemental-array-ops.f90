! Test lowering of elemental intrinsic operations with array arguments to HLFIR
! RUN: bbc -emit-hlfir -I nowhere -o - %s 2>&1 | FileCheck %s

subroutine binary(x, y)
  integer :: x(100), y(100)
  x = x+y
end subroutine
! CHECK-LABEL: func.func @_QPbinary(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_3:[^)]*]]) {{.*}}x
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_6:[^)]*]]) {{.*}}y
! CHECK:  %[[VAL_8:.*]] = hlfir.elemental %[[VAL_3]] unordered : (!fir.shape<1>) -> !hlfir.expr<100xi32> {
! CHECK:  ^bb0(%[[VAL_9:.*]]: index):
! CHECK:    %[[VAL_10:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_9]])  : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_11:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_9]])  : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_12:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
! CHECK:    %[[VAL_13:.*]] = fir.load %[[VAL_11]] : !fir.ref<i32>
! CHECK:    %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_13]] : i32
! CHECK:    hlfir.yield_element %[[VAL_14]] : i32
! CHECK:  }
! CHECK: hlfir.assign
! CHECK: hlfir.destroy %[[VAL_8]]

subroutine binary_with_scalar_and_array(x, y)
  integer :: x(100), y
  x = x+y
end subroutine
! CHECK-LABEL: func.func @_QPbinary_with_scalar_and_array(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_3:[^)]*]]) {{.*}}x
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}} {{.*}}y
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_7:.*]] = hlfir.elemental %[[VAL_3]] unordered : (!fir.shape<1>) -> !hlfir.expr<100xi32> {
! CHECK:  ^bb0(%[[VAL_8:.*]]: index):
! CHECK:    %[[VAL_9:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_8]])  : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_10:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
! CHECK:    %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_6]] : i32
! CHECK:    hlfir.yield_element %[[VAL_11]] : i32
! CHECK:  }
! CHECK: hlfir.assign
! CHECK: hlfir.destroy %[[VAL_7]]

subroutine char_binary(x, y)
  character(*) :: x(100), y(100)
  call test_char(x//y)
end subroutine
! CHECK-LABEL: func.func @_QPchar_binary(
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_5:.*]]) typeparams %[[VAL_2:.*]]#1 {{.*}}x
! CHECK:  %[[VAL_11:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_10:.*]]) typeparams %[[VAL_7:.*]]#1 {{.*}}y
! CHECK:  %[[VAL_12:.*]] = arith.addi %[[VAL_2]]#1, %[[VAL_7]]#1 : index
! CHECK:  %[[VAL_13:.*]] = hlfir.elemental %[[VAL_5]] typeparams %[[VAL_12]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<100x!fir.char<1,?>> {
! CHECK:  ^bb0(%[[VAL_14:.*]]: index):
! CHECK:    %[[VAL_15:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_14]])  typeparams %[[VAL_2]]#1 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:    %[[VAL_16:.*]] = hlfir.designate %[[VAL_11]]#0 (%[[VAL_14]])  typeparams %[[VAL_7]]#1 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:    %[[VAL_17:.*]] = hlfir.concat %[[VAL_15]], %[[VAL_16]] len %[[VAL_12]] : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:    hlfir.yield_element %[[VAL_17]] : !hlfir.expr<!fir.char<1,?>>
! CHECK:  }
! CHECK: fir.call
! CHECK: hlfir.destroy %[[VAL_13]]

subroutine unary(x, n)
  integer :: n
  logical :: x(n)
  x = .not.x
end subroutine
! CHECK-LABEL: func.func @_QPunary(
! CHECK:  %[[VAL_10:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_9:[^)]*]]) {{.*}}x
! CHECK:  %[[VAL_11:.*]] = hlfir.elemental %[[VAL_9]] unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
! CHECK:  ^bb0(%[[VAL_12:.*]]: index):
! CHECK:    %[[VAL_13:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_12]])  : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
! CHECK:    %[[VAL_14:.*]] = fir.load %[[VAL_13]] : !fir.ref<!fir.logical<4>>
! CHECK:    %[[VAL_15:.*]] = arith.constant true
! CHECK:    %[[VAL_16:.*]] = fir.convert %[[VAL_14]] : (!fir.logical<4>) -> i1
! CHECK:    %[[VAL_17:.*]] = arith.xori %[[VAL_16]], %[[VAL_15]] : i1
! CHECK:    %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i1) -> !fir.logical<4>
! CHECK:    hlfir.yield_element %[[VAL_18]] : !fir.logical<4>
! CHECK:  }
! CHECK: hlfir.assign
! CHECK: hlfir.destroy %[[VAL_11]]

subroutine char_unary(x)
  character(10) :: x(20)
  call test_char_2((x))
end subroutine
! CHECK-LABEL: func.func @_QPchar_unary(
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_5:.*]]) typeparams %[[VAL_2:[^ ]*]] {{.*}}x
! CHECK:  %[[VAL_7:.*]] = hlfir.elemental %[[VAL_5]] typeparams %[[VAL_2]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<20x!fir.char<1,?>> {
! CHECK:  ^bb0(%[[VAL_8:.*]]: index):
! CHECK:    %[[VAL_9:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_8]])  typeparams %[[VAL_2]] : (!fir.ref<!fir.array<20x!fir.char<1,10>>>, index, index) -> !fir.ref<!fir.char<1,10>>
! CHECK:    %[[VAL_10:.*]] = hlfir.as_expr %[[VAL_9]] : (!fir.ref<!fir.char<1,10>>) -> !hlfir.expr<!fir.char<1,10>>
! CHECK:    hlfir.yield_element %[[VAL_10]] : !hlfir.expr<!fir.char<1,10>>
! CHECK:  }
! CHECK: fir.call
! CHECK: hlfir.destroy %[[VAL_7]]

subroutine chained_elemental(x, y, z)
  integer :: x(100), y(100), z(100)
  x = x+y+z
end subroutine
! CHECK-LABEL: func.func @_QPchained_elemental(
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_4:[^)]*]]) {{.*}}x
! CHECK:  %[[VAL_8:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_7:[^)]*]]) {{.*}}y
! CHECK:  %[[VAL_11:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_10:[^)]*]]) {{.*}}z
! CHECK:  %[[VAL_12:.*]] = hlfir.elemental %[[VAL_4]] unordered : (!fir.shape<1>) -> !hlfir.expr<100xi32> {
! CHECK:  ^bb0(%[[VAL_13:.*]]: index):
! CHECK:    %[[VAL_14:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_15:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_16:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
! CHECK:    %[[VAL_17:.*]] = fir.load %[[VAL_15]] : !fir.ref<i32>
! CHECK:    %[[VAL_18:.*]] = arith.addi %[[VAL_16]], %[[VAL_17]] : i32
! CHECK:    hlfir.yield_element %[[VAL_18]] : i32
! CHECK:  }
! CHECK:  %[[VAL_19:.*]] = hlfir.elemental %[[VAL_4]] unordered : (!fir.shape<1>) -> !hlfir.expr<100xi32> {
! CHECK:  ^bb0(%[[VAL_20:.*]]: index):
! CHECK:    %[[VAL_21:.*]] = hlfir.apply %[[VAL_22:.*]], %[[VAL_20]] : (!hlfir.expr<100xi32>, index) -> i32
! CHECK:    %[[VAL_23:.*]] = hlfir.designate %[[VAL_11]]#0 (%[[VAL_20]])  : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
! CHECK:    %[[VAL_25:.*]] = arith.addi %[[VAL_21]], %[[VAL_24]] : i32
! CHECK:    hlfir.yield_element %[[VAL_25]] : i32
! CHECK:  }
! CHECK: hlfir.assign
! CHECK: hlfir.destroy %[[VAL_19]]
! CHECK: hlfir.destroy %[[VAL_12]]

subroutine lower_bounds(x)
  integer :: x(2:101)
  call test((x))
end subroutine
! CHECK-LABEL: func.func @_QPlower_bounds(
! CHECK:  %[[VAL_1:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_2:.*]] = arith.constant 100 : index
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_3:[^)]*]]) {{.*}}x
! CHECK:  %[[VAL_5:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_6:.*]] = hlfir.elemental %[[VAL_5]] unordered : (!fir.shape<1>) -> !hlfir.expr<100xi32> {
! CHECK:  ^bb0(%[[VAL_7:.*]]: index):
! CHECK:    %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:    %[[VAL_9:.*]] = arith.subi %[[VAL_1]], %[[VAL_8]] : index
! CHECK:    %[[VAL_10:.*]] = arith.addi %[[VAL_7]], %[[VAL_9]] : index
! CHECK:    %[[VAL_11:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_10]])  : (!fir.box<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_12:.*]] = fir.load %[[VAL_11]] : !fir.ref<i32>
! CHECK:    %[[VAL_13:.*]] = hlfir.no_reassoc %[[VAL_12]] : i32
! CHECK:    hlfir.yield_element %[[VAL_13]] : i32
! CHECK:  }
! CHECK: fir.call
! CHECK: hlfir.destroy %[[VAL_6]]

! Check that the character length for hlfir.associate is taken from
! hlfir.apply:
subroutine char_return(x,y)
  interface
     elemental character(3) function callee(x)
       character(3), intent(in) :: x
     end function callee
  end interface
  character(3), intent(in) :: x(:), y(:)
  logical, allocatable :: l(:)
  l = x==callee(y)
end subroutine char_return
! CHECK-LABEL:   func.func @_QPchar_return(
! CHECK-SAME:                              %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,3>>> {fir.bindc_name = "x"},
! CHECK-SAME:                              %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.char<1,3>>> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.char<1,3> {bindc_name = ".result"}
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>> {bindc_name = "l", uniq_name = "_QFchar_returnEl"}
! CHECK:           %[[VAL_4:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.logical<4>>>
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]] = fir.embox %[[VAL_4]](%[[VAL_6]]) : (!fir.heap<!fir.array<?x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_3]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFchar_returnEl"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>)
! CHECK:           %[[VAL_9:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_9]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFchar_returnEx"} : (!fir.box<!fir.array<?x!fir.char<1,3>>>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,3>>>, !fir.box<!fir.array<?x!fir.char<1,3>>>)
! CHECK:           %[[VAL_11:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_1]] typeparams %[[VAL_11]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFchar_returnEy"} : (!fir.box<!fir.array<?x!fir.char<1,3>>>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,3>>>, !fir.box<!fir.array<?x!fir.char<1,3>>>)
! CHECK:           %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_14:.*]]:3 = fir.box_dims %[[VAL_12]]#0, %[[VAL_13]] : (!fir.box<!fir.array<?x!fir.char<1,3>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_15:.*]] = fir.shape %[[VAL_14]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_16:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_17:.*]] = hlfir.elemental %[[VAL_15]] typeparams %[[VAL_16]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1,3>> {
! CHECK:           ^bb0(%[[VAL_18:.*]]: index):
! CHECK:             %[[VAL_19:.*]] = hlfir.designate %[[VAL_12]]#0 (%[[VAL_18]])  typeparams %[[VAL_11]] : (!fir.box<!fir.array<?x!fir.char<1,3>>>, index, index) -> !fir.ref<!fir.char<1,3>>
! CHECK:             %[[VAL_20:.*]] = fir.emboxchar %[[VAL_19]], %[[VAL_11]] : (!fir.ref<!fir.char<1,3>>, index) -> !fir.boxchar<1>
! CHECK:             %[[VAL_21:.*]] = arith.constant 3 : i64
! CHECK:             %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i64) -> index
! CHECK:             %[[VAL_23:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_24:.*]] = arith.cmpi sgt, %[[VAL_22]], %[[VAL_23]] : index
! CHECK:             %[[VAL_25:.*]] = arith.select %[[VAL_24]], %[[VAL_22]], %[[VAL_23]] : index
! CHECK:             %[[VAL_27:.*]] = fir.call @_QPcallee(%[[VAL_2]], %[[VAL_25]], %[[VAL_20]]) fastmath<contract> : (!fir.ref<!fir.char<1,3>>, index, !fir.boxchar<1>) -> !fir.boxchar<1>
! CHECK:             %[[VAL_28:.*]]:2 = hlfir.declare %[[VAL_2]] typeparams %[[VAL_25]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.char<1,3>>, index) -> (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>)
! CHECK:             %[[MustFree:.*]] = arith.constant false
! CHECK:             %[[ResultTemp:.*]] = hlfir.as_expr %[[VAL_28]]#0 move %[[MustFree]] : (!fir.ref<!fir.char<1,3>>, i1) -> !hlfir.expr<!fir.char<1,3>>
! CHECK:             hlfir.yield_element %[[ResultTemp]] : !hlfir.expr<!fir.char<1,3>>
! CHECK:           }
! CHECK:           %[[VAL_29:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_30:.*]]:3 = fir.box_dims %[[VAL_10]]#0, %[[VAL_29]] : (!fir.box<!fir.array<?x!fir.char<1,3>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_31:.*]] = fir.shape %[[VAL_30]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_32:.*]] = hlfir.elemental %[[VAL_31]] unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
! CHECK:           ^bb0(%[[VAL_33:.*]]: index):
! CHECK:             %[[VAL_34:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_33]])  typeparams %[[VAL_9]] : (!fir.box<!fir.array<?x!fir.char<1,3>>>, index, index) -> !fir.ref<!fir.char<1,3>>
! CHECK:             %[[VAL_35:.*]] = hlfir.apply %[[VAL_36:.*]], %[[VAL_33]] typeparams %[[VAL_16]] : (!hlfir.expr<?x!fir.char<1,3>>, index, index) -> !hlfir.expr<!fir.char<1,3>>
! CHECK:             %[[VAL_37:.*]]:3 = hlfir.associate %[[VAL_35]] typeparams %[[VAL_16]] {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,3>>, index) -> (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>, i1)
! CHECK:             %[[VAL_38:.*]] = fir.convert %[[VAL_34]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_39:.*]] = fir.convert %[[VAL_37]]#1 : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_9]] : (index) -> i64
! CHECK:             %[[VAL_41:.*]] = fir.convert %[[VAL_16]] : (index) -> i64
! CHECK:             %[[VAL_42:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[VAL_38]], %[[VAL_39]], %[[VAL_40]], %[[VAL_41]]) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
! CHECK:             %[[VAL_43:.*]] = arith.constant 0 : i32
! CHECK:             %[[VAL_44:.*]] = arith.cmpi eq, %[[VAL_42]], %[[VAL_43]] : i32
! CHECK:             hlfir.end_associate %[[VAL_37]]#1, %[[VAL_37]]#2 : !fir.ref<!fir.char<1,3>>, i1
! CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i1) -> !fir.logical<4>
! CHECK:             hlfir.yield_element %[[VAL_45]] : !fir.logical<4>
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_46:.*]] to %[[VAL_8]]#0 realloc : !hlfir.expr<?x!fir.logical<4>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
! CHECK:           hlfir.destroy %[[VAL_46]] : !hlfir.expr<?x!fir.logical<4>>
! CHECK:           hlfir.destroy %[[VAL_47:.*]] : !hlfir.expr<?x!fir.char<1,3>>
! CHECK:           return
! CHECK:         }

subroutine polymorphic_parenthesis(x, y)
  type t
  end type t
  class(t), allocatable :: x(:)
  class(t), intent(in) :: y(:)
  x = (y)
end subroutine polymorphic_parenthesis
! CHECK-LABEL:   func.func @_QPpolymorphic_parenthesis(
! CHECK-SAME:        %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QFpolymorphic_parenthesisTt>>>>> {fir.bindc_name = "x"},
! CHECK-SAME:        %[[VAL_1:.*]]: !fir.class<!fir.array<?x!fir.type<_QFpolymorphic_parenthesisTt>>> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFpolymorphic_parenthesisEx"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QFpolymorphic_parenthesisTt>>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QFpolymorphic_parenthesisTt>>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QFpolymorphic_parenthesisTt>>>>>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFpolymorphic_parenthesisEy"} : (!fir.class<!fir.array<?x!fir.type<_QFpolymorphic_parenthesisTt>>>, !fir.dscope) -> (!fir.class<!fir.array<?x!fir.type<_QFpolymorphic_parenthesisTt>>>, !fir.class<!fir.array<?x!fir.type<_QFpolymorphic_parenthesisTt>>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_3]]#0, %[[VAL_4]] : (!fir.class<!fir.array<?x!fir.type<_QFpolymorphic_parenthesisTt>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]] = hlfir.elemental %[[VAL_6]] mold %[[VAL_3]]#0 unordered : (!fir.shape<1>, !fir.class<!fir.array<?x!fir.type<_QFpolymorphic_parenthesisTt>>>) -> !hlfir.expr<?x!fir.type<_QFpolymorphic_parenthesisTt>?> {
! CHECK:           ^bb0(%[[VAL_8:.*]]: index):
! CHECK:             %[[VAL_9:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_8]])  : (!fir.class<!fir.array<?x!fir.type<_QFpolymorphic_parenthesisTt>>>, index) -> !fir.class<!fir.type<_QFpolymorphic_parenthesisTt>>
! CHECK:             %[[VAL_10:.*]] = hlfir.as_expr %[[VAL_9]] : (!fir.class<!fir.type<_QFpolymorphic_parenthesisTt>>) -> !hlfir.expr<!fir.type<_QFpolymorphic_parenthesisTt>?>
! CHECK:             hlfir.yield_element %[[VAL_10]] : !hlfir.expr<!fir.type<_QFpolymorphic_parenthesisTt>?>
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_7]] to %[[VAL_2]]#0 realloc : !hlfir.expr<?x!fir.type<_QFpolymorphic_parenthesisTt>?>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QFpolymorphic_parenthesisTt>>>>>
! CHECK:           hlfir.destroy %[[VAL_7]] : !hlfir.expr<?x!fir.type<_QFpolymorphic_parenthesisTt>?>
! CHECK:           return
! CHECK:         }

subroutine unlimited_polymorphic_parenthesis(x, y)
  class(*), allocatable :: x(:)
  class(*), intent(in) :: y(:)
  x = (y)
end subroutine unlimited_polymorphic_parenthesis
! CHECK-LABEL:   func.func @_QPunlimited_polymorphic_parenthesis(
! CHECK-SAME:        %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>> {fir.bindc_name = "x"},
! CHECK-SAME:        %[[VAL_1:.*]]: !fir.class<!fir.array<?xnone>> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFunlimited_polymorphic_parenthesisEx"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFunlimited_polymorphic_parenthesisEy"} : (!fir.class<!fir.array<?xnone>>, !fir.dscope) -> (!fir.class<!fir.array<?xnone>>, !fir.class<!fir.array<?xnone>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_3]]#0, %[[VAL_4]] : (!fir.class<!fir.array<?xnone>>, index) -> (index, index, index)
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]] = hlfir.elemental %[[VAL_6]] mold %[[VAL_3]]#0 unordered : (!fir.shape<1>, !fir.class<!fir.array<?xnone>>) -> !hlfir.expr<?xnone?> {
! CHECK:           ^bb0(%[[VAL_8:.*]]: index):
! CHECK:             %[[VAL_9:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_8]])  : (!fir.class<!fir.array<?xnone>>, index) -> !fir.class<none>
! CHECK:             %[[VAL_10:.*]] = hlfir.as_expr %[[VAL_9]] : (!fir.class<none>) -> !hlfir.expr<none?>
! CHECK:             hlfir.yield_element %[[VAL_10]] : !hlfir.expr<none?>
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_7]] to %[[VAL_2]]#0 realloc : !hlfir.expr<?xnone?>, !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>
! CHECK:           hlfir.destroy %[[VAL_7]] : !hlfir.expr<?xnone?>
! CHECK:           return
! CHECK:         }
