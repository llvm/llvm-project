! Test lowering of user defined elemental procedure reference to HLFIR
! With polymorphic arguments.
! RUN: bbc -emit-hlfir -I nw -o - %s 2>&1 | FileCheck %s
module def_some_types
  type :: t
    integer :: i
  end type
end module

subroutine test_polymporphic_array(scalar)
  use def_some_types
  interface
    function return_array_with_side_effects()
      import :: t
      class(t), pointer :: return_array_with_side_effects(:)
    end function
    integer elemental function elem(lhs, rhs)
      import :: t
      class(t), intent(in) :: lhs, rhs
    end function elem
  end interface
  class(t) :: scalar
  call bar(elem(scalar, return_array_with_side_effects()))
contains
end

! CHECK-LABEL:   func.func @_QPtest_polymporphic_array(
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMdef_some_typesTt{i:i32}>>>> {bindc_name = ".result"}
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare {{.*}}scalar
! CHECK:  %[[VAL_3:.*]] = fir.call @_QPreturn_array_with_side_effects() {{.*}}: () -> !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMdef_some_typesTt{i:i32}>>>>
! CHECK:  fir.save_result %[[VAL_3]] to %[[VAL_1:.*]] : !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMdef_some_typesTt{i:i32}>>>>, !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMdef_some_typesTt{i:i32}>>>>>
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMdef_some_typesTt{i:i32}>>>>>) -> (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMdef_some_typesTt{i:i32}>>>>>, !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMdef_some_typesTt{i:i32}>>>>>)
! CHECK:  %[[VAL_5:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMdef_some_typesTt{i:i32}>>>>>
! CHECK:  %[[VAL_6:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_5]], %[[VAL_6]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMdef_some_typesTt{i:i32}>>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_8:.*]] = fir.shape %[[VAL_7]]#1 : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_9:.*]] = hlfir.elemental %[[VAL_8]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
! CHECK:  ^bb0(%[[VAL_10:.*]]: index):
! CHECK:    %[[VAL_11:.*]] = arith.constant 0 : index
! CHECK:    %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_5]], %[[VAL_11]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMdef_some_typesTt{i:i32}>>>>, index) -> (index, index, index)
! CHECK:    %[[VAL_13:.*]] = arith.constant 1 : index
! CHECK:    %[[VAL_14:.*]] = arith.subi %[[VAL_12]]#0, %[[VAL_13]] : index
! CHECK:    %[[VAL_15:.*]] = arith.addi %[[VAL_10]], %[[VAL_14]] : index
! CHECK:    %[[VAL_16:.*]] = hlfir.designate %[[VAL_5]] (%[[VAL_15]])  : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMdef_some_typesTt{i:i32}>>>>, index) -> !fir.class<!fir.type<_QMdef_some_typesTt{i:i32}>>
! CHECK:    %[[VAL_17:.*]] = fir.call @_QPelem(%[[VAL_2]]#0, %[[VAL_16]]) {{.*}}: (!fir.class<!fir.type<_QMdef_some_typesTt{i:i32}>>, !fir.class<!fir.type<_QMdef_some_typesTt{i:i32}>>) -> i32
! CHECK:    hlfir.yield_element %[[VAL_17]] : i32
! CHECK:  }
! CHECK:  %[[VAL_18:.*]]:3 = hlfir.associate %[[VAL_9]](%[[VAL_8]]) {adapt.valuebyref} : (!hlfir.expr<?xi32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1)
! CHECK:  fir.call @_QPbar(%[[VAL_18]]#1) {{.*}}: (!fir.ref<!fir.array<?xi32>>) -> ()
