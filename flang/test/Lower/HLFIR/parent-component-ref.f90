! Test lowering of parent component references to HLFIR.
! RUN: bbc -emit-hlfir -polymorphic-type -o - %s -I nw | FileCheck %s

module pc_types
  type t
    integer :: i
  end type
  type, extends(t) :: t2
    integer :: j
  end type
interface
subroutine takes_t_type_array(x)
  import :: t
  type(t) :: x(:)
end subroutine
subroutine takes_t_class_array(x)
  import :: t
  class(t) :: x(:)
end subroutine
subroutine takes_int_array(x)
  integer :: x(:)
end subroutine
end interface
end module

subroutine test_ignored_inner_parent_comp(x)
 use pc_types
 type(t2) :: x
 call takes_int(x%t%i)
end subroutine
! CHECK-LABEL: func.func @_QPtest_ignored_inner_parent_comp(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}Ex"
! CHECK:  %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#0{"i"}   : (!fir.ref<!fir.type<_QMpc_typesTt2{i:i32,j:i32}>>) -> !fir.ref<i32>
! CHECK:  fir.call @_QPtakes_int(%[[VAL_2]])

subroutine test_ignored_inner_parent_comp_2(x)
 use pc_types
 type(t2) :: x(:)
 call takes_int_array(x%t%i)
end subroutine
! CHECK-LABEL: func.func @_QPtest_ignored_inner_parent_comp_2(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}Ex"
! CHECK:  %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_3:.*]]:3 = fir.box_dims %[[VAL_1]]#0, %[[VAL_2]] : (!fir.box<!fir.array<?x!fir.type<_QMpc_typesTt2{i:i32,j:i32}>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_4:.*]] = fir.shape %[[VAL_3]]#1 : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_5:.*]] = hlfir.designate %[[VAL_1]]#0{"i"}   shape %[[VAL_4]] : (!fir.box<!fir.array<?x!fir.type<_QMpc_typesTt2{i:i32,j:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:  fir.call @_QPtakes_int_array(%[[VAL_5]])

subroutine test_leaf_parent_ref(x)
 use pc_types
 type(t2) :: x
 call takes_parent(x%t)
end subroutine
! CHECK-LABEL: func.func @_QPtest_leaf_parent_ref(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}Ex"
! CHECK:  %[[VAL_2:.*]] = hlfir.parent_comp %[[VAL_1]]#0 : (!fir.ref<!fir.type<_QMpc_typesTt2{i:i32,j:i32}>>) -> !fir.ref<!fir.type<_QMpc_typesTt{i:i32}>>
! CHECK:  fir.call @_QPtakes_parent(%[[VAL_2]])

subroutine test_leaf_parent_ref_array(x)
 use pc_types
 class(t2) :: x(42:)
! CHECK-LABEL: func.func @_QPtest_leaf_parent_ref_array(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare {{.*}}Ex"
 call takes_t_type_array(x%t)
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_4]]#0, %[[VAL_5]] : (!fir.class<!fir.array<?x!fir.type<_QMpc_typesTt2{i:i32,j:i32}>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_7:.*]] = fir.shape %[[VAL_6]]#1 : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_8:.*]] = hlfir.parent_comp %[[VAL_4]]#0 shape %[[VAL_7]] : (!fir.class<!fir.array<?x!fir.type<_QMpc_typesTt2{i:i32,j:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.type<_QMpc_typesTt{i:i32}>>>
! CHECK:  fir.call @_QPtakes_t_type_array(%[[VAL_8]])
 call takes_t_class_array(x%t)
! CHECK:  %[[VAL_12:.*]] = hlfir.parent_comp %[[VAL_4]]#0 shape %{{.*}} : (!fir.class<!fir.array<?x!fir.type<_QMpc_typesTt2{i:i32,j:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.type<_QMpc_typesTt{i:i32}>>>
! CHECK:  %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (!fir.box<!fir.array<?x!fir.type<_QMpc_typesTt{i:i32}>>>) -> !fir.class<!fir.array<?x!fir.type<_QMpc_typesTt{i:i32}>>>
! CHECK:  fir.call @_QPtakes_t_class_array(%[[VAL_13]])
end subroutine

subroutine test_parent_section_leaf_array(x)
 use pc_types
 class(t2) :: x(:)
 call takes_t_type_array(x(2:11)%t)
end subroutine
! CHECK-LABEL: func.func @_QPtest_parent_section_leaf_array(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}Ex"
! CHECK:  %[[VAL_7:.*]] = hlfir.designate %[[VAL_1]]#0 ({{.*}})  shape %[[VAL_6:.*]] : (!fir.class<!fir.array<?x!fir.type<_QMpc_typesTt2{i:i32,j:i32}>>>, index, index, index, !fir.shape<1>) -> !fir.class<!fir.array<10x!fir.type<_QMpc_typesTt2{i:i32,j:i32}>>>
! CHECK:  %[[VAL_8:.*]] = hlfir.parent_comp %[[VAL_7]] shape %[[VAL_6]] : (!fir.class<!fir.array<10x!fir.type<_QMpc_typesTt2{i:i32,j:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.type<_QMpc_typesTt{i:i32}>>>
! CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.array<10x!fir.type<_QMpc_typesTt{i:i32}>>>) -> !fir.box<!fir.array<?x!fir.type<_QMpc_typesTt{i:i32}>>>
! CHECK:  fir.call @_QPtakes_t_type_array(%[[VAL_9]])

subroutine test_pointer_leaf_parent_ref_array(x)
 use pc_types
 class(t2), pointer :: x(:)
 call takes_t_type_array(x%t)
end subroutine
! CHECK-LABEL: func.func @_QPtest_pointer_leaf_parent_ref_array(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}Ex"
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpc_typesTt2{i:i32,j:i32}>>>>>
! CHECK:  %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_3]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpc_typesTt2{i:i32,j:i32}>>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_5:.*]] = fir.shape %[[VAL_4]]#1 : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_6:.*]] = hlfir.parent_comp %[[VAL_2]] shape %[[VAL_5]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpc_typesTt2{i:i32,j:i32}>>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.type<_QMpc_typesTt{i:i32}>>>
