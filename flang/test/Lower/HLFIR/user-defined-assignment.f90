! Test lowering of user defined assignment to HLFIR
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

module user_def
interface assignment ( = )
elemental pure subroutine logical_to_numeric(i, l)
  integer, intent (out) :: i
  logical, intent (in) :: l
end subroutine
elemental pure subroutine logical_to_complex(z, l)
  complex, intent (out) :: z
  logical, value :: l
end subroutine
pure subroutine logical_array_to_real(x, l)
  real, intent (out) :: x(:)
  logical, intent (in) :: l(:)
end subroutine
subroutine real_to_int_pointer(p, x)
  integer, pointer, intent(out) :: p(:)
  real, intent(in) :: x(:, :)
end subroutine
subroutine real_to_int_allocatable(p, x)
  integer, allocatable, intent(out) :: p(:, :)
  real, intent(in) :: x(:)
end subroutine
end interface

contains

subroutine test_user_defined_elemental_array(i, l)
   integer :: i(:)
   logical :: l(:)
   i = l
end subroutine
! CHECK-LABEL:   func.func @_QMuser_defPtest_user_defined_elemental_array(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "i"},
! CHECK-SAME:    %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "l"}) {
! CHECK:    %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMuser_defFtest_user_defined_elemental_arrayEi"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:    %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMuser_defFtest_user_defined_elemental_arrayEl"} : (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
! CHECK:    hlfir.region_assign {
! CHECK:      hlfir.yield %[[VAL_3]]#0 : !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:    } to {
! CHECK:      hlfir.yield %[[VAL_2]]#0 : !fir.box<!fir.array<?xi32>>
! CHECK:    } user_defined_assign  (%[[VAL_4:.*]]: !fir.ref<!fir.logical<4>>) to (%[[VAL_5:.*]]: !fir.ref<i32>) {
! CHECK:      fir.call @_QPlogical_to_numeric(%[[VAL_5]], %[[VAL_4]]) {{.*}}: (!fir.ref<i32>, !fir.ref<!fir.logical<4>>) -> ()
! CHECK:    }

subroutine test_user_defined_elemental_array_value(z, l)
   logical :: l(:)
   complex :: z(:)
   z = l
end subroutine
! CHECK-LABEL:   func.func @_QMuser_defPtest_user_defined_elemental_array_value(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.complex<4>>> {fir.bindc_name = "z"},
! CHECK-SAME:    %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "l"}) {
! CHECK:    %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMuser_defFtest_user_defined_elemental_array_valueEl"} : (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
! CHECK:    %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMuser_defFtest_user_defined_elemental_array_valueEz"} : (!fir.box<!fir.array<?x!fir.complex<4>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.complex<4>>>, !fir.box<!fir.array<?x!fir.complex<4>>>)
! CHECK:    hlfir.region_assign {
! CHECK:      hlfir.yield %[[VAL_2]]#0 : !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:    } to {
! CHECK:      hlfir.yield %[[VAL_3]]#0 : !fir.box<!fir.array<?x!fir.complex<4>>>
! CHECK:    } user_defined_assign  (%[[VAL_4:.*]]: !fir.ref<!fir.logical<4>>) to (%[[VAL_5:.*]]: !fir.ref<!fir.complex<4>>) {
! CHECK:      %[[VAL_6:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.logical<4>>
! CHECK:      fir.call @_QPlogical_to_complex(%[[VAL_5]], %[[VAL_6]]) {{.*}}: (!fir.ref<!fir.complex<4>>, !fir.logical<4>) -> ()
! CHECK:    }

subroutine test_user_defined_scalar(i, l)
   integer :: i
   logical :: l
   i = l
end subroutine
! CHECK-LABEL:   func.func @_QMuser_defPtest_user_defined_scalar(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
! CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "l"}) {
! CHECK:    %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMuser_defFtest_user_defined_scalarEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:    %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMuser_defFtest_user_defined_scalarEl"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:    hlfir.region_assign {
! CHECK:      %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:      hlfir.yield %[[VAL_4]] : !fir.logical<4>
! CHECK:    } to {
! CHECK:      hlfir.yield %[[VAL_2]]#0 : !fir.ref<i32>
! CHECK:    } user_defined_assign  (%[[VAL_5:.*]]: !fir.logical<4>) to (%[[VAL_6:.*]]: !fir.ref<i32>) {
! CHECK:      %[[VAL_7:.*]]:3 = hlfir.associate %[[VAL_5]] {adapt.valuebyref} : (!fir.logical<4>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>, i1)
! CHECK:      fir.call @_QPlogical_to_numeric(%[[VAL_6]], %[[VAL_7]]#1) {{.*}}: (!fir.ref<i32>, !fir.ref<!fir.logical<4>>) -> ()
! CHECK:      hlfir.end_associate %[[VAL_7]]#1, %[[VAL_7]]#2 : !fir.ref<!fir.logical<4>>, i1
! CHECK:    }

subroutine test_non_elemental_array(x)
 real :: x(:)
 x = x.lt.42
end subroutine
! CHECK-LABEL:   func.func @_QMuser_defPtest_non_elemental_array(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
! CHECK:    %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMuser_defFtest_non_elemental_arrayEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:    hlfir.region_assign {
! CHECK:      %[[VAL_2:.*]] = arith.constant 4.200000e+01 : f32
! CHECK:      %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:      %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_1]]#0, %[[VAL_3]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:      %[[VAL_5:.*]] = fir.shape %[[VAL_4]]#1 : (index) -> !fir.shape<1>
! CHECK:      %[[VAL_6:.*]] = hlfir.elemental %[[VAL_5]] unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
! CHECK:      ^bb0(%[[VAL_7:.*]]: index):
! CHECK:        %[[VAL_8:.*]] = hlfir.designate %[[VAL_1]]#0 (%[[VAL_7]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK:        %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ref<f32>
! CHECK:        %[[VAL_10:.*]] = arith.cmpf olt, %[[VAL_9]], %[[VAL_2]] {{.*}} : f32
! CHECK:        %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i1) -> !fir.logical<4>
! CHECK:        hlfir.yield_element %[[VAL_11]] : !fir.logical<4>
! CHECK:      }
! CHECK:      hlfir.yield %[[VAL_12:.*]] : !hlfir.expr<?x!fir.logical<4>> cleanup {
! CHECK:        hlfir.destroy %[[VAL_12]] : !hlfir.expr<?x!fir.logical<4>>
! CHECK:      }
! CHECK:    } to {
! CHECK:      hlfir.yield %[[VAL_1]]#0 : !fir.box<!fir.array<?xf32>>
! CHECK:    } user_defined_assign  (%[[VAL_13:.*]]: !hlfir.expr<?x!fir.logical<4>>) to (%[[VAL_14:.*]]: !fir.box<!fir.array<?xf32>>) {
! CHECK:      %[[VAL_15:.*]] = hlfir.shape_of %[[VAL_13]] : (!hlfir.expr<?x!fir.logical<4>>) -> !fir.shape<1>
! CHECK:      %[[VAL_16:.*]]:3 = hlfir.associate %[[VAL_13]](%[[VAL_15]]) {adapt.valuebyref} : (!hlfir.expr<?x!fir.logical<4>>, !fir.shape<1>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.ref<!fir.array<?x!fir.logical<4>>>, i1)
! CHECK:      fir.call @_QPlogical_array_to_real(%[[VAL_14]], %[[VAL_16]]#0) {{.*}}: (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?x!fir.logical<4>>>) -> ()
! CHECK:      hlfir.end_associate %[[VAL_16]]#1, %[[VAL_16]]#2 : !fir.ref<!fir.array<?x!fir.logical<4>>>, i1
! CHECK:    }

subroutine test_where_user_def_assignment(i, l, l2)
   integer :: i(:)
   logical :: l(:), l2(:)
   where (l) i = l.neqv.l2
end subroutine
! CHECK-LABEL:   func.func @_QMuser_defPtest_where_user_def_assignment(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "i"},
! CHECK-SAME:    %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "l"},
! CHECK-SAME:    %[[VAL_2:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "l2"}) {
! CHECK:    %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMuser_defFtest_where_user_def_assignmentEi"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:    %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMuser_defFtest_where_user_def_assignmentEl"} : (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
! CHECK:    %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMuser_defFtest_where_user_def_assignmentEl2"} : (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
! CHECK:    hlfir.where {
! CHECK:      hlfir.yield %[[VAL_4]]#0 : !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:    } do {
! CHECK:      hlfir.region_assign {
! CHECK:        %[[VAL_6:.*]] = arith.constant 0 : index
! CHECK:        %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_4]]#0, %[[VAL_6]] : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> (index, index, index)
! CHECK:        %[[VAL_8:.*]] = fir.shape %[[VAL_7]]#1 : (index) -> !fir.shape<1>
! CHECK:        %[[VAL_9:.*]] = hlfir.elemental %[[VAL_8]] unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
! CHECK:        ^bb0(%[[VAL_10:.*]]: index):
! CHECK:          %[[VAL_11:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_10]])  : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
! CHECK:          %[[VAL_12:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_10]])  : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
! CHECK:          %[[VAL_13:.*]] = fir.load %[[VAL_11]] : !fir.ref<!fir.logical<4>>
! CHECK:          %[[VAL_14:.*]] = fir.load %[[VAL_12]] : !fir.ref<!fir.logical<4>>
! CHECK:          %[[VAL_15:.*]] = fir.convert %[[VAL_13]] : (!fir.logical<4>) -> i1
! CHECK:          %[[VAL_16:.*]] = fir.convert %[[VAL_14]] : (!fir.logical<4>) -> i1
! CHECK:          %[[VAL_17:.*]] = arith.cmpi ne, %[[VAL_15]], %[[VAL_16]] : i1
! CHECK:          %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i1) -> !fir.logical<4>
! CHECK:          hlfir.yield_element %[[VAL_18]] : !fir.logical<4>
! CHECK:        }
! CHECK:        hlfir.yield %[[VAL_19:.*]] : !hlfir.expr<?x!fir.logical<4>> cleanup {
! CHECK:          hlfir.destroy %[[VAL_19]] : !hlfir.expr<?x!fir.logical<4>>
! CHECK:        }
! CHECK:      } to {
! CHECK:        hlfir.yield %[[VAL_3]]#0 : !fir.box<!fir.array<?xi32>>
! CHECK:      } user_defined_assign  (%[[VAL_20:.*]]: !fir.logical<4>) to (%[[VAL_21:.*]]: !fir.ref<i32>) {
! CHECK:        %[[VAL_22:.*]]:3 = hlfir.associate %[[VAL_20]] {adapt.valuebyref} : (!fir.logical<4>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>, i1)
! CHECK:        fir.call @_QPlogical_to_numeric(%[[VAL_21]], %[[VAL_22]]#1) {{.*}}: (!fir.ref<i32>, !fir.ref<!fir.logical<4>>) -> ()
! CHECK:        hlfir.end_associate %[[VAL_22]]#1, %[[VAL_22]]#2 : !fir.ref<!fir.logical<4>>, i1
! CHECK:      }
! CHECK:    }

subroutine test_forall_user_def_assignment(i, l)
   integer :: i(20, 10)
   logical :: l(20, 10)
   forall (j=1:10) i(:, j) = l(:, j)
end subroutine
! CHECK-LABEL:   func.func @_QMuser_defPtest_forall_user_def_assignment(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<20x10xi32>> {fir.bindc_name = "i"},
! CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<!fir.array<20x10x!fir.logical<4>>> {fir.bindc_name = "l"}) {
! CHECK:    %[[VAL_2:.*]] = arith.constant 20 : index
! CHECK:    %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:    %[[VAL_4:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
! CHECK:    %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_4]]) dummy_scope %{{[0-9]+}} {uniq_name = "_QMuser_defFtest_forall_user_def_assignmentEi"} : (!fir.ref<!fir.array<20x10xi32>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<20x10xi32>>, !fir.ref<!fir.array<20x10xi32>>)
! CHECK:    %[[VAL_6:.*]] = arith.constant 20 : index
! CHECK:    %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:    %[[VAL_8:.*]] = fir.shape %[[VAL_6]], %[[VAL_7]] : (index, index) -> !fir.shape<2>
! CHECK:    %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_8]]) dummy_scope %{{[0-9]+}} {uniq_name = "_QMuser_defFtest_forall_user_def_assignmentEl"} : (!fir.ref<!fir.array<20x10x!fir.logical<4>>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<20x10x!fir.logical<4>>>, !fir.ref<!fir.array<20x10x!fir.logical<4>>>)
! CHECK:    %[[VAL_10:.*]] = arith.constant 1 : i32
! CHECK:    %[[VAL_11:.*]] = arith.constant 10 : i32
! CHECK:    hlfir.forall lb {
! CHECK:      hlfir.yield %[[VAL_10]] : i32
! CHECK:    } ub {
! CHECK:      hlfir.yield %[[VAL_11]] : i32
! CHECK:    }  (%[[VAL_12:.*]]: i32) {
! CHECK:      %[[VAL_13:.*]] = hlfir.forall_index "j" %[[VAL_12]] : (i32) -> !fir.ref<i32>
! CHECK:      hlfir.region_assign {
! CHECK:        %[[VAL_14:.*]] = arith.constant 1 : index
! CHECK:        %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:        %[[VAL_16:.*]] = arith.constant 20 : index
! CHECK:        %[[VAL_17:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
! CHECK:        %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> i64
! CHECK:        %[[VAL_19:.*]] = fir.shape %[[VAL_16]] : (index) -> !fir.shape<1>
! CHECK:        %[[VAL_20:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_14]]:%[[VAL_6]]:%[[VAL_15]], %[[VAL_18]])  shape %[[VAL_19]] : (!fir.ref<!fir.array<20x10x!fir.logical<4>>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<20x!fir.logical<4>>>
! CHECK:        hlfir.yield %[[VAL_20]] : !fir.ref<!fir.array<20x!fir.logical<4>>>
! CHECK:      } to {
! CHECK:        %[[VAL_21:.*]] = arith.constant 1 : index
! CHECK:        %[[VAL_22:.*]] = arith.constant 1 : index
! CHECK:        %[[VAL_23:.*]] = arith.constant 20 : index
! CHECK:        %[[VAL_24:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
! CHECK:        %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i32) -> i64
! CHECK:        %[[VAL_26:.*]] = fir.shape %[[VAL_23]] : (index) -> !fir.shape<1>
! CHECK:        %[[VAL_27:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_21]]:%[[VAL_2]]:%[[VAL_22]], %[[VAL_25]])  shape %[[VAL_26]] : (!fir.ref<!fir.array<20x10xi32>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<20xi32>>
! CHECK:        hlfir.yield %[[VAL_27]] : !fir.ref<!fir.array<20xi32>>
! CHECK:      } user_defined_assign  (%[[VAL_28:.*]]: !fir.ref<!fir.logical<4>>) to (%[[VAL_29:.*]]: !fir.ref<i32>) {
! CHECK:        fir.call @_QPlogical_to_numeric(%[[VAL_29]], %[[VAL_28]]) {{.*}}: (!fir.ref<i32>, !fir.ref<!fir.logical<4>>) -> ()
! CHECK:      }
! CHECK:    }

subroutine test_forall_user_def_assignment_non_elemental_array(x, l)
   real :: x(20, 10)
   logical :: l(20, 10)
   forall (j=1:10) x(:, j) = l(:, j)
end subroutine
! CHECK-LABEL:   func.func @_QMuser_defPtest_forall_user_def_assignment_non_elemental_array(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<20x10xf32>> {fir.bindc_name = "x"},
! CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<!fir.array<20x10x!fir.logical<4>>> {fir.bindc_name = "l"}) {
! CHECK:    %[[VAL_2:.*]] = arith.constant 20 : index
! CHECK:    %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:    %[[VAL_4:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
! CHECK:    %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_4]]) dummy_scope %{{[0-9]+}} {uniq_name = "_QMuser_defFtest_forall_user_def_assignment_non_elemental_arrayEl"} : (!fir.ref<!fir.array<20x10x!fir.logical<4>>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<20x10x!fir.logical<4>>>, !fir.ref<!fir.array<20x10x!fir.logical<4>>>)
! CHECK:    %[[VAL_6:.*]] = arith.constant 20 : index
! CHECK:    %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:    %[[VAL_8:.*]] = fir.shape %[[VAL_6]], %[[VAL_7]] : (index, index) -> !fir.shape<2>
! CHECK:    %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_8]]) dummy_scope %{{[0-9]+}} {uniq_name = "_QMuser_defFtest_forall_user_def_assignment_non_elemental_arrayEx"} : (!fir.ref<!fir.array<20x10xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<20x10xf32>>, !fir.ref<!fir.array<20x10xf32>>)
! CHECK:    %[[VAL_10:.*]] = arith.constant 1 : i32
! CHECK:    %[[VAL_11:.*]] = arith.constant 10 : i32
! CHECK:    hlfir.forall lb {
! CHECK:      hlfir.yield %[[VAL_10]] : i32
! CHECK:    } ub {
! CHECK:      hlfir.yield %[[VAL_11]] : i32
! CHECK:    }  (%[[VAL_12:.*]]: i32) {
! CHECK:      %[[VAL_13:.*]] = hlfir.forall_index "j" %[[VAL_12]] : (i32) -> !fir.ref<i32>
! CHECK:      hlfir.region_assign {
! CHECK:        %[[VAL_14:.*]] = arith.constant 1 : index
! CHECK:        %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:        %[[VAL_16:.*]] = arith.constant 20 : index
! CHECK:        %[[VAL_17:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
! CHECK:        %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> i64
! CHECK:        %[[VAL_19:.*]] = fir.shape %[[VAL_16]] : (index) -> !fir.shape<1>
! CHECK:        %[[VAL_20:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_14]]:%[[VAL_2]]:%[[VAL_15]], %[[VAL_18]])  shape %[[VAL_19]] : (!fir.ref<!fir.array<20x10x!fir.logical<4>>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<20x!fir.logical<4>>>
! CHECK:        hlfir.yield %[[VAL_20]] : !fir.ref<!fir.array<20x!fir.logical<4>>>
! CHECK:      } to {
! CHECK:        %[[VAL_21:.*]] = arith.constant 1 : index
! CHECK:        %[[VAL_22:.*]] = arith.constant 1 : index
! CHECK:        %[[VAL_23:.*]] = arith.constant 20 : index
! CHECK:        %[[VAL_24:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
! CHECK:        %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i32) -> i64
! CHECK:        %[[VAL_26:.*]] = fir.shape %[[VAL_23]] : (index) -> !fir.shape<1>
! CHECK:        %[[VAL_27:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_21]]:%[[VAL_6]]:%[[VAL_22]], %[[VAL_25]])  shape %[[VAL_26]] : (!fir.ref<!fir.array<20x10xf32>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<20xf32>>
! CHECK:        hlfir.yield %[[VAL_27]] : !fir.ref<!fir.array<20xf32>>
! CHECK:      } user_defined_assign  (%[[VAL_28:.*]]: !fir.ref<!fir.array<20x!fir.logical<4>>>) to (%[[VAL_29:.*]]: !fir.ref<!fir.array<20xf32>>) {
! CHECK:        %[[VAL_30:.*]] = arith.constant 20 : index
! CHECK:        %[[VAL_31:.*]] = fir.shape %[[VAL_30]] : (index) -> !fir.shape<1>
! CHECK:        %[[VAL_32:.*]] = fir.embox %[[VAL_29]](%[[VAL_31]]) : (!fir.ref<!fir.array<20xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<20xf32>>
! CHECK:        %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (!fir.box<!fir.array<20xf32>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:        %[[VAL_34:.*]] = arith.constant 20 : index
! CHECK:        %[[VAL_35:.*]] = fir.shape %[[VAL_34]] : (index) -> !fir.shape<1>
! CHECK:        %[[VAL_36:.*]] = fir.embox %[[VAL_28]](%[[VAL_35]]) : (!fir.ref<!fir.array<20x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.array<20x!fir.logical<4>>>
! CHECK:        %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (!fir.box<!fir.array<20x!fir.logical<4>>>) -> !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:        fir.call @_QPlogical_array_to_real(%[[VAL_33]], %[[VAL_37]]) {{.*}}: (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?x!fir.logical<4>>>) -> ()
! CHECK:      }

subroutine test_pointer(p, x)
  integer, pointer :: p(:)
  real :: x(:, :)
  p = x
end subroutine
! CHECK-LABEL:   func.func @_QMuser_defPtest_pointer(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "p"},
! CHECK-SAME:    %[[VAL_1:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
! CHECK:    %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMuser_defFtest_pointerEp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! CHECK:    %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMuser_defFtest_pointerEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
! CHECK:    hlfir.region_assign {
! CHECK:      hlfir.yield %[[VAL_3]]#0 : !fir.box<!fir.array<?x?xf32>>
! CHECK:    } to {
! CHECK:      hlfir.yield %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:    } user_defined_assign  (%[[VAL_4:.*]]: !fir.box<!fir.array<?x?xf32>>) to (%[[VAL_5:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) {
! CHECK:      fir.call @_QPreal_to_int_pointer(%[[VAL_5]], %[[VAL_4]]) {{.*}}: (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.box<!fir.array<?x?xf32>>) -> ()
! CHECK:    }

subroutine test_allocatable(a, x)
  integer, allocatable :: a(:,:)
  real :: x(:)
  a = x
end subroutine
! CHECK-LABEL:   func.func @_QMuser_defPtest_allocatable(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[VAL_1:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
! CHECK:    %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMuser_defFtest_allocatableEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>)
! CHECK:    %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMuser_defFtest_allocatableEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:    hlfir.region_assign {
! CHECK:      hlfir.yield %[[VAL_3]]#0 : !fir.box<!fir.array<?xf32>>
! CHECK:    } to {
! CHECK:      hlfir.yield %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
! CHECK:    } user_defined_assign  (%[[VAL_4:.*]]: !fir.box<!fir.array<?xf32>>) to (%[[VAL_5:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) {
! CHECK:      fir.call @_QPreal_to_int_allocatable(%[[VAL_5]], %[[VAL_4]]) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>, !fir.box<!fir.array<?xf32>>) -> ()
! CHECK:    }

end module

subroutine test_char_get_length(ch)
  integer :: x
  interface assignment(=)
     subroutine test_char_get_length_callee(a,b)
       integer, intent(out) :: a
       character, intent(in) :: b
     end subroutine test_char_get_length_callee
  end interface assignment(=)
  character(*) :: ch
  x = 'abc'//ch
end subroutine test_char_get_length
! CHECK-LABEL:   func.func @_QPtest_char_get_length(
! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "ch"}) {
! CHECK:           %[[VAL_1:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]]#0 typeparams %[[VAL_1]]#1 dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest_char_get_lengthEch"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:           %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFtest_char_get_lengthEx"}
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFtest_char_get_lengthEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           hlfir.region_assign {
! CHECK:             %[[VAL_5:.*]] = fir.address_of(@_QQclX616263) : !fir.ref<!fir.char<1,3>>
! CHECK:             %[[VAL_6:.*]] = arith.constant 3 : index
! CHECK:             %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_5]] typeparams %[[VAL_6]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX616263"} : (!fir.ref<!fir.char<1,3>>, index) -> (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>)
! CHECK:             %[[VAL_8:.*]] = arith.addi %[[VAL_6]], %[[VAL_1]]#1 : index
! CHECK:             %[[VAL_9:.*]] = hlfir.concat %[[VAL_7]]#0, %[[VAL_2]]#0 len %[[VAL_8]] : (!fir.ref<!fir.char<1,3>>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:             hlfir.yield %[[VAL_9]] : !hlfir.expr<!fir.char<1,?>>
! CHECK:           } to {
! CHECK:             hlfir.yield %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:           } user_defined_assign  (%[[VAL_10:.*]]: !hlfir.expr<!fir.char<1,?>>) to (%[[VAL_11:.*]]: !fir.ref<i32>) {
! CHECK:             %[[VAL_12:.*]] = hlfir.get_length %[[VAL_10]] : (!hlfir.expr<!fir.char<1,?>>) -> index
! CHECK:             %[[VAL_13:.*]]:3 = hlfir.associate %[[VAL_10]] typeparams %[[VAL_12]] {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
! CHECK:             fir.call @_QPtest_char_get_length_callee(%[[VAL_11]], %[[VAL_13]]#0) fastmath<contract> : (!fir.ref<i32>, !fir.boxchar<1>) -> ()
! CHECK:             hlfir.end_associate %[[VAL_13]]#1, %[[VAL_13]]#2 : !fir.ref<!fir.char<1,?>>, i1
! CHECK:           }
! CHECK:           return
! CHECK:         }
