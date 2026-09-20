! Test how transformational intrinsic function references are lowered

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! The exact intrinsic being tested does not really matter, what is
! tested here is that transformational intrinsics are lowered correctly
! regardless of the context they appear into.



module test2
interface
  subroutine takes_array_desc(l)
    logical(1) :: l(:)
  end subroutine
end interface

contains

! CHECK-LABEL: func @_QMtest2Pin_io(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<1>>>{{.*}}) {
subroutine in_io(x)
  logical(1) :: x(:, :)
  ! CHECK: %[[xdecl:.*]]:2 = hlfir.declare %[[arg0]]{{.*}}{uniq_name = "_QMtest2Fin_ioEx"}
  ! CHECK: %[[c1:.*]] = arith.constant 1 : i32
  ! CHECK: %[[all:.*]] = hlfir.all %[[xdecl]]#0 dim %[[c1]] : (!fir.box<!fir.array<?x?x!fir.logical<1>>>, i32) -> !hlfir.expr<?x!fir.logical<1>>
  ! CHECK: %[[shape:.*]] = hlfir.shape_of %[[all]] : (!hlfir.expr<?x!fir.logical<1>>) -> !fir.shape<1>
  ! CHECK: %[[assoc:.*]]:3 = hlfir.associate %[[all]](%[[shape]]) {adapt.valuebyref} : (!hlfir.expr<?x!fir.logical<1>>, !fir.shape<1>) -> (!fir.box<!fir.array<?x!fir.logical<1>>>, !fir.ref<!fir.array<?x!fir.logical<1>>>, i1)
  ! CHECK: %[[ext:.*]] = hlfir.get_extent %[[shape]] {dim = 0 : index} : (!fir.shape<1>) -> index
  ! CHECK: %[[s2:.*]] = fir.shape %[[ext]] : (index) -> !fir.shape<1>
  ! CHECK: %[[box:.*]] = fir.embox %[[assoc]]#1(%[[s2]]) : (!fir.ref<!fir.array<?x!fir.logical<1>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.logical<1>>>
  ! CHECK: %[[boxnone:.*]] = fir.convert %[[box]] : (!fir.box<!fir.array<?x!fir.logical<1>>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAioOutputDescriptor({{.*}}, %[[boxnone]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
  ! CHECK: hlfir.end_associate %[[assoc]]#1, %[[assoc]]#2 : !fir.ref<!fir.array<?x!fir.logical<1>>>, i1
  ! CHECK: hlfir.destroy %[[all]] : !hlfir.expr<?x!fir.logical<1>>
  print *, all(x, 1)
end subroutine

! CHECK-LABEL: func @_QMtest2Pin_call(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<1>>>{{.*}}) {
subroutine in_call(x)
  implicit none
  logical(1) :: x(:, :)
  ! CHECK: %[[xdecl:.*]]:2 = hlfir.declare %[[arg0]]{{.*}}{uniq_name = "_QMtest2Fin_callEx"}
  ! CHECK: %[[c1:.*]] = arith.constant 1 : i32
  ! CHECK: %[[all:.*]] = hlfir.all %[[xdecl]]#0 dim %[[c1]] : (!fir.box<!fir.array<?x?x!fir.logical<1>>>, i32) -> !hlfir.expr<?x!fir.logical<1>>
  ! CHECK: %[[shape:.*]] = hlfir.shape_of %[[all]] : (!hlfir.expr<?x!fir.logical<1>>) -> !fir.shape<1>
  ! CHECK: %[[assoc:.*]]:3 = hlfir.associate %[[all]](%[[shape]]) {adapt.valuebyref} : (!hlfir.expr<?x!fir.logical<1>>, !fir.shape<1>) -> (!fir.box<!fir.array<?x!fir.logical<1>>>, !fir.ref<!fir.array<?x!fir.logical<1>>>, i1)
  ! CHECK: fir.call @_QPtakes_array_desc(%[[assoc]]#0) {{.*}}: (!fir.box<!fir.array<?x!fir.logical<1>>>) -> ()
  ! CHECK: hlfir.end_associate %[[assoc]]#1, %[[assoc]]#2 : !fir.ref<!fir.array<?x!fir.logical<1>>>, i1
  ! CHECK: hlfir.destroy %[[all]] : !hlfir.expr<?x!fir.logical<1>>
  call takes_array_desc(all(x, 1))
end subroutine

! CHECK-LABEL: func @_QMtest2Pin_implicit_call(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<1>>>{{.*}}) {
subroutine in_implicit_call(x)
  logical(1) :: x(:, :)
  ! CHECK: %[[xdecl:.*]]:2 = hlfir.declare %[[arg0]]{{.*}}{uniq_name = "_QMtest2Fin_implicit_callEx"}
  ! CHECK: %[[c1:.*]] = arith.constant 1 : i32
  ! CHECK: %[[all:.*]] = hlfir.all %[[xdecl]]#0 dim %[[c1]] : (!fir.box<!fir.array<?x?x!fir.logical<1>>>, i32) -> !hlfir.expr<?x!fir.logical<1>>
  ! CHECK: %[[shape:.*]] = hlfir.shape_of %[[all]] : (!hlfir.expr<?x!fir.logical<1>>) -> !fir.shape<1>
  ! CHECK: %[[assoc:.*]]:3 = hlfir.associate %[[all]](%[[shape]]) {adapt.valuebyref} : (!hlfir.expr<?x!fir.logical<1>>, !fir.shape<1>) -> (!fir.box<!fir.array<?x!fir.logical<1>>>, !fir.ref<!fir.array<?x!fir.logical<1>>>, i1)
  ! CHECK: fir.call @_QPtakes_implicit_array(%[[assoc]]#1) {{.*}}: (!fir.ref<!fir.array<?x!fir.logical<1>>>) -> ()
  ! CHECK: hlfir.end_associate %[[assoc]]#1, %[[assoc]]#2 : !fir.ref<!fir.array<?x!fir.logical<1>>>, i1
  ! CHECK: hlfir.destroy %[[all]] : !hlfir.expr<?x!fir.logical<1>>
  call takes_implicit_array(all(x, 1))
end subroutine

! CHECK-LABEL: func @_QMtest2Pin_assignment(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<1>>>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?x!fir.logical<1>>>{{.*}})
subroutine in_assignment(x, y)
  logical(1) :: x(:, :), y(:)
  ! CHECK: %[[xdecl:.*]]:2 = hlfir.declare %[[arg0]]{{.*}}{uniq_name = "_QMtest2Fin_assignmentEx"}
  ! CHECK: %[[ydecl:.*]]:2 = hlfir.declare %[[arg1]]{{.*}}{uniq_name = "_QMtest2Fin_assignmentEy"}
  ! CHECK: %[[c1:.*]] = arith.constant 1 : i32
  ! CHECK: %[[all:.*]] = hlfir.all %[[xdecl]]#0 dim %[[c1]] : (!fir.box<!fir.array<?x?x!fir.logical<1>>>, i32) -> !hlfir.expr<?x!fir.logical<1>>
  ! CHECK: hlfir.assign %[[all]] to %[[ydecl]]#0 : !hlfir.expr<?x!fir.logical<1>>, !fir.box<!fir.array<?x!fir.logical<1>>>
  ! CHECK: hlfir.destroy %[[all]] : !hlfir.expr<?x!fir.logical<1>>
  y = all(x, 1)
end subroutine

! CHECK-LABEL: func @_QMtest2Pin_elem_expr(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<1>>>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?x!fir.logical<1>>>{{.*}}, %[[arg2:.*]]: !fir.box<!fir.array<?x!fir.logical<1>>>{{.*}})
subroutine in_elem_expr(x, y, z)
  logical(1) :: x(:, :), y(:), z(:)
  ! CHECK: %[[xdecl:.*]]:2 = hlfir.declare %[[arg0]]{{.*}}{uniq_name = "_QMtest2Fin_elem_exprEx"}
  ! CHECK: %[[ydecl:.*]]:2 = hlfir.declare %[[arg1]]{{.*}}{uniq_name = "_QMtest2Fin_elem_exprEy"}
  ! CHECK: %[[zdecl:.*]]:2 = hlfir.declare %[[arg2]]{{.*}}{uniq_name = "_QMtest2Fin_elem_exprEz"}
  ! CHECK: %[[c1:.*]] = arith.constant 1 : i32
  ! CHECK: %[[all:.*]] = hlfir.all %[[xdecl]]#0 dim %[[c1]] : (!fir.box<!fir.array<?x?x!fir.logical<1>>>, i32) -> !hlfir.expr<?x!fir.logical<1>>
  ! CHECK: %[[ydims:.*]]:3 = fir.box_dims %[[ydecl]]#0, %{{.*}} : (!fir.box<!fir.array<?x!fir.logical<1>>>, index) -> (index, index, index)
  ! CHECK: %[[shape:.*]] = fir.shape %[[ydims]]#1 : (index) -> !fir.shape<1>
  ! CHECK: %[[elem:.*]] = hlfir.elemental %[[shape]] unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<1>> {
  ! CHECK:   %[[ydes:.*]] = hlfir.designate %[[ydecl]]#0 (%{{.*}})  : (!fir.box<!fir.array<?x!fir.logical<1>>>, index) -> !fir.ref<!fir.logical<1>>
  ! CHECK:   %[[allval:.*]] = hlfir.apply %[[all]], %{{.*}} : (!hlfir.expr<?x!fir.logical<1>>, index) -> !fir.logical<1>
  ! CHECK:   %[[yval:.*]] = fir.load %[[ydes]] : !fir.ref<!fir.logical<1>>
  ! CHECK:   %[[neqv:.*]] = fir.neqv %[[yval]], %[[allval]] : !fir.logical<1>
  ! CHECK:   hlfir.yield_element %[[neqv]] : !fir.logical<1>
  ! CHECK: }
  ! CHECK: hlfir.assign %[[elem]] to %[[zdecl]]#0 : !hlfir.expr<?x!fir.logical<1>>, !fir.box<!fir.array<?x!fir.logical<1>>>
  ! CHECK: hlfir.destroy %[[elem]] : !hlfir.expr<?x!fir.logical<1>>
  ! CHECK: hlfir.destroy %[[all]] : !hlfir.expr<?x!fir.logical<1>>
  z = y .neqv. all(x, 1)
end subroutine

! CSHIFT

  ! CHECK-LABEL: func @_QMtest2Pcshift_test() {
  ! CHECK:         %[[VAL_5:.*]] = fir.alloca !fir.array<3x3xi32> {bindc_name = "array", uniq_name = "_QMtest2Fcshift_testEarray"}
  ! CHECK:         %[[ARRAY_DECL:.*]]:2 = hlfir.declare %[[VAL_5]](%{{.*}}) {uniq_name = "_QMtest2Fcshift_testEarray"}
  ! CHECK:         %[[VAL_8:.*]] = fir.alloca !fir.array<3x3xi32> {bindc_name = "result", uniq_name = "_QMtest2Fcshift_testEresult"}
  ! CHECK:         %[[RESULT_DECL:.*]]:2 = hlfir.declare %[[VAL_8]](%{{.*}}) {uniq_name = "_QMtest2Fcshift_testEresult"}
  ! CHECK:         %[[VAL_10:.*]] = fir.alloca !fir.array<3xi32> {bindc_name = "shift", uniq_name = "_QMtest2Fcshift_testEshift"}
  ! CHECK:         %[[SHIFT_DECL:.*]]:2 = hlfir.declare %[[VAL_10]](%{{.*}}) {uniq_name = "_QMtest2Fcshift_testEshift"}
  ! CHECK:         %[[VAL_12:.*]] = fir.alloca !fir.array<6xi32> {bindc_name = "vector", uniq_name = "_QMtest2Fcshift_testEvector"}
  ! CHECK:         %[[VECTOR_DECL:.*]]:2 = hlfir.declare %[[VAL_12]](%{{.*}}) {uniq_name = "_QMtest2Fcshift_testEvector"}
  ! CHECK:         %[[VAL_14:.*]] = fir.alloca !fir.array<6xi32> {bindc_name = "vectorresult", uniq_name = "_QMtest2Fcshift_testEvectorresult"}
  ! CHECK:         %[[VECTORRESULT_DECL:.*]]:2 = hlfir.declare %[[VAL_14]](%{{.*}}) {uniq_name = "_QMtest2Fcshift_testEvectorresult"}
  ! CHECK:         %[[C2:.*]] = arith.constant 2 : i32
  ! CHECK:         %[[CSHIFT1:.*]] = hlfir.cshift %[[ARRAY_DECL]]#0 %[[SHIFT_DECL]]#0 dim %[[C2]] : (!fir.ref<!fir.array<3x3xi32>>, !fir.ref<!fir.array<3xi32>>, i32) -> !hlfir.expr<3x3xi32>
  ! CHECK:         hlfir.assign %[[CSHIFT1]] to %[[RESULT_DECL]]#0 : !hlfir.expr<3x3xi32>, !fir.ref<!fir.array<3x3xi32>>
  ! CHECK:         hlfir.destroy %[[CSHIFT1]] : !hlfir.expr<3x3xi32>
  ! CHECK:         %[[C3:.*]] = arith.constant 3 : i32
  ! CHECK:         %[[CSHIFT2:.*]] = hlfir.cshift %[[VECTOR_DECL]]#0 %[[C3]] : (!fir.ref<!fir.array<6xi32>>, i32) -> !hlfir.expr<6xi32>
  ! CHECK:         hlfir.assign %[[CSHIFT2]] to %[[VECTORRESULT_DECL]]#0 : !hlfir.expr<6xi32>, !fir.ref<!fir.array<6xi32>>
  ! CHECK:         hlfir.destroy %[[CSHIFT2]] : !hlfir.expr<6xi32>
  ! CHECK:         return
  ! CHECK:       }

subroutine cshift_test()
  integer, dimension (3, 3) :: array
  integer, dimension(3) :: shift
  integer, dimension(3, 3) :: result
  integer, dimension(6) :: vectorResult
  integer, dimension (6) :: vector
  result = cshift(array, shift, 2) ! non-vector case
  vectorResult = cshift(vector, 3) ! vector case
end subroutine cshift_test

! UNPACK
! CHECK-LABEL: func @_QMtest2Punpack_test
subroutine unpack_test()
  integer, dimension(3) :: vector
  integer, dimension (3,3) :: field

  logical, dimension(3,3) :: mask
  integer, dimension(3,3) :: result
  result = unpack(vector, mask, field)
  ! CHECK-DAG: %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>>
  ! CHECK-DAG: %[[a1:.*]] = fir.alloca i32
  ! CHECK-DAG: %[[a2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>>
  ! CHECK-DAG: %[[a3:.*]] = fir.alloca !fir.array<3x3xi32> {bindc_name = "field", uniq_name = "_QMtest2Funpack_testEfield"}
  ! CHECK-DAG: %[[FIELD_DECL:.*]]:2 = hlfir.declare %[[a3]]{{.*}}{uniq_name = "_QMtest2Funpack_testEfield"}
  ! CHECK-DAG: %[[a4:.*]] = fir.alloca !fir.array<3x3x!fir.logical<4>> {bindc_name = "mask", uniq_name = "_QMtest2Funpack_testEmask"}
  ! CHECK-DAG: %[[MASK_DECL:.*]]:2 = hlfir.declare %[[a4]]{{.*}}{uniq_name = "_QMtest2Funpack_testEmask"}
  ! CHECK-DAG: %[[a5:.*]] = fir.alloca !fir.array<3x3xi32> {bindc_name = "result", uniq_name = "_QMtest2Funpack_testEresult"}
  ! CHECK-DAG: %[[RESULT_DECL:.*]]:2 = hlfir.declare %[[a5]]{{.*}}{uniq_name = "_QMtest2Funpack_testEresult"}
  ! CHECK-DAG: %[[a6:.*]] = fir.alloca !fir.array<3xi32> {bindc_name = "vector", uniq_name = "_QMtest2Funpack_testEvector"}
  ! CHECK-DAG: %[[VECTOR_DECL:.*]]:2 = hlfir.declare %[[a6]]{{.*}}{uniq_name = "_QMtest2Funpack_testEvector"}
  ! CHECK: %[[v_embox:.*]] = fir.embox %[[VECTOR_DECL]]#0(%{{.*}}) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
  ! CHECK: %[[m_embox:.*]] = fir.embox %[[MASK_DECL]]#0(%{{.*}}) : (!fir.ref<!fir.array<3x3x!fir.logical<4>>>, !fir.shape<2>) -> !fir.box<!fir.array<3x3x!fir.logical<4>>>
  ! CHECK: %[[f_embox:.*]] = fir.embox %[[FIELD_DECL]]#0(%{{.*}}) : (!fir.ref<!fir.array<3x3xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<3x3xi32>>
  ! CHECK: %[[zero:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xi32>>
  ! CHECK: %[[zero_embox:.*]] = fir.embox %[[zero]](%{{.*}}) : (!fir.heap<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xi32>>>
  ! CHECK: fir.store %[[zero_embox]] to %[[a2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  ! CHECK: %[[r_arg:.*]] = fir.convert %[[a2]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: %[[v_arg:.*]] = fir.convert %[[v_embox]] : (!fir.box<!fir.array<3xi32>>) -> !fir.box<none>
  ! CHECK: %[[m_arg:.*]] = fir.convert %[[m_embox]] : (!fir.box<!fir.array<3x3x!fir.logical<4>>>) -> !fir.box<none>
  ! CHECK: %[[f_arg:.*]] = fir.convert %[[f_embox]] : (!fir.box<!fir.array<3x3xi32>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAUnpack(%[[r_arg]], %[[v_arg]], %[[m_arg]], %[[f_arg]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
  ! CHECK: %[[r_load:.*]] = fir.load %[[a2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  ! CHECK: %[[r_addr:.*]] = fir.box_addr %[[r_load]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>) -> !fir.heap<!fir.array<?x?xi32>>
  ! CHECK: %[[r_decl:.*]]:2 = hlfir.declare %[[r_addr]](%{{.*}}) {uniq_name = ".tmp.intrinsic_result"}
  ! CHECK: %[[r_expr:.*]] = hlfir.as_expr %[[r_decl]]#0 move %{{.*}} : (!fir.box<!fir.array<?x?xi32>>, i1) -> !hlfir.expr<?x?xi32>
  ! CHECK: hlfir.assign %[[r_expr]] to %[[RESULT_DECL]]#0 : !hlfir.expr<?x?xi32>, !fir.ref<!fir.array<3x3xi32>>
  ! CHECK: hlfir.destroy %[[r_expr]] : !hlfir.expr<?x?xi32>
  ! CHECK: %[[c343:.*]] = arith.constant 343 : i32
  ! CHECK: %[[v_embox2:.*]] = fir.embox %[[VECTOR_DECL]]#0(%{{.*}}) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
  ! CHECK: %[[m_embox2:.*]] = fir.embox %[[MASK_DECL]]#0(%{{.*}}) : (!fir.ref<!fir.array<3x3x!fir.logical<4>>>, !fir.shape<2>) -> !fir.box<!fir.array<3x3x!fir.logical<4>>>
  ! CHECK: fir.store %[[c343]] to %[[a1]] : !fir.ref<i32>
  ! CHECK: %[[scalar_embox:.*]] = fir.embox %[[a1]] : (!fir.ref<i32>) -> !fir.box<i32>
  result = unpack(vector, mask, 343)
  ! CHECK: fir.call @_FortranAUnpack(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
  ! CHECK: return
end subroutine unpack_test

end module
