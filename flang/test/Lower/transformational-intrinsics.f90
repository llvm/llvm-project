! Test how transformational intrinsic function references are lowered

! RUN: bbc -emit-fir %s -o - | FileCheck %s

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
  ! CHECK: %[[res_desc:.]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>
  ! CHECK-DAG: %[[res_arg:.*]] = fir.convert %[[res_desc]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[x_arg:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?x!fir.logical<1>>>) -> !fir.box<none>
  ! CHECK: fir.call @_Fortran{{.*}}AllDim(%[[res_arg]], %[[x_arg]], {{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  ! CHECK: %[[res_desc_load:.*]] = fir.load %[[res_desc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[res_desc_load]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[res_addr:.*]] = fir.box_addr %[[res_desc_load]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>) -> !fir.heap<!fir.array<?x!fir.logical<1>>>
  ! CHECK-DAG: %[[res_shape:.*]] = fir.shape_shift %[[dims]]#0, %[[dims]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[io_embox:.*]] = fir.embox %[[res_addr]](%[[res_shape]]) : (!fir.heap<!fir.array<?x!fir.logical<1>>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?x!fir.logical<1>>>
  ! CHECK: %[[io_embox_cast:.*]] = fir.convert %[[io_embox]] : (!fir.box<!fir.array<?x!fir.logical<1>>>) -> !fir.box<none>
  ! CHECK: fir.call @_Fortran{{.*}}ioOutputDescriptor({{.*}}, %[[io_embox_cast]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
  print *, all(x, 1)
  ! CHECK: fir.freemem %[[res_addr]] : !fir.heap<!fir.array<?x!fir.logical<1>>>
end subroutine

! CHECK-LABEL: func @_QMtest2Pin_call(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<1>>>{{.*}}) {
subroutine in_call(x)
  implicit none
  logical(1) :: x(:, :)
  ! CHECK: %[[res_desc:.]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>
  ! CHECK-DAG: %[[res_arg:.*]] = fir.convert %[[res_desc]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[x_arg:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?x!fir.logical<1>>>) -> !fir.box<none>
  ! CHECK: fir.call @_Fortran{{.*}}AllDim(%[[res_arg]], %[[x_arg]], {{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  ! CHECK: %[[res_desc_load:.*]] = fir.load %[[res_desc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[res_desc_load]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[res_addr:.*]] = fir.box_addr %[[res_desc_load]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>) -> !fir.heap<!fir.array<?x!fir.logical<1>>>
  ! CHECK-DAG: %[[res_shape:.*]] = fir.shape_shift %[[dims]]#0, %[[dims]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[call_embox:.*]] = fir.embox %[[res_addr]](%[[res_shape]]) : (!fir.heap<!fir.array<?x!fir.logical<1>>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?x!fir.logical<1>>>
  ! CHECK: fir.call @_QPtakes_array_desc(%[[call_embox]]) : (!fir.box<!fir.array<?x!fir.logical<1>>>) -> ()
  call takes_array_desc(all(x, 1))
  ! CHECK: fir.freemem %[[res_addr]] : !fir.heap<!fir.array<?x!fir.logical<1>>>
end subroutine

! CHECK-LABEL: func @_QMtest2Pin_implicit_call(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<1>>>{{.*}}) {
subroutine in_implicit_call(x)
  logical(1) :: x(:, :)
  ! CHECK: %[[res_desc:.]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>
  ! CHECK: fir.call @_Fortran{{.*}}AllDim
  ! CHECK: %[[res_desc_load:.*]] = fir.load %[[res_desc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>
  ! CHECK: %[[res_addr:.*]] = fir.box_addr %[[res_desc_load]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>) -> !fir.heap<!fir.array<?x!fir.logical<1>>>
  ! CHECK: %[[res_addr_cast:.*]] = fir.convert %[[res_addr]] : (!fir.heap<!fir.array<?x!fir.logical<1>>>) -> !fir.ref<!fir.array<?x!fir.logical<1>>>
  ! CHECK: fir.call @_QPtakes_implicit_array(%[[res_addr_cast]]) : (!fir.ref<!fir.array<?x!fir.logical<1>>>) -> ()
  call takes_implicit_array(all(x, 1))
  ! CHECK: fir.freemem %[[res_addr]] : !fir.heap<!fir.array<?x!fir.logical<1>>>
end subroutine

! CHECK-LABEL: func @_QMtest2Pin_assignment(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<1>>>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?x!fir.logical<1>>>{{.*}})
subroutine in_assignment(x, y)
  logical(1) :: x(:, :), y(:)
  ! CHECK: %[[res_desc:.]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>
  ! CHECK: %[[y_load:.*]] = fir.array_load %[[arg1]] : (!fir.box<!fir.array<?x!fir.logical<1>>>) -> !fir.array<?x!fir.logical<1>>
  ! CHECK: fir.call @_Fortran{{.*}}AllDim

  ! CHECK: %[[res_desc_load:.*]] = fir.load %[[res_desc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[res_desc_load]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[res_addr:.*]] = fir.box_addr %[[res_desc_load]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>) -> !fir.heap<!fir.array<?x!fir.logical<1>>>
  ! CHECK-DAG: %[[res_shape:.*]] = fir.shape_shift %[[dims]]#0, %[[dims]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[res_load:.*]] = fir.array_load %[[res_addr]](%[[res_shape]]) : (!fir.heap<!fir.array<?x!fir.logical<1>>>, !fir.shapeshift<1>) -> !fir.array<?x!fir.logical<1>>

  ! CHECK: %[[assign:.*]] = fir.do_loop %[[idx:.*]] = %{{.*}} to {{.*}} {
    ! CHECK: %[[res_elt:.*]] = fir.array_fetch %[[res_load]], %[[idx]] : (!fir.array<?x!fir.logical<1>>, index) -> !fir.logical<1>
    ! CHECK: fir.array_update %{{.*}} %[[res_elt]], %[[idx]] : (!fir.array<?x!fir.logical<1>>, !fir.logical<1>, index) -> !fir.array<?x!fir.logical<1>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[y_load]], %[[assign]] to %[[arg1]] : !fir.array<?x!fir.logical<1>>, !fir.array<?x!fir.logical<1>>, !fir.box<!fir.array<?x!fir.logical<1>>>
  y = all(x, 1)
  ! CHECK: fir.freemem %[[res_addr]] : !fir.heap<!fir.array<?x!fir.logical<1>>>
end subroutine

! CHECK-LABEL: func @_QMtest2Pin_elem_expr(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<1>>>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?x!fir.logical<1>>>{{.*}}, %[[arg2:.*]]: !fir.box<!fir.array<?x!fir.logical<1>>>{{.*}})
subroutine in_elem_expr(x, y, z)
  logical(1) :: x(:, :), y(:), z(:)
  ! CHECK: %[[res_desc:.]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>
  ! CHECK-DAG: %[[y_load:.*]] = fir.array_load %[[arg1]] : (!fir.box<!fir.array<?x!fir.logical<1>>>) -> !fir.array<?x!fir.logical<1>>
  ! CHECK-DAG: %[[z_load:.*]] = fir.array_load %[[arg2]] : (!fir.box<!fir.array<?x!fir.logical<1>>>) -> !fir.array<?x!fir.logical<1>>
  ! CHECK: fir.call @_Fortran{{.*}}AllDim

  ! CHECK: %[[res_desc_load:.*]] = fir.load %[[res_desc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[res_desc_load]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[res_addr:.*]] = fir.box_addr %[[res_desc_load]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>) -> !fir.heap<!fir.array<?x!fir.logical<1>>>
  ! CHECK-DAG: %[[res_shape:.*]] = fir.shape_shift %[[dims]]#0, %[[dims]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[res_load:.*]] = fir.array_load %[[res_addr]](%[[res_shape]]) : (!fir.heap<!fir.array<?x!fir.logical<1>>>, !fir.shapeshift<1>) -> !fir.array<?x!fir.logical<1>>

  ! CHECK: %[[elem_expr:.*]] = fir.do_loop %[[idx:.*]] = %{{.*}} to {{.*}} {
    ! CHECK-DAG: %[[y_elt:.*]] = fir.array_fetch %[[y_load]], %[[idx]] : (!fir.array<?x!fir.logical<1>>, index) -> !fir.logical<1>
    ! CHECK-DAG: %[[res_elt:.*]] = fir.array_fetch %[[res_load]], %[[idx]] : (!fir.array<?x!fir.logical<1>>, index) -> !fir.logical<1>
    ! CHECK-DAG: %[[y_elt_i1:.*]] = fir.convert %[[y_elt]] : (!fir.logical<1>) -> i1
    ! CHECK-DAG: %[[res_elt_i1:.*]] = fir.convert %[[res_elt]] : (!fir.logical<1>) -> i1
    ! CHECK: %[[neqv_i1:.*]] = arith.cmpi ne, %[[y_elt_i1]], %[[res_elt_i1]] : i1
    ! CHECK: %[[neqv:.*]] = fir.convert %[[neqv_i1]] : (i1) -> !fir.logical<1>
    ! CHECK: fir.array_update %{{.*}} %[[neqv]], %[[idx]] : (!fir.array<?x!fir.logical<1>>, !fir.logical<1>, index) -> !fir.array<?x!fir.logical<1>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[z_load]], %[[elem_expr]] to %[[arg2]] : !fir.array<?x!fir.logical<1>>, !fir.array<?x!fir.logical<1>>, !fir.box<!fir.array<?x!fir.logical<1>>>
  z = y .neqv. all(x, 1)
  ! CHECK: fir.freemem %[[res_addr]] : !fir.heap<!fir.array<?x!fir.logical<1>>>
end subroutine

! CSHIFT

  ! CHECK-LABEL: func @_QMtest2Pcshift_test() {
  ! CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK:         %[[VAL_1:.*]] = fir.alloca i32
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>>
  ! CHECK:         %[[VAL_3:.*]] = arith.constant 3 : index
  ! CHECK:         %[[VAL_4:.*]] = arith.constant 3 : index
  ! CHECK:         %[[VAL_5:.*]] = fir.alloca !fir.array<3x3xi32> {bindc_name = "array", uniq_name = "_QMtest2Fcshift_testEarray"}
  ! CHECK:         %[[VAL_6:.*]] = arith.constant 3 : index
  ! CHECK:         %[[VAL_7:.*]] = arith.constant 3 : index
  ! CHECK:         %[[VAL_8:.*]] = fir.alloca !fir.array<3x3xi32> {bindc_name = "result", uniq_name = "_QMtest2Fcshift_testEresult"}
  ! CHECK:         %[[VAL_9:.*]] = arith.constant 3 : index
  ! CHECK:         %[[VAL_10:.*]] = fir.alloca !fir.array<3xi32> {bindc_name = "shift", uniq_name = "_QMtest2Fcshift_testEshift"}
  ! CHECK:         %[[VAL_11:.*]] = arith.constant 6 : index
  ! CHECK:         %[[VAL_12:.*]] = fir.alloca !fir.array<6xi32> {bindc_name = "vector", uniq_name = "_QMtest2Fcshift_testEvector"}
  ! CHECK:         %[[VAL_13:.*]] = arith.constant 6 : index
  ! CHECK:         %[[VAL_14:.*]] = fir.alloca !fir.array<6xi32> {bindc_name = "vectorresult", uniq_name = "_QMtest2Fcshift_testEvectorresult"}
  ! CHECK:         %[[VAL_15:.*]] = fir.shape %[[VAL_6]], %[[VAL_7]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_16:.*]] = fir.array_load %[[VAL_8]](%[[VAL_15]]) : (!fir.ref<!fir.array<3x3xi32>>, !fir.shape<2>) -> !fir.array<3x3xi32>
  ! CHECK:         %[[VAL_17:.*]] = arith.constant -2 : i32
  ! CHECK:         %[[VAL_18:.*]] = fir.shape %[[VAL_3]], %[[VAL_4]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_19:.*]] = fir.embox %[[VAL_5]](%[[VAL_18]]) : (!fir.ref<!fir.array<3x3xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<3x3xi32>>
  ! CHECK:         %[[VAL_20:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xi32>>
  ! CHECK:         %[[VAL_21:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_22:.*]] = fir.shape %[[VAL_21]], %[[VAL_21]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_23:.*]] = fir.embox %[[VAL_20]](%[[VAL_22]]) : (!fir.heap<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xi32>>>
  ! CHECK:         fir.store %[[VAL_23]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  ! CHECK:         %[[VAL_24:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_25:.*]] = fir.embox %[[VAL_10]](%[[VAL_24]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
  ! CHECK:         %[[VAL_26:.*]] = fir.address_of(@_QQcl{{.*}}) : !fir.ref<!fir.char<1,
  ! CHECK:         %[[VAL_27:.*]] = arith.constant {{[0-9]+}} : i32
  ! CHECK:         %[[VAL_28:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:         %[[VAL_29:.*]] = fir.convert %[[VAL_19]] : (!fir.box<!fir.array<3x3xi32>>) -> !fir.box<none>
  ! CHECK:         %[[VAL_30:.*]] = fir.convert %[[VAL_25]] : (!fir.box<!fir.array<3xi32>>) -> !fir.box<none>
  ! CHECK:         %[[VAL_31:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,{{[0-9]+}}>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_32:.*]] = fir.call @_FortranACshift(%[[VAL_28]], %[[VAL_29]], %[[VAL_30]], %[[VAL_17]], %[[VAL_31]], %[[VAL_27]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  ! CHECK:         %[[VAL_33:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  ! CHECK:         %[[VAL_34:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_35:.*]]:3 = fir.box_dims %[[VAL_33]], %[[VAL_34]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>, index) -> (index, index, index)
  ! CHECK:         %[[VAL_36:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_37:.*]]:3 = fir.box_dims %[[VAL_33]], %[[VAL_36]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>, index) -> (index, index, index)
  ! CHECK:         %[[VAL_38:.*]] = fir.box_addr %[[VAL_33]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>) -> !fir.heap<!fir.array<?x?xi32>>
  ! CHECK:         %[[VAL_39:.*]] = fir.shape_shift %[[VAL_35]]#0, %[[VAL_35]]#1, %[[VAL_37]]#0, %[[VAL_37]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
  ! CHECK:         %[[VAL_40:.*]] = fir.array_load %[[VAL_38]](%[[VAL_39]]) : (!fir.heap<!fir.array<?x?xi32>>, !fir.shapeshift<2>) -> !fir.array<?x?xi32>
  ! CHECK:         %[[VAL_41:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_42:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_43:.*]] = arith.subi %[[VAL_6]], %[[VAL_41]] : index
  ! CHECK:         %[[VAL_44:.*]] = arith.subi %[[VAL_7]], %[[VAL_41]] : index
  ! CHECK:         %[[VAL_45:.*]] = fir.do_loop %[[VAL_46:.*]] = %[[VAL_42]] to %[[VAL_44]] step %[[VAL_41]] unordered iter_args(%[[VAL_47:.*]] = %[[VAL_16]]) -> (!fir.array<3x3xi32>) {
  ! CHECK:           %[[VAL_48:.*]] = fir.do_loop %[[VAL_49:.*]] = %[[VAL_42]] to %[[VAL_43]] step %[[VAL_41]] unordered iter_args(%[[VAL_50:.*]] = %[[VAL_47]]) -> (!fir.array<3x3xi32>) {
  ! CHECK:             %[[VAL_51:.*]] = fir.array_fetch %[[VAL_40]], %[[VAL_49]], %[[VAL_46]] : (!fir.array<?x?xi32>, index, index) -> i32
  ! CHECK:             %[[VAL_52:.*]] = fir.array_update %[[VAL_50]], %[[VAL_51]], %[[VAL_49]], %[[VAL_46]] : (!fir.array<3x3xi32>, i32, index, index) -> !fir.array<3x3xi32>
  ! CHECK:             fir.result %[[VAL_52]] : !fir.array<3x3xi32>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_53:.*]] : !fir.array<3x3xi32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_16]], %[[VAL_54:.*]] to %[[VAL_8]] : !fir.array<3x3xi32>, !fir.array<3x3xi32>, !fir.ref<!fir.array<3x3xi32>>
  ! CHECK:         fir.freemem %[[VAL_38]] : !fir.heap<!fir.array<?x?xi32>>
  ! CHECK:         %[[VAL_55:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_56:.*]] = fir.array_load %[[VAL_14]](%[[VAL_55]]) : (!fir.ref<!fir.array<6xi32>>, !fir.shape<1>) -> !fir.array<6xi32>
  ! CHECK:         %[[VAL_57:.*]] = arith.constant 3 : i32
  ! CHECK:         fir.store %[[VAL_57]] to %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:         %[[VAL_58:.*]] = fir.shape %[[VAL_11]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_59:.*]] = fir.embox %[[VAL_12]](%[[VAL_58]]) : (!fir.ref<!fir.array<6xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<6xi32>>
  ! CHECK:         %[[VAL_60:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  ! CHECK:         %[[VAL_61:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_62:.*]] = fir.shape %[[VAL_61]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_63:.*]] = fir.embox %[[VAL_60]](%[[VAL_62]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK:         fir.store %[[VAL_63]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK:         %[[VAL_64:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:         %[[VAL_65:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,
  ! CHECK:         %[[VAL_66:.*]] = arith.constant {{[0-9]+}} : i32
  ! CHECK:         %[[VAL_67:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:         %[[VAL_68:.*]] = fir.convert %[[VAL_59]] : (!fir.box<!fir.array<6xi32>>) -> !fir.box<none>
  ! CHECK:         %[[VAL_69:.*]] = fir.convert %[[VAL_64]] : (i32) -> i64
  ! CHECK:         %[[VAL_70:.*]] = fir.convert %[[VAL_65]] : (!fir.ref<!fir.char<1,{{[0-9]+}}>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_71:.*]] = fir.call @_FortranACshiftVector(%[[VAL_67]], %[[VAL_68]], %[[VAL_69]], %[[VAL_70]], %[[VAL_66]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i64, !fir.ref<i8>, i32) -> none
  ! CHECK:         %[[VAL_72:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK:         %[[VAL_73:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_74:.*]]:3 = fir.box_dims %[[VAL_72]], %[[VAL_73]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
  ! CHECK:         %[[VAL_75:.*]] = fir.box_addr %[[VAL_72]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK:         %[[VAL_76:.*]] = fir.shape_shift %[[VAL_74]]#0, %[[VAL_74]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK:         %[[VAL_77:.*]] = fir.array_load %[[VAL_75]](%[[VAL_76]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.array<?xi32>
  ! CHECK:         %[[VAL_78:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_79:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_80:.*]] = arith.subi %[[VAL_13]], %[[VAL_78]] : index
  ! CHECK:         %[[VAL_81:.*]] = fir.do_loop %[[VAL_82:.*]] = %[[VAL_79]] to %[[VAL_80]] step %[[VAL_78]] unordered iter_args(%[[VAL_83:.*]] = %[[VAL_56]]) -> (!fir.array<6xi32>) {
  ! CHECK:           %[[VAL_84:.*]] = fir.array_fetch %[[VAL_77]], %[[VAL_82]] : (!fir.array<?xi32>, index) -> i32
  ! CHECK:           %[[VAL_85:.*]] = fir.array_update %[[VAL_83]], %[[VAL_84]], %[[VAL_82]] : (!fir.array<6xi32>, i32, index) -> !fir.array<6xi32>
  ! CHECK:           fir.result %[[VAL_85]] : !fir.array<6xi32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_56]], %[[VAL_86:.*]] to %[[VAL_14]] : !fir.array<6xi32>, !fir.array<6xi32>, !fir.ref<!fir.array<6xi32>>
  ! CHECK:         fir.freemem %[[VAL_75]] : !fir.heap<!fir.array<?xi32>>
  ! CHECK:         return
  ! CHECK:       }

subroutine cshift_test()
  integer, dimension (3, 3) :: array
  integer, dimension(3) :: shift
  integer, dimension(3, 3) :: result
  integer, dimension(6) :: vectorResult
  integer, dimension (6) :: vector
  result = cshift(array, shift, -2) ! non-vector case
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
  ! CHECK-DAG: %[[a4:.*]] = fir.alloca !fir.array<3x3x!fir.logical<4>> {bindc_name = "mask", uniq_name = "_QMtest2Funpack_testEmask"}
  ! CHECK-DAG: %[[a5:.*]] = fir.alloca !fir.array<3x3xi32> {bindc_name = "result", uniq_name = "_QMtest2Funpack_testEresult"}
  ! CHECK-DAG: %[[a6:.*]] = fir.alloca !fir.array<3xi32> {bindc_name = "vector", uniq_name = "_QMtest2Funpack_testEvector"}
  ! CHECK: %[[a7:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK-NEXT: %[[a8:.*]] = fir.array_load %[[a5]](%[[a7]]) : (!fir.ref<!fir.array<3x3xi32>>, !fir.shape<2>) -> !fir.array<3x3xi32>
  ! CHECK: %[[a9:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[a10:.*]] = fir.embox %[[a6]](%[[a9]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
  ! CHECK: %[[a11:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[a12:.*]] = fir.embox %[[a4]](%[[a11]]) : (!fir.ref<!fir.array<3x3x!fir.logical<4>>>, !fir.shape<2>) -> !fir.box<!fir.array<3x3x!fir.logical<4>>>
  ! CHECK: %[[a13:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK-NEXT: %[[a14:.*]] = fir.embox %[[a3]](%[[a13]]) : (!fir.ref<!fir.array<3x3xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<3x3xi32>>
  ! CHECK: %[[a15:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xi32>>
  ! CHECK: %[[a16:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[a17:.*]] = fir.embox %[[a15]](%[[a16]]) : (!fir.heap<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xi32>>>
  ! CHECK-NEXT: fir.store %[[a17]] to %[[a2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  ! CHECK-DAG: %[[a19:.*]] = fir.convert %[[a2]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[a20:.*]] = fir.convert %[[a10]] : (!fir.box<!fir.array<3xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[a21:.*]] = fir.convert %[[a12]] : (!fir.box<!fir.array<3x3x!fir.logical<4>>>) -> !fir.box<none>
  ! CHECK-DAG: %[[a22:.*]] = fir.convert %[[a14]] : (!fir.box<!fir.array<3x3xi32>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAUnpack(%[[a19]], %[[a20]], %[[a21]], %[[a22]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  ! CHECK-NEXT: %[[a22:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  ! CHECK: %[[a25:.*]] = fir.box_addr %[[a22]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>) -> !fir.heap<!fir.array<?x?xi32>>
  ! CHECK: fir.freemem %[[a25]] : !fir.heap<!fir.array<?x?xi32>>
  ! CHECK: %[[a36:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[a38:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
  ! CHECK-NEXT: %[[a39:.*]] = fir.embox %[[a6]](%[[a38]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
  ! CHECK: %[[a40:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK-NEXT: %[[a41:.*]] = fir.embox %[[a4]](%[[a40]]) : (!fir.ref<!fir.array<3x3x!fir.logical<4>>>, !fir.shape<2>) -> !fir.box<!fir.array<3x3x!fir.logical<4>>>
  ! CHECK: %[[a42:.*]] = fir.embox %[[a1]] : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK: %[[a43:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xi32>>
  ! CHECK: %[[a44:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[a45:.*]] = fir.embox %[[a43]](%[[a44]]) : (!fir.heap<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xi32>>>
  ! CHECK-NEXT: fir.store %[[a45]] to %[[a0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  ! CHECK: %[[a47:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: %[[a48:.*]] = fir.convert %[[a39]] : (!fir.box<!fir.array<3xi32>>) -> !fir.box<none>
  ! CHECK: %[[a49:.*]] = fir.convert %[[a41]] : (!fir.box<!fir.array<3x3x!fir.logical<4>>>) -> !fir.box<none>
  ! CHECK: %[[a50:.*]] = fir.convert %[[a42]] : (!fir.box<i32>) -> !fir.box<none>
  result = unpack(vector, mask, 343)
  ! CHECK: fir.call @_FortranAUnpack(%[[a47]], %[[a48]], %[[a49]], %[[a50]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  ! CHECK: %[[a53:.*]] = fir.load %[[a0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  ! CHECK: %[[a56:.*]] = fir.box_addr %[[a53]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>) -> !fir.heap<!fir.array<?x?xi32>>
  ! CHECK: fir.freemem %[[a56]] : !fir.heap<!fir.array<?x?xi32>>
  ! CHECK-NEXT: return
end subroutine unpack_test

end module
