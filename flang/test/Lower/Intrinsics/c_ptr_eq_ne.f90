! Test C_PTR_EQ and C_PTR_NE lowering.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

function test_c_ptr_eq(ptr1, ptr2)
  use, intrinsic :: iso_c_binding
  type(c_ptr), intent(in) :: ptr1, ptr2
  logical :: test_c_ptr_eq
  test_c_ptr_eq = (ptr1 .eq. ptr2)
end

! CHECK-LABEL: func.func @_QPtest_c_ptr_eq(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "ptr1"}, %[[ARG1:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "ptr2"}) -> !fir.logical<4> {
! CHECK: %[[DECL_ARG0:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest_c_ptr_eqEptr1"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)
! CHECK: %[[DECL_ARG1:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest_c_ptr_eqEptr2"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)
! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.logical<4> {bindc_name = "test_c_ptr_eq", uniq_name = "_QFtest_c_ptr_eqEtest_c_ptr_eq"}
! CHECK: %[[DECL_RET:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "_QFtest_c_ptr_eqEtest_c_ptr_eq"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK: %[[FIELD_ADDRESS:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK: %[[COORD_ADDRESS0:.*]] = fir.coordinate_of %[[DECL_ARG0]]#1, %[[FIELD_ADDRESS]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK: %[[ADDRESS0:.*]] = fir.load %[[COORD_ADDRESS0]] : !fir.ref<i64>
! CHECK: %[[FIELD_ADDRESS:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK: %[[COORD_ADDRESS1:.*]] = fir.coordinate_of %[[DECL_ARG1]]#1, %[[FIELD_ADDRESS]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK: %[[ADDRESS1:.*]] = fir.load %[[COORD_ADDRESS1]] : !fir.ref<i64>
! CHECK: %[[CMP:.*]] = arith.cmpi eq, %[[ADDRESS0]], %[[ADDRESS1]] : i64
! CHECK: %[[RES:.*]] = fir.convert %[[CMP]] : (i1) -> !fir.logical<4>
! CHECK: %[[NO_REASSOC_RES:.*]] = hlfir.no_reassoc %[[RES]] : !fir.logical<4>
! CHECK: hlfir.assign %[[NO_REASSOC_RES]] to %[[DECL_RET]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK: %[[LOAD_RET:.*]] = fir.load %[[DECL_RET]]#1 : !fir.ref<!fir.logical<4>>
! CHECK: return %[[LOAD_RET]] : !fir.logical<4>
! CHECK: }

function test_c_ptr_ne(ptr1, ptr2)
  use, intrinsic :: iso_c_binding
  type(c_ptr), intent(in) :: ptr1, ptr2
  logical :: test_c_ptr_ne
  test_c_ptr_ne = (ptr1 .ne. ptr2)
end

! CHECK-LABEL: func.func @_QPtest_c_ptr_ne(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "ptr1"}, %[[ARG1:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "ptr2"}) -> !fir.logical<4> {
! CHECK: %[[DECL_ARG0:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest_c_ptr_neEptr1"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)
! CHECK: %[[DECL_ARG1:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest_c_ptr_neEptr2"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)
! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.logical<4> {bindc_name = "test_c_ptr_ne", uniq_name = "_QFtest_c_ptr_neEtest_c_ptr_ne"}
! CHECK: %[[DECL_RET:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "_QFtest_c_ptr_neEtest_c_ptr_ne"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK: %[[FIELD_ADDRESS:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK: %[[COORD_ADDRESS0:.*]] = fir.coordinate_of %[[DECL_ARG0]]#1, %[[FIELD_ADDRESS]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK: %[[ADDRESS0:.*]] = fir.load %[[COORD_ADDRESS0]] : !fir.ref<i64>
! CHECK: %[[FIELD_ADDRESS:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK: %[[COORD_ADDRESS1:.*]] = fir.coordinate_of %[[DECL_ARG1]]#1, %[[FIELD_ADDRESS]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK: %[[ADDRESS1:.*]] = fir.load %[[COORD_ADDRESS1]] : !fir.ref<i64>
! CHECK: %[[CMP:.*]] = arith.cmpi ne, %[[ADDRESS0]], %[[ADDRESS1]] : i64
! CHECK: %[[RES:.*]] = fir.convert %[[CMP]] : (i1) -> !fir.logical<4>
! CHECK: %[[NO_REASSOC_RES:.*]] = hlfir.no_reassoc %[[RES]] : !fir.logical<4>
! CHECK: hlfir.assign %[[NO_REASSOC_RES]] to %[[DECL_RET]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK: %[[LOAD_RET:.*]] = fir.load %[[DECL_RET]]#1 : !fir.ref<!fir.logical<4>>
! CHECK: return %[[LOAD_RET]] : !fir.logical<4>
! CHECK: }
