! Test C_DEVPTR_EQ and C_DEVPTR_NE lowering.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

function test_c_devptr_eq(d1, d2)
  use __fortran_builtins, only: &
      c_devptr => __builtin_c_devptr, operator(==), operator(/=)
  type(c_devptr), intent(in) :: d1, d2
  logical :: test_c_devptr_eq
  test_c_devptr_eq = (d1 .eq. d2)
end

! CHECK-LABEL: func.func @_QPtest_c_devptr_eq(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_devptr{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>> {fir.bindc_name = "d1"}, %[[ARG1:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_devptr{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>> {fir.bindc_name = "d2"}) -> !fir.logical<4> {
! CHECK: %[[DECL_ARG0:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest_c_devptr_eqEd1"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_devptr{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_devptr{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_devptr{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>)
! CHECK: %[[DECL_ARG1:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest_c_devptr_eqEd2"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_devptr{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_devptr{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_devptr{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>)
! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.logical<4> {bindc_name = "test_c_devptr_eq", uniq_name = "_QFtest_c_devptr_eqEtest_c_devptr_eq"}
! CHECK: %[[DECL_RET:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "_QFtest_c_devptr_eqEtest_c_devptr_eq"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK: %[[COORD_CPTR0:.*]] = fir.coordinate_of %[[DECL_ARG0]]#0, cptr : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_devptr{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK: %[[COORD_ADDRESS0:.*]] = fir.coordinate_of %[[COORD_CPTR0]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK: %[[ADDRESS0:.*]] = fir.load %[[COORD_ADDRESS0]] : !fir.ref<i64>
! CHECK: %[[COORD_CPTR1:.*]] = fir.coordinate_of %[[DECL_ARG1]]#0, cptr : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_devptr{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK: %[[COORD_ADDRESS1:.*]] = fir.coordinate_of %[[COORD_CPTR1]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK: %[[ADDRESS1:.*]] = fir.load %[[COORD_ADDRESS1]] : !fir.ref<i64>
! CHECK: %[[CMP:.*]] = arith.cmpi eq, %[[ADDRESS0]], %[[ADDRESS1]] : i64
! CHECK: %[[RES:.*]] = fir.convert %[[CMP]] : (i1) -> !fir.logical<4>
! CHECK: %[[NO_REASSOC_RES:.*]] = hlfir.no_reassoc %[[RES]] : !fir.logical<4>
! CHECK: hlfir.assign %[[NO_REASSOC_RES]] to %[[DECL_RET]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK: %[[LOAD_RET:.*]] = fir.load %[[DECL_RET]]#0 : !fir.ref<!fir.logical<4>>
! CHECK: return %[[LOAD_RET]] : !fir.logical<4>
! CHECK: }

function test_c_devptr_ne(d1, d2)
  use __fortran_builtins, only: &
      c_devptr => __builtin_c_devptr, operator(==), operator(/=)
  type(c_devptr), intent(in) :: d1, d2
  logical :: test_c_devptr_ne
  test_c_devptr_ne = (d1 .ne. d2)
end

! CHECK-LABEL: func.func @_QPtest_c_devptr_ne(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_devptr{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>> {fir.bindc_name = "d1"}, %[[ARG1:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_devptr{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>> {fir.bindc_name = "d2"}) -> !fir.logical<4> {
! CHECK: %[[DECL_ARG0:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest_c_devptr_neEd1"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_devptr{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_devptr{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_devptr{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>)
! CHECK: %[[DECL_ARG1:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest_c_devptr_neEd2"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_devptr{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_devptr{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_devptr{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>)
! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.logical<4> {bindc_name = "test_c_devptr_ne", uniq_name = "_QFtest_c_devptr_neEtest_c_devptr_ne"}
! CHECK: %[[DECL_RET:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "_QFtest_c_devptr_neEtest_c_devptr_ne"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK: %[[COORD_CPTR0:.*]] = fir.coordinate_of %[[DECL_ARG0]]#0, cptr : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_devptr{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK: %[[COORD_ADDRESS0:.*]] = fir.coordinate_of %[[COORD_CPTR0]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK: %[[ADDRESS0:.*]] = fir.load %[[COORD_ADDRESS0]] : !fir.ref<i64>
! CHECK: %[[COORD_CPTR1:.*]] = fir.coordinate_of %[[DECL_ARG1]]#0, cptr : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_devptr{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK: %[[COORD_ADDRESS1:.*]] = fir.coordinate_of %[[COORD_CPTR1]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK: %[[ADDRESS1:.*]] = fir.load %[[COORD_ADDRESS1]] : !fir.ref<i64>
! CHECK: %[[CMP:.*]] = arith.cmpi ne, %[[ADDRESS0]], %[[ADDRESS1]] : i64
! CHECK: %[[RES:.*]] = fir.convert %[[CMP]] : (i1) -> !fir.logical<4>
! CHECK: %[[NO_REASSOC_RES:.*]] = hlfir.no_reassoc %[[RES]] : !fir.logical<4>
! CHECK: hlfir.assign %[[NO_REASSOC_RES]] to %[[DECL_RET]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK: %[[LOAD_RET:.*]] = fir.load %[[DECL_RET]]#0 : !fir.ref<!fir.logical<4>>
! CHECK: return %[[LOAD_RET]] : !fir.logical<4>
! CHECK: }
