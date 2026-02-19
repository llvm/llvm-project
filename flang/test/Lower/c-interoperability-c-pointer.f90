! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest(
! CHECK-SAME:                     %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "ptr1"},
! CHECK-SAME:                     %[[VAL_1:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>> {fir.bindc_name = "ptr2"}) {
! CHECK:         %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:         %[[VAL_131:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] arg 1 {uniq_name = "_QFtestEptr1"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)
! CHECK:         %[[VAL_132:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] arg 2 {uniq_name = "_QFtestEptr2"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>)
! CHECK:         %[[VAL_133:.*]] = fir.coordinate_of %[[VAL_131]]#0, __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[VAL_134:.*]] = fir.load %[[VAL_133]] : !fir.ref<i64>
! CHECK:         %[[VAL_135:.*]] = fir.convert %[[VAL_134]] : (i64) -> !fir.ref<i64>
! CHECK:         %[[VAL_136:.*]] = fir.coordinate_of %[[VAL_132]]#0, __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[VAL_137:.*]] = fir.load %[[VAL_136]] : !fir.ref<i64>
! CHECK:         %[[VAL_138:.*]] = fir.convert %[[VAL_137]] : (i64) -> !fir.ref<i64>
! CHECK:         fir.call @c_func(%[[VAL_135]], %[[VAL_138]]) {{.*}}: (!fir.ref<i64>, !fir.ref<i64>) -> ()
! CHECK:         return
! CHECK:       }

subroutine test(ptr1, ptr2)
  use, intrinsic :: iso_c_binding
  type(c_ptr) :: ptr1
  type(c_funptr) :: ptr2

  interface
    subroutine c_func(c_t1, c_t2) bind(c, name="c_func")
      import :: c_ptr, c_funptr
      type(c_ptr), value :: c_t1
      type(c_funptr), value :: c_t2
    end
  end interface

  call c_func(ptr1, ptr2)
end

! CHECK-LABEL: func.func @test_callee_c_ptr(
! CHECK-SAME:                               %[[VAL_0:.*]]: !fir.ref<i64>
! CHECK:         %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:         %[[VAL_5:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = "local", uniq_name = "_QFtest_callee_c_ptrElocal"}
! CHECK:         %[[VAL_132:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFtest_callee_c_ptrElocal"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)
! CHECK:         %[[VAL_133:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_134:.*]] = fir.coordinate_of %[[VAL_133]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[VAL_135:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<i64>) -> i64
! CHECK:         fir.store %[[VAL_135]] to %[[VAL_134]] : !fir.ref<i64>
! CHECK:         %[[VAL_136:.*]]:2 = hlfir.declare %[[VAL_133]] dummy_scope %[[VAL_1]] arg 1 {fortran_attrs = #fir.var_attrs<value>, uniq_name = "_QFtest_callee_c_ptrEptr1"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)
! CHECK:         hlfir.assign %[[VAL_136]]#0 to %[[VAL_132]]#0 : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK:         return
! CHECK:       }

subroutine test_callee_c_ptr(ptr1) bind(c)
  use, intrinsic :: iso_c_binding
  type(c_ptr), value :: ptr1
  type(c_ptr) :: local
  local = ptr1
end subroutine

! CHECK-LABEL: func.func @test_callee_c_funptr(
! CHECK-SAME:                                  %[[VAL_0:.*]]: !fir.ref<i64>
! CHECK:         %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:         %[[VAL_5:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}> {bindc_name = "local", uniq_name = "_QFtest_callee_c_funptrElocal"}
! CHECK:         %[[VAL_132:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFtest_callee_c_funptrElocal"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>)
! CHECK:         %[[VAL_133:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:         %[[VAL_134:.*]] = fir.coordinate_of %[[VAL_133]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[VAL_135:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<i64>) -> i64
! CHECK:         fir.store %[[VAL_135]] to %[[VAL_134]] : !fir.ref<i64>
! CHECK:         %[[VAL_136:.*]]:2 = hlfir.declare %[[VAL_133]] dummy_scope %[[VAL_1]] arg 1 {fortran_attrs = #fir.var_attrs<value>, uniq_name = "_QFtest_callee_c_funptrEptr1"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>)
! CHECK:         hlfir.assign %[[VAL_136]]#0 to %[[VAL_132]]#0 : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>
! CHECK:         return
! CHECK:       }

subroutine test_callee_c_funptr(ptr1) bind(c)
  use, intrinsic :: iso_c_binding
  type(c_funptr), value :: ptr1
  type(c_funptr) :: local
  local = ptr1
end subroutine
