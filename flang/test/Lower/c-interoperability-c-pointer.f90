! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest(
! CHECK-SAME:                     %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "ptr1"},
! CHECK-SAME:                     %[[VAL_1:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>> {fir.bindc_name = "ptr2"}) {
! CHECK:         %[[VAL_2:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<i64>
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> !fir.ref<i64>
! CHECK:         %[[VAL_6:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:         %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_6]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_7]] : !fir.ref<i64>
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> !fir.ref<i64>
! CHECK:         fir.call @c_func(%[[VAL_5]], %[[VAL_9]]) {{.*}}: (!fir.ref<i64>, !fir.ref<i64>) -> ()
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
! CHECK-SAME:                               %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "ptr1"}) attributes {fir.bindc_name = "test_callee_c_ptr"} {
! CHECK:         %[[VAL_5:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = "local", uniq_name = "_QFtest_callee_c_ptrElocal"}
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_2:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<i64>) -> i64
! CHECK:         fir.store %[[VAL_4]] to %[[VAL_3]] : !fir.ref<i64>
! CHECK:         %[[VAL_6:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_6]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_8:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_5]], %[[VAL_8]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_10:.*]] = fir.load %[[VAL_7]] : !fir.ref<i64>
! CHECK:         fir.store %[[VAL_10]] to %[[VAL_9]] : !fir.ref<i64>
! CHECK:         return
! CHECK:       }

subroutine test_callee_c_ptr(ptr1) bind(c)
  use, intrinsic :: iso_c_binding
  type(c_ptr), value :: ptr1
  type(c_ptr) :: local
  local = ptr1
end subroutine

! CHECK-LABEL: func.func @test_callee_c_funptr(
! CHECK-SAME:                                  %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "ptr1"}) attributes {fir.bindc_name = "test_callee_c_funptr"} {
! CHECK:         %[[VAL_5:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}> {bindc_name = "local", uniq_name = "_QFtest_callee_c_funptrElocal"}
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:         %[[VAL_2:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:         %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<i64>) -> i64
! CHECK:         fir.store %[[VAL_4]] to %[[VAL_3]] : !fir.ref<i64>

! CHECK:         %[[VAL_6:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:         %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_6]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_8:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:         %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_5]], %[[VAL_8]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_10:.*]] = fir.load %[[VAL_7]] : !fir.ref<i64>
! CHECK:         fir.store %[[VAL_10]] to %[[VAL_9]] : !fir.ref<i64>
! CHECK:         return
! CHECK:       }

subroutine test_callee_c_funptr(ptr1) bind(c)
  use, intrinsic :: iso_c_binding
  type(c_funptr), value :: ptr1
  type(c_funptr) :: local
  local = ptr1
end subroutine
