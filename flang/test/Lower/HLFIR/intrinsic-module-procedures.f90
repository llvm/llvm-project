! Test lowering of intrinsic module procedures to HLFIR. This
! test is not meant to test every intrinsic module procedure,
! it only tests that the HFLIR procedure reference lowering
! infrastructure properly detects and dispatches intrinsic module
! procedure calls.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine foo(cptr, x)
  use iso_c_binding, only : c_ptr, c_loc
  type(c_ptr) :: cptr
  integer, target :: x
  cptr = c_loc(x)
end subroutine
! CHECK-LABEL: func.func @_QPfoo(
! CHECK:         %[[VAL_3:.*]]:2 = hlfir.declare {{.*}}Ecptr"
! CHECK:         %[[VAL_3:.*]]:2 = hlfir.declare {{.*}}Ex"
! CHECK:         %[[VAL_4:.*]] = fir.embox %[[VAL_3]]#1 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:         %[[VAL_5:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_6:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_5]], %[[VAL_6]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_8:.*]] = fir.box_addr %[[VAL_4]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<i32>) -> i64
! CHECK:         fir.store %[[VAL_9]] to %[[VAL_7]] : !fir.ref<i64>

subroutine test_renaming(p)
  use iso_c_binding, only: c_associated_alias => c_associated, c_ptr
  type(c_ptr) p
  print *, c_associated_alias(p)
end subroutine

! CHECK-LABEL: func.func @_QPtest_renaming
! CHECK:  %[[C_PTR_TARG:.*]] = fir.load %{{.*}} : !fir.ref<i64>
! CHECK:  %[[NULL:.*]] = arith.constant 0 : i64
! CHECK:  arith.cmpi ne, %[[C_PTR_TARG]], %[[NULL]] : i64
