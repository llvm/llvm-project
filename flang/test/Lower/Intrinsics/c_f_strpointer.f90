! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! Test intrinsic module procedure c_f_strpointer

! CHECK-LABEL: func.func @_QPtest_cstrarray(
! CHECK-SAME: %[[CSTRARRAY:.*]]: !fir.boxchar<1> {fir.bindc_name = "cstrarray", fir.target},
! CHECK-SAME: %[[FSTRPTR:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>> {fir.bindc_name = "fstrptr"}
subroutine test_cstrarray(cstrarray, fstrptr)
  use iso_c_binding
  character(len=1, kind=c_char), dimension(*), target, intent(in) :: cstrarray
  character(len=:), pointer, intent(out) :: fstrptr
  ! CHECK: %[[UNBOXED:.*]]:2 = fir.unboxchar %[[CSTRARRAY]]
  ! CHECK: %[[CONVERTED:.*]] = fir.convert %[[UNBOXED]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK: %[[NCHARS:.*]] = arith.constant 100 : i32
  ! CHECK: %[[NCHARS_IDX:.*]] = fir.convert %[[NCHARS]] : (i32) -> index
  ! CHECK: %[[PTR:.*]] = fir.convert %[[CONVERTED]] : (!fir.ref<!fir.array<?x!fir.char<1>>>) -> !fir.ptr<!fir.char<1,?>>
  ! CHECK: %[[BOX:.*]] = fir.embox %[[PTR]] typeparams %[[NCHARS_IDX]] : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
  ! CHECK: fir.store %[[BOX]] to %[[FSTRPTR]]
  call c_f_strpointer(cstrarray, fstrptr, 100)
end subroutine

! CHECK-LABEL: func.func @_QPtest_cstrarray_no_nchars(
! CHECK-SAME: %[[FSTRPTR:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>> {fir.bindc_name = "fstrptr"}
subroutine test_cstrarray_no_nchars(fstrptr)
  use iso_c_binding
  character(len=1, kind=c_char), dimension(100), target :: cstrarray
  character(len=:), pointer, intent(out) :: fstrptr
  ! CHECK: %[[CSTRARRAY:.*]] = fir.alloca !fir.array<100x!fir.char<1>> {bindc_name = "cstrarray"
  ! CHECK: %[[I8PTR:.*]] = fir.convert %[[CSTRARRAY]] : (!fir.ref<!fir.array<100x!fir.char<1>>>) -> !fir.ref<i8>
  ! CHECK: %[[STRLEN:.*]] = fir.call @strlen(%[[I8PTR]]) {{.*}} : (!fir.ref<i8>) -> i64
  ! CHECK: %[[STRLEN_IDX:.*]] = fir.convert %[[STRLEN]] : (i64) -> index
  ! CHECK: %[[PTR:.*]] = fir.convert %[[CSTRARRAY]] : (!fir.ref<!fir.array<100x!fir.char<1>>>) -> !fir.ptr<!fir.char<1,?>>
  ! CHECK: %[[BOX:.*]] = fir.embox %[[PTR]] typeparams %[[STRLEN_IDX]] : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
  ! CHECK: fir.store %[[BOX]] to %[[FSTRPTR]]
  cstrarray = 'Hello' // c_null_char
  call c_f_strpointer(cstrarray, fstrptr)
end subroutine

! CHECK-LABEL: func.func @_QPtest_cstrptr(
! CHECK-SAME: %[[CPTR:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "cptr"},
! CHECK-SAME: %[[FSTRPTR:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>> {fir.bindc_name = "fstrptr"},
! CHECK-SAME: %[[NCHARS:.*]]: !fir.ref<i32> {fir.bindc_name = "nchars"}
subroutine test_cstrptr(cptr, fstrptr, nchars)
  use iso_c_binding
  type(c_ptr), intent(in) :: cptr
  character(len=:), pointer, intent(out) :: fstrptr
  integer, intent(in) :: nchars
  ! CHECK: %[[NCHARS_LOAD:.*]] = fir.load %[[NCHARS]]
  ! CHECK: %[[ADDR_REF:.*]] = fir.coordinate_of %[[CPTR]], __address
  ! CHECK: %[[ADDR_VAL:.*]] = fir.load %[[ADDR_REF]] : !fir.ref<i64>
  ! CHECK: %[[NCHARS_IDX:.*]] = fir.convert %[[NCHARS_LOAD]] : (i32) -> index
  ! CHECK: %[[PTR:.*]] = fir.convert %[[ADDR_VAL]] : (i64) -> !fir.ptr<!fir.char<1,?>>
  ! CHECK: %[[BOX:.*]] = fir.embox %[[PTR]] typeparams %[[NCHARS_IDX]] : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
  ! CHECK: fir.store %[[BOX]] to %[[FSTRPTR]]
  call c_f_strpointer(cptr, fstrptr, nchars)
end subroutine

end
