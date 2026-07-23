! RUN: bbc -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test intrinsic module procedure c_f_strpointer

! CHECK-LABEL: func.func @_QPtest_cstrarray(
subroutine test_cstrarray(cstrarray, fstrptr)
  use iso_c_binding
  character(len=1, kind=c_char), dimension(*), target, intent(in) :: cstrarray
  character(len=:), pointer, intent(out) :: fstrptr
  ! CHECK-DAG: %[[CSTRARRAY_DECL:.*]]:2 = hlfir.declare %{{.*}} {{.*}} {{{.*}}uniq_name = "_QFtest_cstrarrayEcstrarray"}
  ! CHECK-DAG: %[[FSTRPTR_DECL:.*]]:2 = hlfir.declare %{{.*}} {{.*}} {{{.*}}uniq_name = "_QFtest_cstrarrayEfstrptr"}
  ! CHECK: %[[NCHARS:.*]] = arith.constant 100 : i32
  ! CHECK: %[[NCHARS_IDX:.*]] = fir.convert %[[NCHARS]] : (i32) -> index
  ! CHECK: %[[PTR:.*]] = fir.convert %[[CSTRARRAY_DECL]]#1 : (!fir.ref<!fir.array<?x!fir.char<1>>>) -> !fir.ptr<!fir.char<1,?>>
  ! CHECK: %[[BOX:.*]] = fir.embox %[[PTR]] typeparams %[[NCHARS_IDX]] : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
  ! CHECK: fir.store %[[BOX]] to %[[FSTRPTR_DECL]]#0
  call c_f_strpointer(cstrarray, fstrptr, 100)
end subroutine

! CHECK-LABEL: func.func @_QPtest_cstrarray_no_nchars(
subroutine test_cstrarray_no_nchars(fstrptr)
  use iso_c_binding
  character(len=1, kind=c_char), dimension(100), target :: cstrarray
  character(len=:), pointer, intent(out) :: fstrptr
  ! CHECK-DAG: %[[CSTRARRAY_DECL:.*]]:2 = hlfir.declare %{{.*}} {{.*}} {{{.*}}uniq_name = "_QFtest_cstrarray_no_ncharsEcstrarray"}
  ! CHECK-DAG: %[[FSTRPTR_DECL:.*]]:2 = hlfir.declare %{{.*}} {{.*}} {{{.*}}uniq_name = "_QFtest_cstrarray_no_ncharsEfstrptr"}
  ! CHECK: hlfir.assign %{{.*}} to %[[CSTRARRAY_DECL]]#0
  ! CHECK: %[[I8PTR:.*]] = fir.convert %[[CSTRARRAY_DECL]]#0 : (!fir.ref<!fir.array<100x!fir.char<1>>>) -> !fir.ref<i8>
  ! CHECK: %[[STRLEN:.*]] = fir.call @strlen(%[[I8PTR]]) {{.*}} : (!fir.ref<i8>) -> i64
  ! CHECK: %[[STRLEN_IDX:.*]] = fir.convert %[[STRLEN]] : (i64) -> index
  ! CHECK: %[[PTR:.*]] = fir.convert %[[CSTRARRAY_DECL]]#0 : (!fir.ref<!fir.array<100x!fir.char<1>>>) -> !fir.ptr<!fir.char<1,?>>
  ! CHECK: %[[BOX:.*]] = fir.embox %[[PTR]] typeparams %[[STRLEN_IDX]] : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
  ! CHECK: fir.store %[[BOX]] to %[[FSTRPTR_DECL]]#0
  cstrarray = 'Hello' // c_null_char
  call c_f_strpointer(cstrarray, fstrptr)
end subroutine

! CHECK-LABEL: func.func @_QPtest_cstrptr(
subroutine test_cstrptr(cptr, fstrptr, nchars)
  use iso_c_binding
  type(c_ptr), intent(in) :: cptr
  character(len=:), pointer, intent(out) :: fstrptr
  integer, intent(in) :: nchars
  ! CHECK-DAG: %[[CPTR_DECL:.*]]:2 = hlfir.declare %{{.*}} {{.*}} {{{.*}}uniq_name = "_QFtest_cstrptrEcptr"}
  ! CHECK-DAG: %[[FSTRPTR_DECL:.*]]:2 = hlfir.declare %{{.*}} {{.*}} {{{.*}}uniq_name = "_QFtest_cstrptrEfstrptr"}
  ! CHECK-DAG: %[[NCHARS_DECL:.*]]:2 = hlfir.declare %{{.*}} {{.*}} {{{.*}}uniq_name = "_QFtest_cstrptrEnchars"}
  ! CHECK: %[[NCHARS_LOAD:.*]] = fir.load %[[NCHARS_DECL]]#0
  ! CHECK: %[[ADDR_REF:.*]] = fir.coordinate_of %[[CPTR_DECL]]#0, __address
  ! CHECK: %[[ADDR_VAL:.*]] = fir.load %[[ADDR_REF]] : !fir.ref<i64>
  ! CHECK: %[[NCHARS_IDX:.*]] = fir.convert %[[NCHARS_LOAD]] : (i32) -> index
  ! CHECK: %[[PTR:.*]] = fir.convert %[[ADDR_VAL]] : (i64) -> !fir.ptr<!fir.char<1,?>>
  ! CHECK: %[[BOX:.*]] = fir.embox %[[PTR]] typeparams %[[NCHARS_IDX]] : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
  ! CHECK: fir.store %[[BOX]] to %[[FSTRPTR_DECL]]#0
  call c_f_strpointer(cptr, fstrptr, nchars)
end subroutine

end
