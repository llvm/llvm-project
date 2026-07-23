! Test dummy procedure pointers that are not an argument in every entry.
! This requires creating a mock value in the entries where it is not an
! argument.
!
!RUN: %flang_fc1 -emit-hlfir %s -o - 2>&1 | FileCheck %s

!CHECK-LABEL: func @_QPdummy_char_proc_ptr() -> !fir.boxproc<(!fir.ref<!fir.char<1>>, index) -> !fir.boxchar<1>> {
!CHECK:         %[[UNDEF:.*]] = fir.undefined !fir.ref<!fir.boxproc<() -> ()>>
!CHECK:         %{{.*}}:2 = hlfir.declare %[[UNDEF]]
!CHECK-SAME:      {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFdummy_char_proc_ptrEdummy"}
!CHECK-SAME:      : (!fir.ref<!fir.boxproc<() -> ()>>)
!CHECK-SAME:      -> (!fir.ref<!fir.boxproc<() -> ()>>, !fir.ref<!fir.boxproc<() -> ()>>)

!CHECK-LABEL: func @_QPdummy_char_proc_ptr_entry(
!CHECK-SAME:        %[[ARG:.*]]: !fir.ref<!fir.boxproc<() -> ()>>)
!CHECK-SAME:        -> !fir.boxproc<(!fir.ref<!fir.char<1>>, index) -> !fir.boxchar<1>> {
!CHECK:         %{{.*}}:2 = hlfir.declare %[[ARG]] dummy_scope %{{[^ ]*}}
!CHECK-SAME:      {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFdummy_char_proc_ptrEdummy"}
!CHECK-SAME:      : (!fir.ref<!fir.boxproc<() -> ()>>, !fir.dscope)
!CHECK-SAME:      -> (!fir.ref<!fir.boxproc<() -> ()>>, !fir.ref<!fir.boxproc<() -> ()>>)
function dummy_char_proc_ptr() result(fun)
  interface
    character function char_fun()
    end function
  end interface

  procedure (char_fun), pointer :: fun, dummy_char_proc_ptr_entry, dummy
  fun => null()
  return

  entry dummy_char_proc_ptr_entry(dummy)
end function

!CHECK-LABEL: func @_QPdummy_int_proc_ptr()
!CHECK:         %[[UNDEF:.*]] = fir.undefined !fir.ref<!fir.boxproc<() -> ()>>
!CHECK:         %{{.*}}:2 = hlfir.declare %[[UNDEF]]
!CHECK-SAME:      {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFdummy_int_proc_ptrEdummy"}
!CHECK-SAME:      : (!fir.ref<!fir.boxproc<() -> ()>>)
!CHECK-SAME:      -> (!fir.ref<!fir.boxproc<() -> ()>>, !fir.ref<!fir.boxproc<() -> ()>>)

!CHECK-LABEL: func @_QPdummy_int_proc_ptr_entry(
!CHECK-SAME:        %[[ARG:.*]]: !fir.ref<!fir.boxproc<() -> ()>>)
!CHECK-SAME:        -> !fir.boxproc<() -> i32> {
!CHECK:         %{{.*}}:2 = hlfir.declare %[[ARG]] dummy_scope %{{[^ ]*}}
!CHECK-SAME:      {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFdummy_int_proc_ptrEdummy"}
!CHECK-SAME:      : (!fir.ref<!fir.boxproc<() -> ()>>, !fir.dscope)
!CHECK-SAME:      -> (!fir.ref<!fir.boxproc<() -> ()>>, !fir.ref<!fir.boxproc<() -> ()>>)
function dummy_int_proc_ptr() result(fun)
  interface
    integer function int_fun()
    end function
  end interface

  procedure (int_fun), pointer :: fun, dummy_int_proc_ptr_entry, dummy
  fun => null()
  return

  entry dummy_int_proc_ptr_entry(dummy)
end function
