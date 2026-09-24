! RUN: bbc -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test lowering of F_C_STRING intrinsic from ISO_C_BINDING

! CHECK-LABEL: func @_QPtest_default(
! CHECK-SAME: %[[arg0:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine test_default(str)
  use iso_c_binding
  character(*) :: str
  character(:), allocatable :: result
  
  ! CHECK: %[[tmpBox:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
  ! CHECK: %[[strBox:.*]] = fir.embox %{{.*}} typeparams %{{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  ! CHECK: %[[resBoxNone:.*]] = fir.convert %[[tmpBox]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: %[[strBoxNone:.*]] = fir.convert %[[strBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
  ! CHECK: %{{.*}} = fir.convert %{{.*}} : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK: fir.call @_FortranAFCString(%[[resBoxNone]], %[[strBoxNone]], %{{(false|.*)}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.ref<i8>, i32) -> ()
  result = f_c_string(str)
  
  ! CHECK: fir.freemem
end subroutine

! CHECK-LABEL: func @_QPtest_with_asis(
! CHECK-SAME: %[[arg0:.*]]: !fir.boxchar<1>{{.*}}, %[[arg1:.*]]: !fir.ref<!fir.logical<4>>{{.*}}) {
subroutine test_with_asis(str, keep_blanks)
  use iso_c_binding
  character(*) :: str
  logical :: keep_blanks
  character(:), allocatable :: result
  
  ! CHECK: %[[tmpBox:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
  ! CHECK: %[[strBox:.*]] = fir.embox %{{.*}} typeparams %{{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  ! CHECK: %[[resBoxNone:.*]] = fir.convert %[[tmpBox]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: %[[strBoxNone:.*]] = fir.convert %[[strBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
  ! CHECK: %[[asisBool:.*]] = fir.convert %{{.*}} : (!fir.logical<4>) -> i1
  ! CHECK: %{{.*}} = fir.convert %{{.*}} : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK: fir.call @_FortranAFCString(%[[resBoxNone]], %[[strBoxNone]], %[[asisBool]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.ref<i8>, i32) -> ()
  result = f_c_string(str, keep_blanks)
  
  ! CHECK: fir.freemem
end subroutine

! CHECK-LABEL: func @_QPtest_literal_asis(
subroutine test_literal_asis()
  use iso_c_binding
  character(:), allocatable :: result
  
  ! CHECK: %{{.*}} = fir.convert %{{.*}} : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK: fir.call @_FortranAFCString(%{{.*}}, %{{.*}}, %{{(true|.*)}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.ref<i8>, i32) -> ()
  result = f_c_string('hello', .true.)
end subroutine
