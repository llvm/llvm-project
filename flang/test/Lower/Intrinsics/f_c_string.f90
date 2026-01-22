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
  ! CHECK: %{{.*}} = fir.convert %{{.*}} : (!fir.box<i1>) -> !fir.box<none>
  ! CHECK: %{{.*}} = fir.convert %{{.*}} : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK: fir.call @_FortranAFCString(%[[resBoxNone]], %[[strBoxNone]], %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
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
  ! CHECK: %[[asisBox:.*]] = fir.embox %{{.*}} : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  ! CHECK: %[[resBoxNone:.*]] = fir.convert %[[tmpBox]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: %[[strBoxNone:.*]] = fir.convert %[[strBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
  ! CHECK: %[[asisBoxNone:.*]] = fir.convert %[[asisBox]] : (!fir.box<!fir.logical<4>>) -> !fir.box<none>
  ! CHECK: %{{.*}} = fir.convert %{{.*}} : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK: fir.call @_FortranAFCString(%[[resBoxNone]], %[[strBoxNone]], %[[asisBoxNone]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
  result = f_c_string(str, keep_blanks)
  
  ! CHECK: fir.freemem
end subroutine

! CHECK-LABEL: func @_QPtest_literal_asis(
subroutine test_literal_asis()
  use iso_c_binding
  character(:), allocatable :: result
  
  ! CHECK: %[[asisTemp:.*]] = fir.alloca !fir.logical<4>
  ! CHECK: %[[trueVal:.*]] = arith.constant true
  ! CHECK: %[[trueLogical:.*]] = fir.convert %[[trueVal]] : (i1) -> !fir.logical<4>
  ! CHECK: fir.store %[[trueLogical]] to %[[asisTemp]] : !fir.ref<!fir.logical<4>>
  ! CHECK: %{{.*}} = fir.convert %{{.*}} : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK: fir.call @_FortranAFCString(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
  result = f_c_string('hello', .true.)
end subroutine
