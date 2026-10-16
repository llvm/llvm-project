! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPcount_test1(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>{{.*}})
subroutine count_test1(rslt, mask)
  integer :: rslt
  logical :: mask(:)
  ! CHECK: %[[mask_decl:.*]]:2 = hlfir.declare %[[arg1]] {{.*}}
  ! CHECK: %[[rslt_decl:.*]]:2 = hlfir.declare %[[arg0]] {{.*}}
  ! CHECK: %[[res:.*]] = hlfir.count %[[mask_decl]]#0 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> i32
  ! CHECK: hlfir.assign %[[res]] to %[[rslt_decl]]#0 : i32, !fir.ref<i32>
  rslt = count(mask)
end subroutine

! CHECK-LABEL: func.func @_QPtest_count2(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>{{.*}})
subroutine test_count2(rslt, mask)
  integer :: rslt(:)
  logical :: mask(:,:)
  ! CHECK: %[[mask_decl:.*]]:2 = hlfir.declare %[[arg1]] {{.*}}
  ! CHECK: %[[rslt_decl:.*]]:2 = hlfir.declare %[[arg0]] {{.*}}
  ! CHECK: %[[c1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[res:.*]] = hlfir.count %[[mask_decl]]#0 dim %[[c1_i32]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, i32) -> !hlfir.expr<?xi32>
  ! CHECK: hlfir.assign %[[res]] to %[[rslt_decl]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
  ! CHECK: hlfir.destroy %[[res]] : !hlfir.expr<?xi32>
  rslt = count(mask, dim=1)
end subroutine

! CHECK-LABEL: func.func @_QPtest_count3(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>{{.*}})
subroutine test_count3(rslt, mask)
  integer :: rslt
  logical :: mask(:)
  ! CHECK: %[[mask_decl:.*]]:2 = hlfir.declare %[[arg1]] {{.*}}
  ! CHECK: %[[res:.*]] = hlfir.count %[[mask_decl]]#0 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> i16
  ! CHECK: %[[assoc:.*]]:3 = hlfir.associate %[[res]] {{.*}}: (i16) -> (!fir.ref<i16>, !fir.ref<i16>, i1)
  ! CHECK: fir.call @_QPbar(%[[assoc]]#0) {{.*}}: (!fir.ref<i16>) -> ()
  ! CHECK: hlfir.end_associate %[[assoc]]#1, %[[assoc]]#2 : !fir.ref<i16>, i1
  call bar(count(mask, kind=2))
end subroutine
