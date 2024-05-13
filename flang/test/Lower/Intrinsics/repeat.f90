! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPrepeat_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.boxchar<1>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}) {
subroutine repeat_test(c, n)
  character(*) :: c
  integer :: n
  ! CHECK: %[[tmpBox:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
  ! CHECK-DAG: %[[c:.*]]:2 = fir.unboxchar %[[arg0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK-DAG: %[[ni32:.*]] = fir.load %[[arg1]] : !fir.ref<i32>
  ! CHECK-DAG: %[[n:.*]] = fir.convert %[[ni32]] : (i32) -> i64
  ! CHECK-DAG: %[[cBox:.*]] = fir.embox %[[c]]#0 typeparams %[[c]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  ! CHECK-DAG: %[[cBoxNone:.*]] = fir.convert %[[cBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
  ! CHECK-DAG: %[[resBox:.*]] = fir.convert %[[tmpBox]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}Repeat(%[[resBox]], %[[cBoxNone]], %[[n]], {{.*}}, {{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i64, !fir.ref<i8>, i32) -> none
  ! CHECK-DAG: %[[tmpAddr:.*]] = fir.box_addr
  ! CHECK-DAG: fir.box_elesize
  ! CHECK: fir.call @{{.*}}bar_repeat_test
  call bar_repeat_test(repeat(c,n))
  ! CHECK: fir.freemem %[[tmpAddr]] : !fir.heap<!fir.char<1,?>>
  return
end subroutine

