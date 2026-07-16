! This test is used to demonstrate that coindexed expressions can be lowered, but it does not in any way validate the assignment `val = a[2]`.
! This test is intended to be removed or modified once PUT/GET operations on coarrays have been supported.

! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program main
  integer :: a[*]
  integer :: val
  
  val = a[2]
end program
    
!CHECK: %[[VAL_1:.*]] = fir.address_of(@_QFEa) : !fir.ref<i32>
!CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "val", uniq_name = "_QFEval"}
!CHECK: %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFEval"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[VAL_5:.*]] = hlfir.designate %[[VAL_2]]#0 : (!fir.ref<i32>) -> !fir.ref<i32>

