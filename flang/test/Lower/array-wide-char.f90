! RUN: bbc %s -o - | tco | FileCheck %s

character(LEN=128, KIND=4), PARAMETER :: conarr(3) = &
     [ character(128,4) :: "now is the time", "for all good men to come", &
     "to the aid of the country" ]       
character(LEN=10, KIND=4) :: arr(3) = &
     [ character(10,4) :: "good buddy", "best buddy", " " ]
call action_on_char4(conarr)
call action_on_char4(arr)
end program

subroutine sub1
  integer, parameter :: k = 4
  character(63,k), parameter :: wiggle = k_"wiggle"
  call sub2(wiggle)
end subroutine sub1

! CHECK-LABEL: @_QFEarr = internal global [3 x [10 x i32]] [
! CHECK-SAME: [10 x i32] [i32 103, i32 111, i32 111, i32 100, i32 32, i32 98, i32 117, i32 100, i32 100, i32 121],
! CHECK-SAME: [10 x i32] [i32 98, i32 101, i32 115, i32 116, i32 32, i32 98, i32 117, i32 100, i32 100, i32 121],
! CHECK-SAME: [10 x i32] [i32 32, i32 32, i32 32, i32 32, i32 32, i32 32, i32 32, i32 32, i32 32, i32 32]]
! CHECK-LABEL: @_QFsub1ECwiggle = internal constant [63 x i32] [i32 119,
! CHECK-SAME: i32 105, i32 103, i32 103, i32 108, i32 101, i32 32, i32 32,
! CHECK: @_QQcl[[inline:.*]] = linkonce constant [63 x i32] [i32 119, i32 105, i32 103, i32 103, i32 108, i32 101, i32 32,

! CHECK-LABEL: define void @_QQmain()
! CHECK: call void @_QPaction_on_char4(ptr @_QFEarr, i64 10)

! CHECK-LABEL: define void @_QPsub1(
! CHECK: call void @_QPsub2(ptr @_QQcl.77, i64 63)
