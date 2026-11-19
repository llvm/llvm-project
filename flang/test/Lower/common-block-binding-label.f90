! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck %s

! Differently named COMMON blocks with the same label share one object.
! CHECK: @shared = common global [4 x i8] zeroinitializer
! CHECK-NOT: = common global
! CHECK-LABEL: define {{.*}} @first_(
! CHECK: store i32 1, ptr @shared
! CHECK-LABEL: define {{.*}} @second_(
! CHECK: store i32 2, ptr @shared
! CHECK-LABEL: define {{.*}} @third_(
! CHECK: store i32 3, ptr @shared

subroutine first
  integer :: x
  common /Shared/ x
  bind(c) :: /Shared/
  x = 1
end subroutine

subroutine second
  integer :: x
  common /other/ x
  bind(c, name="shared") :: /other/
  x = 2
end subroutine

subroutine third
  integer :: x
  common /third_name/ x
  bind(c, name="shared") :: /third_name/
  x = 3
end subroutine
