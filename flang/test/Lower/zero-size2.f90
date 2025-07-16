! RUN: %flang_fc1 -emit-llvm -o - %s | FileCheck %s

subroutine sub1()
  integer, allocatable :: arr(:)
  allocate(arr(0))
! CHECK-LABEL: @sub1_
! CHECK: %[[p:.*]] = call ptr @malloc(i64 1)
end

subroutine sub2()
  real, allocatable :: arr(:,:)
  allocate(arr(10,0))
! CHECK-LABEL: @sub2_
! CHECK: %[[p:.*]] = call ptr @malloc(i64 1)
end

subroutine sub3(i)
  integer :: i
  real, allocatable :: arr(:,:)
  allocate(arr(i,0))
! CHECK-LABEL: @sub3_
! CHECK: %[[p:.*]] = call ptr @malloc(i64 1)
end

subroutine sub4()
  character(:), allocatable :: c
  allocate(character(0)::c)
! CHECK-LABEL: @sub4_
! CHECK: %[[p:.*]] = call ptr @malloc(i64 1)
end  

subroutine sub5()
  character(:), allocatable :: c(:)
  allocate(character(5)::c(0))
! CHECK-LABEL: @sub5_
! CHECK: %[[p:.*]] = call ptr @malloc(i64 1)
end
