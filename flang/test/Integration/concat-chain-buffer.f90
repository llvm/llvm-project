! Test that a chain of concatenations is evaluated into a single buffer.
! RUN: %flang_fc1 -emit-llvm -O2 -o - %s | FileCheck %s

subroutine test(a, b, c)
  character(*) :: a, b, c
  call foo(a // b // c)
end subroutine
! CHECK-LABEL: define void @test_(
! CHECK: alloca i8, i64
! CHECK-NOT: alloca i8, i64
! CHECK-NOT: malloc
! CHECK: call void @foo_(
! CHECK-NOT: free
! CHECK: ret void
