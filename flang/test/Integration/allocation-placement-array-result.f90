! Array function results are pinned to the stack in lowering (fir.must_be_stack),
! so allocation placement must not move them to the heap. The abstract-result
! pass turns the result into a caller-provided buffer, hence no malloc/free.

! RUN: %flang_fc1 -emit-llvm -mllvm -enable-allocation-placement -o - %s 2>&1 | FileCheck %s

function test(n)
  real :: test(n)
  call bar(n)
end function

! CHECK-LABEL: define void @test
! CHECK-NOT: @malloc
! CHECK-NOT: @free
! CHECK: ret void
