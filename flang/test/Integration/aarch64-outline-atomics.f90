! RUN: %flang -S -emit-llvm --target=aarch64-none-none -moutline-atomics -o - %s | FileCheck %s --check-prefixes=CHECKON,CHECKALL
! RUN: %flang -S -emit-llvm --target=aarch64-none-none -mno-outline-atomics -o - %s | FileCheck %s --check-prefixes=CHECKOFF,CHECKALL
! REQUIRES: aarch64-registered-target

subroutine test()
  integer :: i

  do i = 1, 10
  end do
end subroutine

! CHECKALL-LABEL: define void @test_()
! CHECKALL-SAME: #[[ATTR:[0-9]*]]
! CHECKALL: attributes #[[ATTR]] =
! Use CHECK-SAME to allow arbitrary other attributes to be present.
! CHECKALL-SAME: target-features
! CHECKON-SAME: +outline-atomics
! CHECKOFF-SAME: -outline-atomics
