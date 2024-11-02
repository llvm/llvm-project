! Verify that`-O{n}` is indeed taken into account when defining the LLVM backend pass pipeline.

! REQUIRES: aarch64-registered-target

! RUN: %flang_fc1 -S -O2 %s -triple aarch64-unknown-linux-gnu -mllvm -debug-pass=Structure -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O2
! RUN: %flang_fc1 -S -O3 %s -triple aarch64-unknown-linux-gnu -mllvm -debug-pass=Structure -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O3

! CHECK-O2-NOT: SVE intrinsics optimizations

! CHECK-O3: SVE intrinsics optimizations

subroutine simple_loop
  integer :: i
  do i=1,5
  end do
end subroutine
