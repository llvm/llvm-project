! Verify that`-O{n}` is indeed taken into account when defining the LLVM optimization/middle-end pass pipeline.

!-----------
! RUN LINES
!-----------
! RUN: %flang -S -O0 %s -Xflang -fdebug-pass-manager -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O0
! RUN: %flang_fc1 -S -O0 %s -fdebug-pass-manager -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O0

! RUN: %flang -S -O2 %s -Xflang -fdebug-pass-manager -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O2
! RUN: %flang_fc1 -S -O2 %s -fdebug-pass-manager -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O2

!-----------------------
! EXPECTED OUTPUT
!-----------------------
! CHECK-O0-NOT: Running pass: SimplifyCFGPass on simple_loop_
! CHECK-O0: Running analysis: TargetLibraryAnalysis on simple_loop_

! CHECK-O2: Running pass: SimplifyCFGPass on simple_loop_

!-------
! INPUT
!-------
subroutine simple_loop
  integer :: i
  do i=1,5
  end do
end subroutine
