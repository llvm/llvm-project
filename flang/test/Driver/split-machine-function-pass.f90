! Verify that the MachineFunctionSplitter pass is enabled while passing -fsplit-machine-functions.

! REQUIRES: x86-registered-target

! RUN: %flang_fc1 -S -fsplit-machine-functions %s \
! RUN:   -triple x86_64-unknown-linux-gnu \
! RUN:   -mllvm -debug-pass=Structure -o %t 2>&1 \
! RUN:   | FileCheck %s --check-prefix=ENABLED

! RUN: %flang_fc1 -S %s \
! RUN:   -triple x86_64-unknown-linux-gnu \
! RUN:   -mllvm -debug-pass=Structure -o %t 2>&1 \
! RUN:   | FileCheck %s --check-prefix=DISABLED

! ENABLED: Machine Function Splitter Transformation
! DISABLED-NOT: Machine Function Splitter Transformation

subroutine test(x)
    integer, intent(in) :: x
  end subroutine test
