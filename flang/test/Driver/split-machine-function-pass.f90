! Verify that the MachineFunctionSplitter pass is enabled while passing -fsplit-machine-functions.

! REQUIRES: x86-registered-target

! RUN: %flang_fc1 -S -fsplit-machine-functions %s -triple x86_64-unknown-linux-gnu -mllvm -debug-pass=Structure -o /dev/null 2>&1 | FileCheck %s --check-prefix=SPLIT
! RUN: %flang_fc1 -S %s -triple x86_64-unknown-linux-gnu -mllvm -debug-pass=Structure -o /dev/null 2>&1 | FileCheck %s --check-prefix=NO-SPLIT

! SPLIT: Machine Function Splitter Transformation
! NO-SPLIT-NOT: Machine Function Splitter Transformation

subroutine test
end subroutine test
