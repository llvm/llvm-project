! Verify that the BasicBlockSections pass is enabled while passing
! -fsplit-machine-functions or -mllvm -function-splitting=all.

! REQUIRES: x86-registered-target

! RUN: %flang_fc1 -S -fsplit-machine-functions %s -triple x86_64-unknown-linux-gnu -mllvm -debug-pass=Structure -o /dev/null 2>&1 | FileCheck %s --check-prefix=SPLIT
! RUN: %flang_fc1 -S %s -triple x86_64-unknown-linux-gnu -mllvm -function-splitting=all -mllvm -debug-pass=Structure -o /dev/null 2>&1 | FileCheck %s --check-prefix=SPLIT
! RUN: %flang_fc1 -S %s -triple x86_64-unknown-linux-gnu -mllvm -debug-pass=Structure -o /dev/null 2>&1 | FileCheck %s --check-prefix=NO-SPLIT

! SPLIT: Basic Block Sections Transformation
! NO-SPLIT-NOT: Basic Block Sections Transformation

subroutine test
end subroutine test
