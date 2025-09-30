! REQUIRES: arm-registered-target
! RUN: %flang_fc1 -triple arm-none-eabi -emit-llvm -o - %s | FileCheck %s --check-prefix=IEEE
! RUN: %flang_fc1 -triple arm-none-eabi -menable-no-infs -menable-no-nans -emit-llvm -o - %s | FileCheck %s --check-prefix=NORMAL

subroutine func
end subroutine func

! IEEE: !{i32 2, !"arm-eabi-fp-number-model", i32 3}
! NORMAL: !{i32 2, !"arm-eabi-fp-number-model", i32 1}
