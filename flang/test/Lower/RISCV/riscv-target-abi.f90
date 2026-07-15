! REQUIRES: riscv-registered-target
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -mabi=lp64  -emit-llvm -o - %s | FileCheck %s --check-prefix=LP64
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -mabi=lp64f  -emit-llvm -o - %s | FileCheck %s --check-prefix=LP64F
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -mabi=lp64d  -emit-llvm -o - %s | FileCheck %s --check-prefix=LP64D

! LP64: !{{[0-9]+}} = !{i32 1, !"target-abi", !"lp64"}
! LP64F: !{{[0-9]+}} = !{i32 1, !"target-abi", !"lp64f"}
! LP64D: !{{[0-9]+}} = !{i32 1, !"target-abi", !"lp64d"}
subroutine func
end subroutine func
