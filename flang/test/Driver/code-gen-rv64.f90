! Test -emit-obj (RISC-V 64)

! REQUIRES: riscv-registered-target

! RUN: %flang_fc1 -triple riscv64-unknown-linux-gnu \
! RUN:   -target-feature +d -target-feature +c -emit-obj %s -o - | \
! RUN: llvm-readobj -h - | FileCheck %s

! RUN: %flang --target=riscv64-unknown-linux-gnu -c %s -o - | \
! RUN: llvm-readobj -h - | FileCheck %s

! If Flang failed to emit target-feature info, then Flags will be 0x0.
! CHECK: Arch: riscv64
! CHECK: Flags [ (0x5)
! CHECK-NEXT: EF_RISCV_FLOAT_ABI_DOUBLE (0x4)
! CHECK-NEXT: EF_RISCV_RVC (0x1)
! CHECK-NEXT: ]
end program
