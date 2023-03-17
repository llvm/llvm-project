! Test -emit-obj (RISC-V 64)

! REQUIRES: riscv-registered-target

! RUN: rm -f %t.o
! RUN: %flang_fc1 -triple riscv64-unknown-linux-gnu \
! RUN:   -target-feature +d -target-feature +c -emit-obj %s -o %t.o
! RUN: llvm-readobj -h %t.o | FileCheck %s

! RUN: rm -f %t.o
! RUN: %flang --target=riscv64-unknown-linux-gnu -c %s -o %t.o
! RUN: llvm-readobj -h %t.o | FileCheck %s

! If Flang failed to emit target-feature info, then Flags will be 0x0.
! 0x5 means set EF_RISCV_RVC (0x1) and EF_RISCV_FLOAT_ABI_DOUBLE (0x4)
! CHECK: Flags: 0x5, RVC, double-float ABI
end program
