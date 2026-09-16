; Check that when clang consumes existing IR, it does not overwrite
; the module's DataLayout with one incorrectly recomputed from the
; TargetMachine. The module uses the target-abi module flag and a
; matching -S32 DataLayout. Without the -target-abi option the
; TargetMachine's default DL would use -S128, so an overwrite would
; silently corrupt the layout.

; REQUIRES: riscv-registered-target

; RUN: %clang_cc1 -triple riscv32-unknown-unknown-elf -x ir %s -emit-llvm -o - \
; RUN:   | FileCheck %s

; A module with no DataLayout is still seeded from the TargetMachine.
; RUN: echo 'target triple = "riscv32-unknown-unknown-elf"' \
; RUN:   | %clang_cc1 -triple riscv32-unknown-unknown-elf -x ir - -emit-llvm -o - \
; RUN:   | FileCheck --check-prefix=SEEDED %s

; A module with no DataLayout but a target-abi module flag is seeded
; with the layout for that ABI, not the target's default.
; RUN: %clang_cc1 -triple riscv32-unknown-unknown-elf -x ir %S/Inputs/riscv-ilp32e-no-datalayout.ll \
; RUN:   -emit-llvm -o - | FileCheck --check-prefix=FLAG-SEEDED %s

; CHECK: target datalayout = "e-m:e-p:32:32-i64:64-n32-S32"
; SEEDED: target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
; FLAG-SEEDED: target datalayout = "e-m:e-p:32:32-i64:64-n32-S32"

target datalayout = "e-m:e-p:32:32-i64:64-n32-S32"
target triple = "riscv32-unknown-unknown-elf"

define i32 @f() #0 {
  ret i32 0
}

attributes #0 = { "target-features"="+32bit,+e" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"target-abi", !"ilp32e"}
