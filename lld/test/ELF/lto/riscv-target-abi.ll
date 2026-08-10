; REQUIRES: riscv

;; The module flag asks for lp64d, but without -mcpu we default to no D extension,
;; so we print a warning and ignore the module flag.
; RUN: llvm-as %s -o %t.bc
; RUN: ld.lld -shared %t.bc -o %t.so 2>&1 | FileCheck %s --check-prefix=WARN --implicit-check-not="ignoring target-abi"
; WARN: Hard-float 'd' ABI can't be used for a target that doesn't support the D instruction set extension (ignoring target-abi)

;; TODO: This is inconsistent: RISCVAsmPrinter::emitStartOfAsmFile sets e_flags
;; based on the raw module flag not the ABI actually used for codegen.
;; This means we are setting EF_RISCV_FLOAT_ABI_DOUBLE on a file built for soft float ABI
; RUN: llvm-readobj --file-headers %t.so | FileCheck %s --check-prefix=FLAGS-ABI-IGNORED
; FLAGS-ABI-IGNORED: Flags [ (0x4)
; FLAGS-ABI-IGNORED-NEXT: EF_RISCV_FLOAT_ABI_DOUBLE (0x4)
; FLAGS-ABI-IGNORED-NEXT: ]

;; Passing -mcpu that has D makes the ABI valid again, so no warning.
; RUN: ld.lld -mllvm -mcpu=sifive-u74 -shared %t.bc -o %t.so 2>&1 | FileCheck %s --check-prefix=NOWARN --allow-empty
; RUN: llvm-readobj --file-headers %t.so | FileCheck %s --check-prefix=FLAGS-MCPU
; RUN: ld.lld -plugin-opt=mcpu=sifive-u74 -shared %t.bc -o %t.so 2>&1 | FileCheck %s --check-prefix=NOWARN --allow-empty
; RUN: llvm-readobj --file-headers %t.so | FileCheck %s --check-prefix=FLAGS-MCPU
; NOWARN-NOT: ignoring target-abi
; FLAGS-MCPU: Flags [ (0x5)
; FLAGS-MCPU-NEXT: EF_RISCV_FLOAT_ABI_DOUBLE (0x4)
; FLAGS-MCPU-NEXT: EF_RISCV_RVC (0x1)
; FLAGS-MCPU-NEXT: ]

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64"

define void @_start() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"lp64d"}
