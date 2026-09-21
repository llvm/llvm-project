; REQUIRES: riscv
; RUN: rm -rf %t && split-file %s %t

;--- no-ext.ll
;; The module flag asks for lp64d, and _start() has no target-features attribute.
;; Without -mcpu we default to no D extension, so RISCVSubtarget prints a note
;; and ignores the module flag.
; RUN: llvm-as %t/no-ext.ll -o %t/no-ext.bc
; RUN: ld.lld -shared %t/no-ext.bc -o %t/no-ext.so 2>&1 | FileCheck %s --check-prefix=WARN \
; RUN:   --implicit-check-not="ignoring target-abi" --implicit-check-not="error:" --implicit-check-not="warning:"
; WARN: note: hard-float 'd' ABI can't be used for a target that doesn't support the D instruction set extension (ignoring target-abi)

;; TODO: This is inconsistent: RISCVAsmPrinter::emitStartOfAsmFile sets e_flags
;; based on the raw module flag not the ABI actually used for codegen.
;; This means we are setting EF_RISCV_FLOAT_ABI_DOUBLE on a file built for soft float ABI
; RUN: llvm-readobj --file-headers %t/no-ext.so | FileCheck %s --check-prefix=FLAGS-ABI-IGNORED
; FLAGS-ABI-IGNORED: Flags [ (0x4)
; FLAGS-ABI-IGNORED-NEXT: EF_RISCV_FLOAT_ABI_DOUBLE (0x4)
; FLAGS-ABI-IGNORED-NEXT: ]

;; Passing -mcpu that has D makes the ABI valid again, so no warning/note.
; RUN: ld.lld -mllvm -mcpu=sifive-u74 -shared %t/no-ext.bc -o %t/no-ext.so 2>&1 | FileCheck %s --check-prefix=NOWARN --allow-empty \
; RUN:   --implicit-check-not="error:" --implicit-check-not="warning:" --implicit-check-not="note:"
; RUN: llvm-readobj --file-headers %t/no-ext.so | FileCheck %s --check-prefix=FLAGS-MCPU
; RUN: ld.lld -plugin-opt=mcpu=sifive-u74 -shared %t/no-ext.bc -o %t/no-ext.so 2>&1 | FileCheck %s --check-prefix=NOWARN --allow-empty \
; RUN:   --implicit-check-not="error:" --implicit-check-not="warning:" --implicit-check-not="note:"
; RUN: llvm-readobj --file-headers %t/no-ext.so | FileCheck %s --check-prefix=FLAGS-MCPU
; NOWARN-NOT: ignoring target-abi
; FLAGS-MCPU: Flags [ (0x5)
; FLAGS-MCPU-NEXT: EF_RISCV_FLOAT_ABI_DOUBLE (0x4)
; FLAGS-MCPU-NEXT: EF_RISCV_RVC (0x1)
; FLAGS-MCPU-NEXT: ]

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64"

module asm "nop"
module asm(target_features: "+c") "c.nop"
module asm(target_features: "+d") "fld f0, 0(sp)"

define void @_start() {
  call void asm sideeffect "nop", ""()
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"lp64d"}

;--- fn-inline-asm-no-ext.ll
;; Function-level inline asm without +d on the function emits the missing D
;; note once from RISCVSubtarget, without re-validating target-abi in RISCVAsmParser.
; RUN: llvm-as %t/fn-inline-asm-no-ext.ll -o %t/fn-inline-asm-no-ext.bc
; RUN: ld.lld -shared %t/fn-inline-asm-no-ext.bc -o %t/fn-inline-asm-no-ext.so 2>&1 | FileCheck %s --check-prefix=WARN \
; RUN:   --implicit-check-not="ignoring target-abi" --implicit-check-not="error:" --implicit-check-not="warning:"

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64"

define void @_start() {
  call void asm sideeffect "nop", ""()
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"lp64d"}

;--- module-asm-no-ext.ll
;; Module-level inline asm without target_features (e.g. Rust global_asm!) should
;; not warn when functions in the module have +f,+d.
; RUN: llvm-as %t/module-asm-no-ext.ll -o %t/module-asm-no-ext.bc
; RUN: ld.lld -plugin-opt=mcpu=generic-rv64 -shared %t/module-asm-no-ext.bc -o %t/module-asm-no-ext.so 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOWARN --allow-empty \
; RUN:       --implicit-check-not="ignoring target-abi" --implicit-check-not="error:" --implicit-check-not="warning:" --implicit-check-not="note:"
; RUN: ld.lld -plugin-opt=mcpu=sifive-u74 -shared %t/module-asm-no-ext.bc -o %t/module-asm-no-ext.so 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOWARN --allow-empty \
; RUN:       --implicit-check-not="error:" --implicit-check-not="warning:" --implicit-check-not="note:"

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64"

module asm "nop"

define void @_start() #0 {
  ret void
}
attributes #0 = { "target-features"="+f,+d" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"lp64d"}

;--- module-asm-abi.ll
;; Regression test for https://github.com/llvm/llvm-project/pull/213410:
;; Module asm and function target-features specifying +c,+d should link cleanly
;; even when the LTO backend is invoked with -plugin-opt=mcpu=generic-rv64.
; RUN: llvm-as %t/module-asm-abi.ll -o %t/module-asm-abi.bc
; RUN: ld.lld -plugin-opt=mcpu=generic-rv64 -shared %t/module-asm-abi.bc -o %t/module-asm-abi.so 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOWARN --allow-empty \
; RUN:       --implicit-check-not="error:" --implicit-check-not="warning:" --implicit-check-not="note:"
; RUN: llvm-readobj --file-headers %t/module-asm-abi.so | FileCheck %s --check-prefix=FLAGS-MCPU

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64"

module asm(target_features: "+c,+d")
    "nop"

define void @_start() #0 {
  call void asm sideeffect "nop", ""()
  ret void
}
attributes #0 = { "target-features"="+c,+d" }

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"target-abi", !"lp64d"}
!1 = !{i32 6, !"riscv-isa", !2}
!2 = !{!"rv64i2p1_c2p0_d2p2"}
