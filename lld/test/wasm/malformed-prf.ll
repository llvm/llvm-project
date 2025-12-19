; RUN: llc -filetype=obj %s -o %t.instr-prof.o
; RUN: llc -filetype=obj %p/Inputs/malformed-prf1.ll -o %t.malformed-prf1.o
; RUN: llc -filetype=obj %p/Inputs/malformed-prf2.ll -o %t.malformed-prf2.o
; RUN: wasm-ld -o %t.wasm %t.instr-prof.o %t.malformed-prf1.o --start-lib %t.malformed-prf2.o --end-lib --gc-sections --no-entry
; RUN: llvm-cov export --object %t.wasm --empty-profile

; Every covfun record holds a hash of its symbol name, and llvm-cov will exit fatally if
; it can't resolve that hash back to an entry in the binary's `__llvm_prf_names` linker section.
;
; WASM stores `__llvm_covfun` in custom section, while `__llvm_prf_names` is stored in the DATA section.
; The former not be GC, whereas the latter may be GC, causing llvm-cov execution to fail.
;
; Now, __llvm_covfun and __llvm_covmap will be dropped if __llvm_prf_names is discarded.

; ModuleID = 'instr-prof.c'
source_filename = "instr-prof.c"
target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

@__llvm_profile_runtime = hidden global i32 0, align 4

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 2, !"EnableValueProfiling", i32 0}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{!"clang version 21.1.6"}

