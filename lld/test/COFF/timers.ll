; REQUIRES: x86
; RUN: llvm-as %s -o %t.obj

; Test different configurations of lld to get all possible timer outputs.
; RUN: lld-link %t.obj -time -entry:main -debug:noghash 2>&1 | \
; RUN:   FileCheck %s --check-prefix=CHECK1
; RUN: lld-link %t.obj -time -entry:main -debug 2>&1 | \
; RUN:   FileCheck %s --check-prefix=CHECK2
; RUN: lld-link %t.obj -time -entry:main -map 2>&1 | \
; RUN:   FileCheck %s --check-prefix=CHECK3

; CHECK1: Input File Reading:
; CHECK1: LTO:
; CHECK1: Code Layout:
; CHECK1: Commit Output File:
; CHECK1: PDB Emission (Cumulative):
; CHECK1:   Add Objects:
; CHECK1:     Type Merging:
; CHECK1:     Symbol Merging:
; CHECK1:   Publics Stream Layout:
; CHECK1:   TPI Stream Layout:
; CHECK1:   Commit to Disk:

; CHECK2: Input File Reading:
; CHECK2: LTO:
; CHECK2: Code Layout:
; CHECK2: Commit Output File:
; CHECK2: PDB Emission (Cumulative):
; CHECK2:   Add Objects:
; CHECK2:     Global Type Hashing:
; CHECK2:     GHash Type Merging:
; CHECK2:     Symbol Merging:
; CHECK2:   Publics Stream Layout:
; CHECK2:   TPI Stream Layout:
; CHECK2:   Commit to Disk:

; CHECK3: Input File Reading:
; CHECK3: LTO:
; CHECK3: GC:
; CHECK3: ICF:
; CHECK3: Code Layout:
; CHECK3: Commit Output File:
; CHECK3: MAP Emission (Cumulative):
; CHECK3:   Gather Symbols:
; CHECK3:   Build Symbol Strings:
; CHECK3:   Write to File:

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.0"

define dso_local i32 @main() {
entry:
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.cpp", directory: "", checksumkind: CSK_MD5, checksum: "495fd79f78a98304e065540d576057d9")
!2 = !{i32 2, !"CodeView", i32 1}
!3 = !{i32 2, !"Debug Info Version", i32 3}
