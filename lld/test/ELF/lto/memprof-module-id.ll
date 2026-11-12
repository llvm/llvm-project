; REQUIRES: x86

; RUN: split-file %s %t
; RUN: llvm-profdata merge %t/memprof.yaml -o %t/memprof.profdata

; RUN: llvm-as %t/a.ll -o %t/a.bc
; RUN: llvm-as %t/b.ll -o %t/b.bc
; RUN: ld.lld -shared --save-temps --lto-debug-pass-manager \
; RUN:   --lto-memprof-profile=%t/memprof.profdata %t/a.bc %t/b.bc \
; RUN:   -o %t/full.so 2>&1 | FileCheck %s --check-prefix=USE
; RUN: llvm-dis %t/full.so.0.0.preopt.bc -o - | FileCheck %s
; RUN: llvm-readelf --notes %t/full.so | FileCheck %s --check-prefix=NOTE

; RUN: opt -module-summary %t/a.ll -o %t/a.thin.bc
; RUN: opt -module-summary %t/b.ll -o %t/b.thin.bc
; RUN: ld.lld -shared --save-temps --lto-debug-pass-manager \
; RUN:   --lto-memprof-profile=%t/memprof.profdata %t/a.thin.bc %t/b.thin.bc \
; RUN:   -o %t/thin.so 2>&1 | FileCheck %s --check-prefix=USE
; RUN: llvm-dis %t/a.thin.bc.0.preopt.bc -o - | FileCheck %s
; RUN: llvm-dis %t/b.thin.bc.0.preopt.bc -o - | FileCheck %s
; RUN: llvm-readelf --notes %t/thin.so | FileCheck %s --check-prefix=NOTE

; CHECK: !llvm.module.flags = !{[[ID:![0-9]+]]}
; CHECK: [[ID]] = !{i32 1, !"MemProfModuleId", !"{{.*}}"}

; USE: Running pass: MemProfUsePass

; NOTE: Displaying notes found in: .note.llvm.memprof.module-id
; NOTE: LLVM
; NOTE-SAME: 0x00000010
; NOTE-NEXT: description data: 81 78 c7 a8 38 0f 9e 15 64 1f 28 07 9e 9e 29 70

;--- memprof.yaml
---
HeapProfileRecords:
  - GUID: 0x7f21e26c4887dd26
    AllocSites:
      - Callstack:
          - { Function: 0x7f21e26c4887dd26, LineOffset: 0, Column: 0,
              IsInlineFrame: false }
        MemInfoBlock:
          AllocCount: 1
          TotalSize: 4
          TotalLifetime: 1
          TotalLifetimeAccessDensity: 1
    CallSites: []
...

;--- a.ll

source_filename = "a.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$__memprof_profile_filename = comdat any
@__memprof_profile_filename = constant [16 x i8] c"memprof.profraw\00", comdat
@__llvm_memprof_tu_id = private constant { i32, i32, i32, [8 x i8], [16 x i8] } { i32 5, i32 16, i32 1297110017, [8 x i8] c"LLVM\00\00\00\00", [16 x i8] c"\1D\2C\7F\98\B2\5E\34\CA\2D\77\C6\2C\1F\EC\F9\02" }, section ".note.llvm.memprof.tu-id", align 4
@llvm.compiler.used = appending global [1 x ptr] [ptr @__llvm_memprof_tu_id], section "llvm.metadata"

define i32 @a() {
  ret i32 1
}

;--- b.ll

source_filename = "b.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$__memprof_profile_filename = comdat any
@__memprof_profile_filename = constant [16 x i8] c"memprof.profraw\00", comdat
@__llvm_memprof_tu_id = private constant { i32, i32, i32, [8 x i8], [16 x i8] } { i32 5, i32 16, i32 1297110017, [8 x i8] c"LLVM\00\00\00\00", [16 x i8] c"\D8\74\6D\4F\04\F0\59\B1\DF\E0\1B\00\ED\84\B3\24" }, section ".note.llvm.memprof.tu-id", align 4
@llvm.compiler.used = appending global [1 x ptr] [ptr @__llvm_memprof_tu_id], section "llvm.metadata"

define i32 @b() {
  ret i32 2
}
