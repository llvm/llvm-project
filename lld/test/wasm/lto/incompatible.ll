; REQUIRES: x86
; RUN: llvm-as %s -o %t.bc
; RUN: not wasm-ld %t.bc -o %t.wasm 2>&1 | FileCheck %s

; RUN: llvm-ar rc %t.a %t.bc
; RUN: not wasm-ld --whole-archive %t.a -o %t.wasm 2>&1 | FileCheck %s --check-prefix=CHECK-ARCHIVE

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: {{.*}}incompatible.ll.tmp.bc: machine type must be wasm32
; CHECK-ARCHIVE: wasm-ld: error: {{.*}}.a(incompatible.ll.tmp.bc): machine type must be wasm32
