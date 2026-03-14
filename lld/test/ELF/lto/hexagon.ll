; REQUIRES: hexagon
;; Test we can infer the e_machine value EM_HEXAGON from a bitcode file.

; RUN: llvm-as %s -o %t.bc
; RUN: ld.lld %t.bc -o %t
; RUN: llvm-readobj -h %t | FileCheck %s

; CHECK:   Class: 32-bit
; CHECK:   DataEncoding: LittleEndian
; CHECK: Machine: EM_HEXAGON

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon-unknown-unknown-elf"

define void @_start() {
  ret void
}
