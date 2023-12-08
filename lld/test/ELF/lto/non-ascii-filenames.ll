; REQUIRES: x86

;; Show that both regular LTO and ThinLTO work correctly when an input file uses
;; a non-ascii filename.

;; Regular LTO.
; RUN: llvm-as %s -o %t£.o
; RUN: ld.lld %t£.o -o %t
; RUN: llvm-readelf -s %t | FileCheck %s

;; Thin LTO.
; RUN: opt -module-summary %s -o %t-thin£.o
; RUN: ld.lld %t-thin£.o -o %t-thin
; RUN: llvm-readelf -s %t-thin | FileCheck %s

; CHECK: _start

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_start = global i32 0
