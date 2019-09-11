; REQUIRES: x86

;; Test we parse symbol versions before LTO, otherwise we may get a symbol
;; named "foo@@VER1", but not "foo" with the version VER1.

; RUN: llvm-as %s -o %t.o
; RUN: echo "VER1 {};" > %t.script
; RUN: ld.lld %t.o -o %t.so -shared --version-script %t.script
; RUN: llvm-readobj --dyn-syms %t.so | FileCheck %s

; CHECK: Name: foo@@VER1 (

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".global foo"
module asm "foo:"
module asm ".symver foo,foo@@@VER1"
