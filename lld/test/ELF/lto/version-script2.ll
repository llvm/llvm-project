; REQUIRES: x86

;; For non-relocatable output, test we parse symbol versions after LTO,
;; otherwise we may get a symbol named "foo@@VER1", but not "foo" with the
;; version VER1.

; RUN: split-file %s %t
; RUN: llvm-as %t/ir -o %t.o
; RUN: llvm-mc -filetype=obj -triple=x86_64 %t/asm -o %tbar.o
; RUN: ld.lld %tbar.o -shared --soname=tbar --version-script %t/ver -o %tbar.so

;; Emit an error if bar@VER1 is not defined.
; RUN: not ld.lld %t.o -o /dev/null -shared --version-script %t/ver 2>&1 | FileCheck %s --check-prefix=UNDEF

; UNDEF: error: undefined symbol: bar@VER1

; RUN: ld.lld %t.o %tbar.so -o %t.so -shared --version-script %t/ver
; RUN: llvm-readelf --dyn-syms %t.so | FileCheck %s

; CHECK:      UND       bar@VER1{{$}}
; CHECK-NEXT: {{[1-9]}} foo@@VER1{{$}}

;; For relocatable output, @ should be retained in the symbol name.
;; Don't parse and drop `@VER1`. Also check that --version-script is ignored.
; RUN: ld.lld %t.o -o %t.ro -r --version-script %t/ver
; RUN: llvm-readelf -s %t.ro | FileCheck %s --check-prefix=RELOCATABLE

; RELOCATABLE:      {{[1-9]}} foo@@VER1{{$}}
; RELOCATABLE-NEXT: UND       bar@VER1{{$}}

;--- ver
VER1 {};
;--- ir
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".global foo"
module asm "foo: call bar"
module asm ".symver foo,foo@@@VER1"
module asm ".symver bar,bar@@@VER1"

;--- asm
.globl bar
.symver bar,bar@@@VER1
bar:
  ret
