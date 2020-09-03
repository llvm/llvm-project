; REQUIRES: x86

;; For non-relocatable output, test we parse symbol versions after LTO,
;; otherwise we may get a symbol named "foo@@VER1", but not "foo" with the
;; version VER1.

; RUN: llvm-as %s -o %t.o
; RUN: echo "VER1 {};" > %t.script
; RUN: ld.lld %t.o -o %t.so -shared --version-script %t.script
; RUN: llvm-readelf --dyn-syms %t.so | FileCheck %s

;; For non-relocatable output, @ in symbol names has no effect on undefined symbols.
; CHECK:      UND       bar{{$}}
; CHECK-NEXT: {{[1-9]}} foo@@VER1

;; For relocatable output, @ should be retained in the symbol name.
;; Don't parse and drop `@VER1`. Also check that --version-script is ignored.
; RUN: ld.lld %t.o -o %t.ro -r --version-script %t.script
; RUN: llvm-readelf -s %t.ro | FileCheck %s --check-prefix=RELOCATABLE

; RELOCATABLE:      {{[1-9]}} foo@@VER1
; RELOCATABLE-NEXT: UND       bar@VER1

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".global foo"
module asm "foo: call bar"
module asm ".symver foo,foo@@@VER1"
module asm ".symver bar,bar@@@VER1"
