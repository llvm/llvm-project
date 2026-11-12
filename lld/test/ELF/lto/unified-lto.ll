; REQUIRES: x86
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -unified-lto %s -o %t0.o
; RUN: opt -thinlto-bc -thinlto-split-lto-unit %s -o %t1.o
; RUN: ld.lld --lto=full %t0.o -o %t0
; RUN: llvm-readelf -s %t0 | FileCheck %s --check-prefix=FULL
; RUN: ld.lld --lto=thin %t0.o -o %t0
; RUN: llvm-readelf -s %t0 | FileCheck %s --check-prefix=THIN
; RUN: ld.lld --lto=default %t0.o -o %t0
; RUN: llvm-readelf -s %t0 | FileCheck %s --check-prefix=THIN
; RUN: ld.lld --lto=default %t1.o -o %t1
; RUN: llvm-readelf -s %t1 | FileCheck %s --check-prefix=THIN
; RUN: ld.lld %t0.o -o %t0 2>&1 | count 0
; RUN: llvm-readelf -s %t0 | FileCheck %s --check-prefix=THIN
; RUN: not ld.lld --lto=unknown %t1.o -o /dev/null 2>&1 | \
; RUN:   FileCheck --implicit-check-not=error: --check-prefix=ERR %s
; ERR: error: unknown LTO mode: unknown

; FULL:      Symbol table '.symtab' contains 3 entries:
; FULL-NEXT: Num:    Value          Size Type    Bind   Vis       Ndx Name
; FULL-NEXT: 0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND
; FULL-NEXT: 1: 0000000000000000     0 FILE    LOCAL  DEFAULT   ABS ld-temp.o
; FULL-NEXT: 2: 0000000000201120     1 FUNC    GLOBAL DEFAULT     1 _start

; THIN:      Symbol table '.symtab' contains 3 entries:
; THIN-NEXT: Num:    Value          Size Type    Bind   Vis       Ndx Name
; THIN-NEXT: 0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND
; THIN-NEXT: 1: 0000000000000000     0 FILE    LOCAL  DEFAULT   ABS unified-lto.ll
; THIN-NEXT: 2: 0000000000201120     1 FUNC    GLOBAL DEFAULT     1 _start

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @_start() {
  ret void
}
