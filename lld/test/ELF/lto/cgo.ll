; REQUIRES: x86

; RUN: llvm-as %s -o %t.o
; RUN: ld.lld --lto-O0 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=OPT %s
; RUN: ld.lld --lto-CGO0 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=NOOPT %s
; RUN: ld.lld --lto-O3 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=OPT %s
; RUN: ld.lld --lto-CGO3 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=OPT %s
; RUN: ld.lld --lto-O0 --lto-CGO0 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=NOOPT %s
; RUN: ld.lld --lto-O3 --lto-CGO0 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=NOOPT %s
; RUN: ld.lld --lto-O0 --lto-CGO3 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=OPT %s
; RUN: ld.lld --lto-O3 --lto-CGO3 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=OPT %s
; RUN: ld.lld --lto-O0 --lto-CGO0 --lto-CGO2 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=OPT %s
; RUN: not ld.lld --lto-O4 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=ERROR-O4 %s
; RUN: not ld.lld --lto-CGO4 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=ERROR-CGO4 %s
; RUN: not ld.lld --lto-O4 --lto-CGO4 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefixes=ERROR-O4,ERROR-CGO4 %s

; NOOPT: Fast Register Allocator
; OPT: Greedy Register Allocator
; ERROR-O4: ld.lld: error: invalid optimization level for LTO: 4
; ERROR-CGO4: ld.lld: error: invalid codegen optimization level for LTO: 4

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @_start() {
entry:
  ret void
}
