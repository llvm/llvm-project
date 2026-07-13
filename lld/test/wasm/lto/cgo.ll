; REQUIRES: x86

; RUN: llvm-as %s -o %t.o
; RUN: wasm-ld --lto-O0 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=OPT %s
; RUN: wasm-ld --lto-CGO0 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=NOOPT %s
; RUN: wasm-ld --lto-O3 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=OPT %s
; RUN: wasm-ld --lto-CGO3 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=OPT %s
; RUN: wasm-ld --lto-O0 --lto-CGO0 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=NOOPT %s
; RUN: wasm-ld --lto-O3 --lto-CGO0 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=NOOPT %s
; RUN: wasm-ld --lto-O0 --lto-CGO3 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=OPT %s
; RUN: wasm-ld --lto-O3 --lto-CGO3 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=OPT %s
; RUN: wasm-ld --lto-O0 --lto-CGO0 --lto-CGO2 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=OPT %s
; RUN: not wasm-ld --lto-O4 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=ERROR-O4 %s
; RUN: not wasm-ld --lto-CGO4 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=ERROR-CGO4 %s
; RUN: not wasm-ld --lto-O4 --lto-CGO4 %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefixes=ERROR-O4,ERROR-CGO4 %s

; NOOPT-NOT: WebAssembly Optimize Returned
; OPT: WebAssembly Optimize Returned
; ERROR-O4: wasm-ld: error: invalid optimization level for LTO: 4
; ERROR-CGO4: wasm-ld: error: invalid codegen optimization level for LTO: 4

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown-wasm"

define void @_start() {
entry:
  ret void
}
