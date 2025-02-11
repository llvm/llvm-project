; Basic ThinLTO tests.
; RUN: rm -rf %t && mkdir %t && cd %t
; RUN: mkdir d e

; RUN: opt -module-summary %s -o a.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o d/b.o

; First force single-threaded mode
; RUN: rm -f out.lto.a.o d/out.lto.b.o
; RUN: wasm-ld -r -save-temps --thinlto-jobs=1 a.o d/b.o -o e/out
; RUN: llvm-nm out.lto.a.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm d/out.lto.b.o | FileCheck %s --check-prefix=NM2

; Next force multi-threaded mode
; RUN: rm -f out.lto.a.o d/out.lto.b.o
; RUN: wasm-ld -r -save-temps --thinlto-jobs=2 a.o d/b.o -o e/out
; RUN: llvm-nm out.lto.a.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm d/out.lto.b.o | FileCheck %s --check-prefix=NM2

;; --thinlto-jobs= defaults to --threads=.
; RUN: rm -f out.lto.a.o d/out.lto.b.o
; RUN: wasm-ld -r -save-temps --threads=2 a.o d/b.o -o e/out
; RUN: llvm-nm out.lto.a.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm d/out.lto.b.o | FileCheck %s --check-prefix=NM2

;; --thinlto-jobs= overrides --threads=.
; RUN: rm -f out.lto.a.o d/out.lto.b.o
; RUN: wasm-ld -r -save-temps --threads=1 --thinlto-jobs=2 a.o d/b.o -o e/out
; RUN: llvm-nm out.lto.a.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm d/out.lto.b.o | FileCheck %s --check-prefix=NM2

; Test with all threads, on all cores, on all CPU sockets
; RUN: rm -f out.lto.a.o d/out.lto.b.o
; RUN: wasm-ld -r -save-temps --thinlto-jobs=all a.o d/b.o -o e/out
; RUN: llvm-nm out.lto.a.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm d/out.lto.b.o | FileCheck %s --check-prefix=NM2

; Test with many more threads than the system has
; RUN: rm -f out.lto.a.o d/out.lto.b.o
; RUN: wasm-ld -r -save-temps --thinlto-jobs=100 a.o d/b.o -o e/out
; RUN: llvm-nm out.lto.a.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm d/out.lto.b.o | FileCheck %s --check-prefix=NM2

; Test with a bad value
; RUN: rm -f out.lto.a.o d/out.lto.b.o
; RUN: not wasm-ld -r -save-temps --thinlto-jobs=foo a.o d/b.o -o e/out 2>&1 | FileCheck %s --check-prefix=BAD-JOBS
; BAD-JOBS: error: --thinlto-jobs: invalid job count: foo

; Check without --thinlto-jobs (which currently defaults to heavyweight_hardware_concurrency, meanning one thread per hardware core -- not SMT)
; RUN: rm -f out.lto.a.o d/out.lto.b.o
; RUN: wasm-ld -r -save-temps a.o d/b.o -o e/out
; RUN: llvm-nm out.lto.a.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm d/out.lto.b.o | FileCheck %s --check-prefix=NM2

; NM1: T f
; NM2: T g

target triple = "wasm32-unknown-unknown"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
