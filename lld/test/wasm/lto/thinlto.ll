; Basic ThinLTO tests.
; RUN: opt -module-summary %s -o %t1.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o %t2.o

; First force single-threaded mode
; RUN: rm -f %t31.lto.o %t32.lto.o
; RUN: wasm-ld -r -save-temps --thinlto-jobs=1 %t1.o %t2.o -o %t3
; RUN: llvm-nm %t31.lto.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm %t32.lto.o | FileCheck %s --check-prefix=NM2

; Next force multi-threaded mode
; RUN: rm -f %t31.lto.o %t32.lto.o
; RUN: wasm-ld -r -save-temps --thinlto-jobs=2 %t1.o %t2.o -o %t3
; RUN: llvm-nm %t31.lto.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm %t32.lto.o | FileCheck %s --check-prefix=NM2

;; --thinlto-jobs= defaults to --threads=.
; RUN: rm -f %t31.lto.o %t32.lto.o
; RUN: wasm-ld -r -save-temps --threads=2 %t1.o %t2.o -o %t3
; RUN: llvm-nm %t31.lto.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm %t32.lto.o | FileCheck %s --check-prefix=NM2

;; --thinlto-jobs= overrides --threads=.
; RUN: rm -f %t31.lto.o %t32.lto.o
; RUN: wasm-ld -r -save-temps --threads=1 --thinlto-jobs=2 %t1.o %t2.o -o %t3
; RUN: llvm-nm %t31.lto.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm %t32.lto.o | FileCheck %s --check-prefix=NM2

; Test with all threads, on all cores, on all CPU sockets
; RUN: rm -f %t31.lto.o %t32.lto.o
; RUN: wasm-ld -r -save-temps --thinlto-jobs=all %t1.o %t2.o -o %t3
; RUN: llvm-nm %t31.lto.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm %t32.lto.o | FileCheck %s --check-prefix=NM2

; Test with many more threads than the system has
; RUN: rm -f %t31.lto.o %t32.lto.o
; RUN: wasm-ld -r -save-temps --thinlto-jobs=100 %t1.o %t2.o -o %t3
; RUN: llvm-nm %t31.lto.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm %t32.lto.o | FileCheck %s --check-prefix=NM2

; Test with a bad value
; RUN: rm -f %t31.lto.o %t32.lto.o
; RUN: not wasm-ld -r -save-temps --thinlto-jobs=foo %t1.o %t2.o -o %t3 2>&1 | FileCheck %s --check-prefix=BAD-JOBS
; BAD-JOBS: error: --thinlto-jobs: invalid job count: foo

; Check without --thinlto-jobs (which currently defaults to heavyweight_hardware_concurrency, meanning one thread per hardware core -- not SMT)
; RUN: rm -f %t31.lto.o %t32.lto.o
; RUN: wasm-ld -r -save-temps %t1.o %t2.o -o %t3
; RUN: llvm-nm %t31.lto.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm %t32.lto.o | FileCheck %s --check-prefix=NM2

; NM1: T f
; NM2: T g

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
