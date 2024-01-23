; REQUIRES: x86

; Basic ThinLTO tests.
; RUN: rm -rf %t && mkdir %t && cd %t
; RUN: mkdir d e
; RUN: opt -module-summary %s -o a.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o d/b.o

; First force single-threaded mode
; RUN: rm -f e/out1.lto.o e/out2.lto.o
; RUN: ld.lld -save-temps --thinlto-jobs=1 -shared a.o d/b.o -o e/out
; RUN: llvm-nm e/out1.lto.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm e/out2.lto.o | FileCheck %s --check-prefix=NM2

; Next force multi-threaded mode
; RUN: rm -f e/out1.lto.o e/out2.lto.o
; RUN: ld.lld -save-temps --thinlto-jobs=2 -shared a.o d/b.o -o e/out
; RUN: llvm-nm e/out1.lto.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm e/out2.lto.o | FileCheck %s --check-prefix=NM2

;; --plugin-opt=jobs= is an alias.
; RUN: rm -f e/out1.lto.o e/out2.lto.o
; RUN: ld.lld -save-temps --plugin-opt=jobs=2 -shared a.o d/b.o -o e/out
; RUN: llvm-nm e/out1.lto.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm e/out2.lto.o | FileCheck %s --check-prefix=NM2

;; --thinlto-jobs= defaults to --threads=.
; RUN: rm -f e/out1.lto.o e/out2.lto.o
; RUN: ld.lld -save-temps --threads=2 -shared a.o d/b.o -o e/out
; RUN: llvm-nm e/out1.lto.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm e/out2.lto.o | FileCheck %s --check-prefix=NM2

;; --thinlto-jobs= overrides --threads=.
; RUN: rm -f e/out1.lto.o e/out2.lto.o
; RUN: ld.lld -save-temps --threads=1 --plugin-opt=jobs=2 -shared a.o d/b.o -o e/out
; RUN: llvm-nm e/out1.lto.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm e/out2.lto.o | FileCheck %s --check-prefix=NM2

; Test with all threads, on all cores, on all CPU sockets
; RUN: rm -f e/out1.lto.o e/out2.lto.o
; RUN: ld.lld -save-temps --thinlto-jobs=all -shared a.o d/b.o -o e/out
; RUN: llvm-nm e/out1.lto.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm e/out2.lto.o | FileCheck %s --check-prefix=NM2

; Test with many more threads than the system has
; RUN: rm -f e/out1.lto.o e/out2.lto.o
; RUN: ld.lld -save-temps --thinlto-jobs=100 -shared a.o d/b.o -o e/out
; RUN: llvm-nm e/out1.lto.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm e/out2.lto.o | FileCheck %s --check-prefix=NM2

; Test with a bad value
; RUN: rm -f e/out1.lto.o e/out2.lto.o
; RUN: not ld.lld -save-temps --thinlto-jobs=foo -shared a.o d/b.o -o e/out 2>&1 | FileCheck %s --check-prefix=BAD-JOBS
; BAD-JOBS: error: --thinlto-jobs: invalid job count: foo

; Then check without --thinlto-jobs (which currently defaults to heavyweight_hardware_concurrency, meanning one thread per hardware core -- not SMT)
; RUN: ld.lld -shared -save-temps a.o d/b.o -o e/out
; RUN: llvm-nm e/out1.lto.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm e/out2.lto.o | FileCheck %s --check-prefix=NM2

; Check that -save-temps is usable with thin archives
; RUN: mkdir dir
; RUN: cp d/b.o dir/t.o
; RUN: llvm-ar rcsT dir/t.a dir/t.o
; RUN: ld.lld -save-temps a.o dir/t.a -o %t.null
; RUN: ls dir/t.a*.0.preopt.bc

; NM1: T f
; NM2: T g

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
