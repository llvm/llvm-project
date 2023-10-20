; REQUIRES: x86

; RUN: opt -module-hash -module-summary %s -o %t.o
; RUN: opt -module-hash -module-summary %p/Inputs/lto-cache.ll -o %t2.o
; RUN: rm -Rf %t.cache && mkdir %t.cache
; RUN: chmod 444 %t.cache

;; Check emit warnings when we can't create the cache dir, note that this test will pass if there is a o: drive.
; RUN: not lld-link /lldltocache:%t.cache/cache /out:%t3 /entry:main %t2.o %t.o 2>&1 | FileCheck %s
; CHECK: LLVM ERROR: Can't create cache directory

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @globalfunc() #0 {
entry:
  ret void
}
