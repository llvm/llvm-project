; REQUIRES: x86

; RUN: opt -module-hash -module-summary %s -o %t.o
; RUN: opt -module-hash -module-summary %p/Inputs/cache.ll -o %t2.o

; RUN: rm -Rf %t.cache && mkdir %t.cache
; Create a file that will be removed by cache pruning due to age.
; RUN: touch -t 197001010000 %t.cache/foo
; RUN: ld.lld --thinlto-cache-dir=%t.cache --thinlto-cache-policy prune_after=1h -o %t3 %t2.o %t.o

; Two cached objects, plus a timestamp file, minus the file we removed.
; RUN: ls %t.cache | count 3

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @globalfunc() #0 {
entry:
  ret void
}
