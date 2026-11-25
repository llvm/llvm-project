; REQUIRES: x86, plugins, examples
; UNSUPPORTED: target={{.*windows.*}}
; RUN: opt -module-hash -module-summary %s -o %t.o
; RUN: ld.lld -%loadnewpmbye --lto-newpm-passes="goodbye" -mllvm=%loadbye -mllvm=-wave-goodbye %t.o -o /dev/null 2>&1 | FileCheck %s
; RUN: ld.lld -%loadnewpmbye --lto-newpm-passes="goodbye" --pass-plugin-arg=Bye,-wave-goodbye %t.o -o /dev/null 2>&1 | FileCheck %s
; RUN: not ld.lld -%loadnewpmbye --pass-plugin-arg=Nope,-x %t.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNKNOWN

;; Plugin arguments are part of the ThinLTO cache key.
; RUN: rm -rf %t.cache
; RUN: ld.lld -%loadnewpmbye --thinlto-cache-dir=%t.cache %t.o -o /dev/null
; RUN: ld.lld -%loadnewpmbye --thinlto-cache-dir=%t.cache --pass-plugin-arg=Bye,-wave-goodbye %t.o -o /dev/null
; RUN: ls %t.cache | count 3

; CHECK: Bye
; UNKNOWN: error: no pass plugin named 'Nope' is loaded

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
@junk = global i32 0

define ptr @somefunk() {
  ret ptr @junk
}
