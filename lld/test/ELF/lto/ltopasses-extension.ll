; REQUIRES: x86, plugins, examples
; UNSUPPORTED: windows
; RUN: opt -module-summary %s -o %t.o
; RUN: ld.lld -%loadnewpmbye --lto-newpm-passes="goodbye" -mllvm=%loadbye -mllvm=-wave-goodbye %t.o -o /dev/null 2>&1 | FileCheck %s
; CHECK: Bye

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
@junk = global i32 0

define i32* @somefunk() {
  ret i32* @junk
}
