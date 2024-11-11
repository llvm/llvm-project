; REQUIRES: x86, plugins, examples

; RUN: opt -module-summary %s -o %t.o
; RUN: %lld -dylib -%loadnewpmbye --lto-newpm-passes="goodbye" -mllvm %loadbye -mllvm -wave-goodbye %t.o -o /dev/null 2>&1 | FileCheck %s
; CHECK: Bye

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
@junk = global i32 0

define ptr @somefunk() {
  ret ptr @junk
}
