; REQUIRES: x86

; Test ThinLTO with time trace
; RUN: opt -module-summary %s -o %t1.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o %t2.o

; Test single-threaded
; RUN: ld.lld --thinlto-jobs=1 --time-trace --time-trace-granularity=0 -shared %t1.o %t2.o -o %t3.so
; RUN: cat %t3.so.time-trace \
; RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
; RUN:   | FileCheck %s

; Test multi-threaded
; RUN: ld.lld --time-trace --time-trace-granularity=0 -shared %t1.o %t2.o -o %t4.so
; RUN: cat %t4.so.time-trace \
; RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
; RUN:   | FileCheck %s

; CHECK: "traceEvents": [
; Check fields for an event are present
; CHECK: "args":
; CHECK-NEXT: "detail":
; CHECK: "dur":
; CHECK-NEXT: "name":
; CHECK-NEXT: "ph":
; CHECK-NEXT: "pid":
; CHECK-NEXT: "tid":
; CHECK-NEXT: "ts":

; Check that an optimisation event is present
; CHECK: "name": "OptModule"

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}

