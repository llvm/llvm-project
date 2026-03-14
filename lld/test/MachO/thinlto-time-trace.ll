; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t

; Test ThinLTO with time trace
; RUN: opt -module-summary %t/f.s -o %t/f.o
; RUN: opt -module-summary %t/g.s -o %t/g.o
; RUN: %lld --time-trace --time-trace-granularity=0 -dylib %t/f.o %t/g.o -o %t/libTest.dylib
; RUN: cat %t/libTest.dylib.time-trace \
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

; Check that an optimization event is present
; CHECK: "name": "OptModule"

;--- f.s
target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}

;--- g.s
target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @g() {
entry:
  ret void
}
