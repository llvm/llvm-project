; REQUIRES: x86
;; Verify that with --thinlto-index-only and split LTO units (e.g., CFI),
;; markBuffersAsDontNeed runs both after adding bitcode files and again after
;; regular LTO linking of deferred split modules with summaries, before
;; Regular LTO backend / ThinLTO.

; RUN: rm -rf %t && split-file %s %t
; RUN: opt --thinlto-bc --thinlto-split-lto-unit -o %t/a.split.o %t/a.ll
; RUN: opt --thinlto-bc --thinlto-split-lto-unit -o %t/b.split.o %t/b.ll
; RUN: opt --thinlto-bc -o %t/a.thin.o %t/a.ll
; RUN: opt --thinlto-bc -o %t/b.thin.o %t/b.ll

; RUN: ld.lld --thinlto-index-only --time-trace=%t/split.json --time-trace-granularity=0 -shared %t/a.split.o %t/b.split.o -o /dev/null
; RUN: %python -c 'import json, sys; evs = sorted([e for e in json.load(open(sys.argv[1]))["traceEvents"] if e.get("ph") == "X" and not e["name"].startswith("Total ")], key=lambda e: e["ts"]); print("\n".join(e["name"] for e in evs))' %t/split.json | FileCheck %s --check-prefix=SPLIT

; RUN: ld.lld --thinlto-index-only --time-trace=%t/nosplit.json --time-trace-granularity=0 -shared %t/a.thin.o %t/b.thin.o -o /dev/null
; RUN: %python -c 'import json, sys; evs = sorted([e for e in json.load(open(sys.argv[1]))["traceEvents"] if e.get("ph") == "X" and not e["name"].startswith("Total ")], key=lambda e: e["ts"]); print("\n".join(e["name"] for e in evs))' %t/nosplit.json | FileCheck %s --check-prefix=ONCE

; RUN: ld.lld --time-trace=%t/inproc.json --time-trace-granularity=0 -shared %t/a.split.o %t/b.split.o -o %t/out.so
; RUN: %python -c 'import json, sys; evs = sorted([e for e in json.load(open(sys.argv[1]))["traceEvents"] if e.get("ph") == "X" and not e["name"].startswith("Total ")], key=lambda e: e["ts"]); print("\n".join(e["name"] for e in evs))' %t/inproc.json | FileCheck %s --check-prefix=ONCE

; SPLIT: Mark buffers as dontneed
; SPLIT: Link regular LTO
; SPLIT: Mark buffers as dontneed
; SPLIT: LTO backend
; SPLIT: Run ThinLTO

; ONCE: Mark buffers as dontneed
; ONCE-NOT: Mark buffers as dontneed

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTV1A = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1A1fEv] }, !type !0

define void @_ZN1A1fEv(ptr %this) {
  ret void
}

!0 = !{i64 16, !"_ZTS1A"}

;--- b.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @caller(ptr %a, ptr %vtable) {
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %p)
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)
