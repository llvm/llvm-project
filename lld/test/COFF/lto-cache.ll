; REQUIRES: x86
; NetBSD: noatime mounts currently inhibit 'touch' from updating atime
; UNSUPPORTED: system-netbsd

; RUN: opt -module-hash -module-summary %s -o %t.o
; RUN: opt -module-hash -module-summary %p/Inputs/lto-cache.ll -o %t2.o

; RUN: rm -Rf %t.cache && mkdir %t.cache
; Create two files that would be removed by cache pruning due to age.
; We should only remove files matching "llvmcache-*" or "Thin-*".
; RUN: touch -t 197001011200 %t.cache/llvmcache-foo %t.cache/Thin-123.tmp.o %t.cache/foo
; RUN: lld-link /lldltocache:%t.cache /lldltocachepolicy:prune_after=1h /out:%t3 /entry:main %t2.o %t.o

; Two cached objects, plus a timestamp file and "foo", minus the file we removed.
; RUN: ls %t.cache | count 4

;; Check that mllvm options participate in the cache key
; RUN: rm -rf %t.cache && mkdir %t.cache
; RUN: lld-link /lldltocache:%t.cache /entry:main /out:%t3 %t2.o %t.o
; RUN: ls %t.cache | count 3
; RUN: lld-link /lldltocache:%t.cache /entry:main /out:%t3 %t2.o %t.o /mllvm:-enable-ml-inliner=default
; RUN: ls %t.cache | count 5

;; Adding another option resuls in 2 more cache entries
; RUN: rm -rf %t.cache && mkdir %t.cache
; RUN: lld-link /lldltocache:%t.cache /entry:main /out:%t3 %t2.o %t.o
; RUN: ls %t.cache | count 3
; RUN: lld-link /lldltocache:%t.cache /entry:main /out:%t3 %t2.o %t.o /mllvm:-enable-ml-inliner=default
; RUN: ls %t.cache | count 5
; RUN: lld-link /lldltocache:%t.cache /entry:main /out:%t3 %t2.o %t.o /mllvm:-enable-ml-inliner=default /mllvm:-max-devirt-iterations=1
; RUN: ls %t.cache | count 7

;; Changing order may matter - e.g. if overriding /mllvm:options - so we get 2 more entries
; RUN: lld-link /lldltocache:%t.cache /entry:main /out:%t3 %t2.o %t.o /mllvm:-max-devirt-iterations=1 /mllvm:-enable-ml-inliner=default
; RUN: ls %t.cache | count 9

;; Going back to a pre-cached order doesn't create more entries.
; RUN: lld-link /lldltocache:%t.cache /entry:main /out:%t3 %t2.o %t.o /mllvm:-enable-ml-inliner=default /mllvm:-max-devirt-iterations=1
; RUN: ls %t.cache | count 9

;; Different flag values matter
; RUN: rm -rf %t.cache && mkdir %t.cache
; RUN: lld-link /lldltocache:%t.cache /entry:main /out:%t3 %t2.o %t.o /mllvm:-enable-ml-inliner=default /mllvm:-max-devirt-iterations=2
; RUN: ls %t.cache | count 3
; RUN: lld-link /lldltocache:%t.cache /entry:main /out:%t3 %t2.o %t.o /mllvm:-enable-ml-inliner=default /mllvm:-max-devirt-iterations=1
; RUN: ls %t.cache | count 5

;; Same flag value passed to different flags matters, and switching the order
;; of the two flags matters.
; RUN: rm -rf %t.cache && mkdir %t.cache
; RUN: lld-link /lldltocache:%t.cache /entry:main /out:%t3 %t2.o %t.o /mllvm:-enable-ml-inliner=default
; RUN: ls %t.cache | count 3
; RUN: lld-link /lldltocache:%t.cache /entry:main /out:%t3 %t2.o %t.o /mllvm:-emit-dwarf-unwind=default
; RUN: ls %t.cache | count 5
; RUN: lld-link /lldltocache:%t.cache /entry:main /out:%t3 %t2.o %t.o /mllvm:-enable-ml-inliner=default
; RUN: ls %t.cache | count 5
; RUN: lld-link /lldltocache:%t.cache /entry:main /out:%t3 %t2.o %t.o /mllvm:-enable-ml-inliner=default /mllvm:-emit-dwarf-unwind=default
; RUN: ls %t.cache | count 7
; RUN: lld-link /lldltocache:%t.cache /entry:main /out:%t3 %t2.o %t.o /mllvm:-emit-dwarf-unwind=default /mllvm:-enable-ml-inliner=default
; RUN: ls %t.cache | count 9

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @globalfunc() #0 {
entry:
  ret void
}
