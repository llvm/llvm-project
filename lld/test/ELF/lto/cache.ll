; REQUIRES: x86
; NetBSD: noatime mounts currently inhibit 'touch' from updating atime
; UNSUPPORTED: system-netbsd

; RUN: rm -rf %t && mkdir %t && cd %t
; RUN: opt -module-hash -module-summary %s -o a.bc
; RUN: opt -module-hash -module-summary %p/Inputs/cache.ll -o b.bc

; RUN: mkdir cache
; Create two files that would be removed by cache pruning due to age.
; We should only remove files matching the pattern "llvmcache-*".
; RUN: touch -t 197001011200 cache/llvmcache-foo cache/foo
; RUN: ld.lld --thinlto-cache-dir=cache --thinlto-cache-policy prune_after=1h:prune_interval=0s -o out b.bc a.bc

; Two cached objects, plus a timestamp file and "foo", minus the file we removed.
; RUN: ls cache | count 4

; Create a file of size 64KB.
; RUN: %python -c "print(' ' * 65536)" > cache/llvmcache-foo

; This should leave the file in place.
; RUN: ld.lld --thinlto-cache-dir=cache --thinlto-cache-policy cache_size_bytes=128k:prune_interval=0s -o out b.bc a.bc
; RUN: ls cache | count 5

; Increase the age of llvmcache-foo, which will give it the oldest time stamp
; so that it is processed and removed first.
; RUN: %python -c 'import os,sys,time; t=time.time()-120; os.utime(sys.argv[1],(t,t))' cache/llvmcache-foo

; This should remove it.
; RUN: ld.lld --thinlto-cache-dir=cache --thinlto-cache-policy cache_size_bytes=32k:prune_interval=0s -o out b.bc a.bc
; RUN: ls cache | count 4

; Setting max number of files to 0 should disable the limit, not delete everything.
; RUN: ld.lld --thinlto-cache-dir=cache --thinlto-cache-policy prune_after=0s:cache_size=0%:cache_size_files=0:prune_interval=0s -o out b.bc a.bc
; RUN: ls cache | count 4

; Delete everything except for the timestamp, "foo" and one cache file.
; RUN: ld.lld --thinlto-cache-dir=cache --thinlto-cache-policy prune_after=0s:cache_size=0%:cache_size_files=1:prune_interval=0s -o out b.bc a.bc
; RUN: ls cache | count 3

; Check that we remove the least recently used file first.
; RUN: rm -fr cache && mkdir cache
; RUN: echo xyz > cache/llvmcache-old
; RUN: touch -t 198002011200 cache/llvmcache-old
; RUN: echo xyz > cache/llvmcache-newer
; RUN: touch -t 198002021200 cache/llvmcache-newer
; RUN: ld.lld --thinlto-cache-dir=cache --thinlto-cache-policy prune_after=0s:cache_size=0%:cache_size_files=3:prune_interval=0s -o out b.bc a.bc
; RUN: ls cache | FileCheck %s

; CHECK-NOT: llvmcache-old
; CHECK: llvmcache-newer
; CHECK-NOT: llvmcache-old

; RUN: rm -fr cache && mkdir cache
; RUN: ld.lld --thinlto-cache-dir=cache --save-temps -o out b.bc a.bc -M | FileCheck %s --check-prefix=MAP
; RUN: ls out1.lto.o a.bc.0.preopt.bc b.bc.0.preopt.bc

; MAP: llvmcache-{{.*}}:(.text)
; MAP: llvmcache-{{.*}}:(.text)

;; Check that mllvm options participate in the cache key
; RUN: rm -rf cache && mkdir cache
; RUN: ld.lld --thinlto-cache-dir=cache -o out b.bc a.bc
; RUN: ls cache | count 3
; RUN: ld.lld --thinlto-cache-dir=cache -o out b.bc a.bc -mllvm -enable-ml-inliner=default
; RUN: ls cache | count 5

;; Adding another option resuls in 2 more cache entries
; RUN: rm -rf cache && mkdir cache
; RUN: ld.lld --thinlto-cache-dir=cache -o out b.bc a.bc
; RUN: ls cache | count 3
; RUN: ld.lld --thinlto-cache-dir=cache -o out b.bc a.bc -mllvm -enable-ml-inliner=default
; RUN: ls cache | count 5
; RUN: ld.lld --thinlto-cache-dir=cache -o out b.bc a.bc -mllvm -enable-ml-inliner=default -mllvm -max-devirt-iterations=1
; RUN: ls cache | count 7

;; Changing order may matter - e.g. if overriding -mllvm options - so we get 2 more entries
; RUN: ld.lld --thinlto-cache-dir=cache -o out b.bc a.bc -mllvm -max-devirt-iterations=1 -mllvm -enable-ml-inliner=default
; RUN: ls cache | count 9

;; Going back to a pre-cached order doesn't create more entries.
; RUN: ld.lld --thinlto-cache-dir=cache -o out b.bc a.bc -mllvm -enable-ml-inliner=default -mllvm -max-devirt-iterations=1
; RUN: ls cache | count 9

;; Different flag values matter
; RUN: rm -rf cache && mkdir cache
; RUN: ld.lld --thinlto-cache-dir=cache -o out b.bc a.bc -mllvm -enable-ml-inliner=default -mllvm -max-devirt-iterations=2
; RUN: ls cache | count 3
; RUN: ld.lld --thinlto-cache-dir=cache -o out b.bc a.bc -mllvm -enable-ml-inliner=default -mllvm -max-devirt-iterations=1
; RUN: ls cache | count 5

;; Same flag value passed to different flags matters, and switching the order
;; of the two flags matters.
; RUN: rm -rf cache && mkdir cache
; RUN: ld.lld --thinlto-cache-dir=cache -o out b.bc a.bc -mllvm -enable-ml-inliner=default
; RUN: ls cache | count 3
; RUN: ld.lld --thinlto-cache-dir=cache -o out b.bc a.bc -mllvm -emit-dwarf-unwind=default
; RUN: ls cache | count 5
; RUN: ld.lld --thinlto-cache-dir=cache -o out b.bc a.bc -mllvm -enable-ml-inliner=default
; RUN: ls cache | count 5
; RUN: ld.lld --thinlto-cache-dir=cache -o out b.bc a.bc -mllvm -enable-ml-inliner=default -mllvm -emit-dwarf-unwind=default
; RUN: ls cache | count 7
; RUN: ld.lld --thinlto-cache-dir=cache -o out b.bc a.bc -mllvm -emit-dwarf-unwind=default -mllvm -enable-ml-inliner=default
; RUN: ls cache | count 9

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @globalfunc() #0 {
entry:
  ret void
}
