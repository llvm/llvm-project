; REQUIRES: x86
; NetBSD: noatime mounts currently inhibit 'touch' from updating atime
; UNSUPPORTED: system-netbsd

; RUN: rm -rf %t; split-file %s %t
; RUN: opt -module-hash -module-summary %t/foo.ll -o %t/foo.o
; RUN: opt -module-hash -module-summary %t/bar.ll -o %t/bar.o

; RUN: rm -Rf %t/cache && mkdir %t/cache
;; Create two files that would be removed by cache pruning due to age.
;; We should only remove files matching the pattern "llvmcache-*".
; RUN: touch -t 197001011200 %t/cache/llvmcache-baz %t/cache/baz
; RUN: %lld -cache_path_lto %t/cache \
; RUN:   --thinlto-cache-policy=prune_after=1h:prune_interval=0s \
; RUN:   -o %t/test %t/foo.o %t/bar.o

;; Two cached objects, plus a timestamp file and "baz", minus the file we removed.
; RUN: ls %t/cache | count 4

;; Same thing, but with `-prune_after_lto`
; RUN: touch -t 197001011200 %t/cache/llvmcache-baz
; RUN: %lld -cache_path_lto %t/cache -prune_after_lto 3600 -prune_interval_lto 0 \
; RUN:   -o %t/test %t/foo.o %t/bar.o
; RUN: ls %t/cache | count 4

;; Create a file of size 64KB.
; RUN: %python -c "print(' ' * 65536)" > %t/cache/llvmcache-baz

;; This should leave the file in place.
; RUN: %lld -cache_path_lto %t/cache \
; RUN:   --thinlto-cache-policy=cache_size_bytes=128k:prune_interval=0s \
; RUN:   -o %t/test %t/foo.o %t/bar.o
; RUN: ls %t/cache | count 5

;; Increase the age of llvmcache-baz, which will give it the oldest time stamp
;; so that it is processed and removed first.
; RUN: %python -c 'import os,sys,time; t=time.time()-120; os.utime(sys.argv[1],(t,t))' \
; RUN:    %t/cache/llvmcache-baz

;; This should remove it.
; RUN: %lld -cache_path_lto %t/cache \
; RUN:   --thinlto-cache-policy=cache_size_bytes=32k:prune_interval=0s \
; RUN:   -o %t/test %t/foo.o %t/bar.o
; RUN: ls %t/cache | count 4

;; Delete everything except for the timestamp, "baz" and one cache file.
; RUN: %lld -cache_path_lto %t/cache \
; RUN:   --thinlto-cache-policy=prune_after=0s:cache_size=0%:cache_size_files=1:prune_interval=0s \
; RUN:   -o %t/test %t/foo.o %t/bar.o
; RUN: ls %t/cache | count 3

;; Check that we remove the least recently used file first.
; RUN: rm -fr %t/cache
; RUN: mkdir %t/cache
; RUN: echo xyz > %t/cache/llvmcache-old
; RUN: touch -t 198002011200 %t/cache/llvmcache-old
; RUN: echo xyz > %t/cache/llvmcache-newer
; RUN: touch -t 198002021200 %t/cache/llvmcache-newer
; RUN: %lld -cache_path_lto %t/cache \
; RUN:   --thinlto-cache-policy=prune_after=0s:cache_size=0%:cache_size_files=3:prune_interval=0s \
; RUN:   -o %t/test %t/foo.o %t/bar.o
; RUN: ls %t/cache | FileCheck %s

;; Check that  `-max_relative_cache_size_lto` is a legal argument.
; RUN: %lld -cache_path_lto %t/cache -max_relative_cache_size_lto 10 \
; RUN:   -o %t/test %t/foo.o %t/bar.o

; CHECK-NOT: llvmcache-old
; CHECK: llvmcache-newer
; CHECK-NOT: llvmcache-old

;; Check that mllvm options participate in the cache key
; RUN: rm -rf %t/cache && mkdir %t/cache
; RUN: %lld -cache_path_lto %t/cache -o %t/test %t/foo.o %t/bar.o
; RUN: ls %t/cache | count 3
; RUN: %lld -cache_path_lto %t/cache -o %t/test %t/foo.o %t/bar.o -mllvm -enable-ml-inliner=default
; RUN: ls %t/cache | count 5

;; Adding another option resuls in 2 more cache entries
; RUN: rm -rf %t/cache && mkdir %t/cache
; RUN: %lld -cache_path_lto %t/cache -o %t/test %t/foo.o %t/bar.o
; RUN: ls %t/cache | count 3
; RUN: %lld -cache_path_lto %t/cache -o %t/test %t/foo.o %t/bar.o -mllvm -enable-ml-inliner=default
; RUN: ls %t/cache | count 5
; RUN: %lld -cache_path_lto %t/cache -o %t/test %t/foo.o %t/bar.o -mllvm -enable-ml-inliner=default -mllvm -max-devirt-iterations=1
; RUN: ls %t/cache | count 7

;; Changing order may matter - e.g. if overriding -mllvm options - so we get 2 more entries
; RUN: %lld -cache_path_lto %t/cache -o %t/test %t/foo.o %t/bar.o -mllvm -max-devirt-iterations=1 -mllvm -enable-ml-inliner=default
; RUN: ls %t/cache | count 9

;; Going back to a pre-cached order doesn't create more entries.
; RUN: %lld -cache_path_lto %t/cache -o %t/test %t/foo.o %t/bar.o -mllvm -enable-ml-inliner=default -mllvm -max-devirt-iterations=1
; RUN: ls %t/cache | count 9

;; Different flag values matter
; RUN: rm -rf %t/cache && mkdir %t/cache
; RUN: %lld -cache_path_lto %t/cache -o %t/test %t/foo.o %t/bar.o -mllvm -enable-ml-inliner=default -mllvm -max-devirt-iterations=2
; RUN: ls %t/cache | count 3
; RUN: %lld -cache_path_lto %t/cache -o %t/test %t/foo.o %t/bar.o -mllvm -enable-ml-inliner=default -mllvm -max-devirt-iterations=1
; RUN: ls %t/cache | count 5

;; Same flag value passed to different flags matters, and switching the order
;; of the two flags matters.
; RUN: rm -rf %t/cache && mkdir %t/cache
; RUN: %lld -cache_path_lto %t/cache -o %t/test %t/foo.o %t/bar.o -mllvm -enable-ml-inliner=default
; RUN: ls %t/cache | count 3
; RUN: %lld -cache_path_lto %t/cache -o %t/test %t/foo.o %t/bar.o -mllvm -emit-dwarf-unwind=default
; RUN: ls %t/cache | count 5
; RUN: %lld -cache_path_lto %t/cache -o %t/test %t/foo.o %t/bar.o -mllvm -enable-ml-inliner=default
; RUN: ls %t/cache | count 5
; RUN: %lld -cache_path_lto %t/cache -o %t/test %t/foo.o %t/bar.o -mllvm -enable-ml-inliner=default -mllvm -emit-dwarf-unwind=default
; RUN: ls %t/cache | count 7
; RUN: %lld -cache_path_lto %t/cache -o %t/test %t/foo.o %t/bar.o -mllvm -emit-dwarf-unwind=default -mllvm -enable-ml-inliner=default
; RUN: ls %t/cache | count 9

;--- foo.ll

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @globalfunc() #0 {
entry:
  ret void
}


;--- bar.ll

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define i32 @main() {
entry:
  call void (...) @globalfunc()
  ret i32 0
}

declare void @globalfunc(...)
