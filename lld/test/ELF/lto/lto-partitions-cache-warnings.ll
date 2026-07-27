; REQUIRES: x86

; RUN: rm -rf %t && mkdir %t && cd %t

; RUN: llvm-as -o %t.bc %s

;; Check cache policies of the number of files.
;; Case 1: A value of 0 disables the number of files based pruning. Therefore, there is no warning.
; RUN: ld.lld --verbose --lto-partitions=2 --lto-partitions-cache-dir=%t --lto-partitions-cache-policy=prune_interval=0s:cache_size_files=0 %t.bc -o %t3 2>&1 | FileCheck %s --implicit-check-not=warning:
;; Case 2: If the total number of the files created by the current link job is less than the maximum number of files, there is no warning.
; RUN: ld.lld --verbose --lto-partitions=2 --lto-partitions-cache-dir=%t --lto-partitions-cache-policy=prune_interval=0s:cache_size_files=3 %t.bc -o %t3 2>&1 | FileCheck %s --implicit-check-not=warning:
;; Case 3: If the total number of the files created by the current link job exceeds the maximum number of files, a warning is given.
; RUN: ld.lld --lto-partitions=2 --lto-partitions-cache-dir=%t --lto-partitions-cache-policy=prune_interval=0s:cache_size_files=1 %t.bc -o %t3 2>&1 | FileCheck %s --check-prefixes=NUM,WARN

;; Check cache policies of the cache size.
;; Case 1: A value of 0 disables the absolute size-based pruning. Therefore, there is no warning.
; RUN: ld.lld --verbose --lto-partitions=2 --lto-partitions-cache-dir=%t --lto-partitions-cache-policy=prune_interval=0s:cache_size_bytes=0 %t.bc -o %t3 2>&1 | FileCheck %s --implicit-check-not=warning:

;; Get the total size of created cache files.
; RUN: rm -rf %t && mkdir %t && cd %t
; RUN: ld.lld --lto-partitions=2 --lto-partitions-cache-dir=%t --lto-partitions-cache-policy=prune_interval=0s:cache_size_bytes=32k %t.bc -o %t3 2>&1
; RUN: %python -c "import os, sys; size=sum(os.path.getsize(filename) for filename in os.listdir('.') if os.path.isfile(filename) and filename.startswith('llvmcache-')); print(size+5); print(size-5)" > %t.size.txt

;; Case 2: If the total size of the cache files created by the current link job is less than the maximum size for the cache directory in bytes, there is no warning.
; RUN: echo -n "--lto-partitions-cache-policy=prune_interval=0s:cache_size_bytes=" > %t.response
; RUN: head -1 %t.size.txt >> %t.response
; RUN: ld.lld --verbose --lto-partitions=2 --lto-partitions-cache-dir=%t @%t.response %t.bc -o %t3 2>&1 | FileCheck %s --implicit-check-not=warning:

;; Case 3: If the total size of the cache files created by the current link job exceeds the maximum size for the cache directory in bytes, a warning is given.
; RUN: echo -n "--lto-partitions-cache-policy=prune_interval=0s:cache_size_bytes=" > %t.response
; RUN: tail -1 %t.size.txt >> %t.response
; RUN: ld.lld --verbose --lto-partitions=2 --lto-partitions-cache-dir=%t @%t.response %t.bc -o %t3 2>&1 | FileCheck %s --check-prefixes=SIZE,WARN

;; Check emit two warnings if pruning happens due to reach both the size and number limits.
; RUN: ld.lld --lto-partitions-cache-dir=%t --lto-partitions=2 --lto-partitions-cache-policy=prune_interval=0s:cache_size_files=1:cache_size_bytes=1 %t.bc -o %t3 2>&1 | FileCheck %s --check-prefixes=NUM,SIZE,WARN

; NUM: warning: ThinLTO cache pruning happens since the number of{{.*}}--thinlto-cache-policy
; SIZE: warning: ThinLTO cache pruning happens since the total size of{{.*}}--thinlto-cache-policy
; WARN-NOT: warning: ThinLTO cache pruning happens{{.*}}--thinlto-cache-policy

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() {
  call void @bar()
  ret void
}

define void @bar() {
  call void @foo()
  ret void

}
define i32 @_start() {
entry:
  ret i32 0
}
