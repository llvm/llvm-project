; REQUIRES: x86, shell

; RUN: rm -rf %t; split-file %s %t
; RUN: opt -module-hash -module-summary %t/foo.ll -o %t/foo.o
; RUN: opt -module-hash -module-summary %t/bar.ll -o %t/bar.o

;; Check cache policies of the number of files.
;; Case 1: A value of 0 disables the number of files based pruning. Therefore, there is no warning.
; RUN: %lld -v -cache_path_lto %t --thinlto-cache-policy=prune_interval=0s:cache_size_files=0 -o %t/test %t/foo.o %t/bar.o 2>&1 | FileCheck %s --implicit-check-not=warning:
;; Case 2: If the total number of the files created by the current link job is less than the maximum number of files, there is no warning.
; RUN: %lld -v -cache_path_lto %t --thinlto-cache-policy=prune_interval=0s:cache_size_files=3 %t/foo.o %t/bar.o -o %t/test 2>&1 | FileCheck %s --implicit-check-not=warning:
;; Case 3: If the total number of the files created by the current link job exceeds the maximum number of files, a warning is given.
; RUN: %lld -cache_path_lto %t --thinlto-cache-policy=prune_interval=0s:cache_size_files=1 %t/foo.o %t/bar.o -o %t/test 2>&1 | FileCheck %s --check-prefixes=NUM,WARN

;; Check cache policies of the cache size.
;; Case 1: A value of 0 disables the absolute size-based pruning. Therefore, there is no warning.
; RUN: %lld -v -cache_path_lto %t --thinlto-cache-policy=prune_interval=0s:cache_size_bytes=0 %t/foo.o %t/bar.o -o %t/test 2>&1 | FileCheck %s --implicit-check-not=warning:

;; Get the total size of created cache files.
; RUN: cd %t
; RUN: %lld -cache_path_lto %t --thinlto-cache-policy=prune_interval=0s:cache_size_bytes=32k %t/foo.o %t/bar.o -o %t/test 2>&1
; RUN: %python -c "import os, sys; print(sum(os.path.getsize(filename) for filename in os.listdir('.') if os.path.isfile(filename) and filename.startswith('llvmcache-')))" > %t.size.txt

;; Case 2: If the total size of the cache files created by the current link job is less than the maximum size for the cache directory in bytes, there is no warning.
; RUN: %lld -v -cache_path_lto %t --thinlto-cache-policy=prune_interval=0s:cache_size_bytes=$(($(cat %t.size.txt) + 5)) %t/foo.o %t/bar.o -o %t/test 2>&1 | FileCheck %s --implicit-check-not=warning:

;; Case 3: If the total size of the cache files created by the current link job exceeds the maximum size for the cache directory in bytes, a warning is given.
; RUN: %lld -v -cache_path_lto %t --thinlto-cache-policy=prune_interval=0s:cache_size_bytes=$(($(cat %t.size.txt) - 5)) %t/foo.o %t/bar.o -o %t/test 2>&1 | FileCheck %s --check-prefixes=SIZE,WARN

;; Check emit two warnings if pruning happens due to reach both the size and number limits.
; RUN: %lld -cache_path_lto %t --thinlto-cache-policy=prune_interval=0s:cache_size_files=1:cache_size_bytes=1 %t/foo.o %t/bar.o -o %t/test 2>&1 | FileCheck %s --check-prefixes=NUM,SIZE,WARN

; NUM: warning: ThinLTO cache pruning happens since the number of{{.*}}--thinlto-cache-policy
; SIZE: warning: ThinLTO cache pruning happens since the total size of{{.*}}--thinlto-cache-policy
; WARN-NOT: warning: ThinLTO cache pruning happens{{.*}}--thinlto-cache-policy

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
