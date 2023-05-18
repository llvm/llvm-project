; REQUIRES: shell

; RUN: opt -module-hash -module-summary %s -o %t.o
; RUN: opt -module-hash -module-summary %p/Inputs/cache.ll -o %t2.o

; RUN: rm -rf %t && mkdir %t

;; Check cache policies of the number of files.
;; Case 1: A value of 0 disables the number of files based pruning. Therefore, there is no warning.
; RUN: wasm-ld --verbose --thinlto-cache-dir=%t --thinlto-cache-policy=prune_interval=0s:cache_size_files=0 %t2.o %t.o -o %t3 2>&1 | FileCheck %s --implicit-check-not=warning:
;; Case 2: If the total number of the files created by the current link job is less than the maximum number of files, there is no warning.
; RUN: wasm-ld --verbose --thinlto-cache-dir=%t --thinlto-cache-policy=prune_interval=0s:cache_size_files=3 %t2.o %t.o -o %t3 2>&1 | FileCheck %s --implicit-check-not=warning:
;; Case 3: If the total number of the files created by the current link job exceeds the maximum number of files, a warning is given.
; RUN: wasm-ld --thinlto-cache-dir=%t --thinlto-cache-policy=prune_interval=0s:cache_size_files=1 %t2.o %t.o -o %t3 2>&1 | FileCheck %s --check-prefixes=NUM,WARN

;; Check cache policies of the cache size.
;; Case 1: A value of 0 disables the absolute size-based pruning. Therefore, there is no warning.
; RUN: wasm-ld --verbose --thinlto-cache-dir=%t --thinlto-cache-policy=prune_interval=0s:cache_size_bytes=0 %t2.o %t.o -o %t3 2>&1 | FileCheck %s --implicit-check-not=warning:

;; Get the total size of created cache files.
; RUN: rm -rf %t && mkdir %t && cd %t
; RUN: wasm-ld --thinlto-cache-dir=%t --thinlto-cache-policy=prune_interval=0s:cache_size_bytes=32k %t2.o %t.o -o %t3 2>&1
; RUN: %python -c "import os, sys; print(sum(os.path.getsize(filename) for filename in os.listdir('.') if os.path.isfile(filename) and filename.startswith('llvmcache-')))" > %t.size.txt

;; Case 2: If the total size of the cache files created by the current link job is less than the maximum size for the cache directory in bytes, there is no warning.
; RUN: wasm-ld --verbose --thinlto-cache-dir=%t --thinlto-cache-policy=prune_interval=0s:cache_size_bytes=$(($(cat %t.size.txt) + 5)) %t2.o %t.o -o %t3 2>&1 | FileCheck %s --implicit-check-not=warning:

;; Case 3: If the total size of the cache files created by the current link job exceeds the maximum size for the cache directory in bytes, a warning is given.
; RUN: wasm-ld --verbose --thinlto-cache-dir=%t --thinlto-cache-policy=prune_interval=0s:cache_size_bytes=$(($(cat %t.size.txt) - 5)) %t2.o %t.o -o %t3 2>&1 | FileCheck %s --check-prefixes=SIZE,WARN

;; Check emit two warnings if pruning happens due to reach both the size and number limits.
; RUN: wasm-ld --thinlto-cache-dir=%t --thinlto-cache-policy=prune_interval=0s:cache_size_files=1:cache_size_bytes=1 %t2.o %t.o -o %t3 2>&1 | FileCheck %s --check-prefixes=NUM,SIZE,WARN

; NUM: warning: ThinLTO cache pruning happens since the number of{{.*}}--thinlto-cache-policy
; SIZE: warning: ThinLTO cache pruning happens since the total size of{{.*}}--thinlto-cache-policy
; WARN-NOT: warning: ThinLTO cache pruning happens{{.*}}--thinlto-cache-policy

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

define void @globalfunc() #0 {
entry:
  ret void
}
