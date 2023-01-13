; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t

;; First ensure that the ThinLTO handling in lld handles
;; bitcode without summary sections gracefully and generates index file.
; RUN: llvm-as %t/f.ll -o %t/1.o
; RUN: llvm-as %t/g.ll -o %t/2.o
; RUN: %lld --thinlto-index-only -dylib %t/1.o %t/2.o -o %t/3
; RUN: ls %t/2.o.thinlto.bc
; RUN: not test -e %t/3
; RUN: %lld -dylib %t/1.o %t/2.o -o %t/3
; RUN: llvm-nm %t/3 | FileCheck %s --check-prefix=NM

;; Basic ThinLTO tests.
; RUN: opt -module-summary %t/f.ll -o %t/1.o
; RUN: opt -module-summary %t/g.ll -o %t/2.o
; RUN: opt -module-summary %t/empty.ll -o %t/3.o

;; Ensure lld doesn't generates index files when thinlto-index-only is not enabled.
; RUN: rm -f %t/1.o.thinlto.bc %t/2.o.thinlto.bc %t/3.o.thinlto.bc
; RUN: %lld -dylib %t/1.o %t/2.o %t/3.o -o %t/5
; RUN: not ls %t/1.o.thinlto.bc
; RUN: not ls %t/2.o.thinlto.bc
; RUN: not ls %t/3.o.thinlto.bc

;; Ensure lld generates an index and not a binary if requested.
; RUN: %lld --thinlto-index-only -dylib %t/1.o %t/2.o -o %t/4
; RUN: llvm-bcanalyzer -dump %t/1.o.thinlto.bc | FileCheck %s -DP1=%t/1.o -DP2=%t/2.o --check-prefix=BACKEND1
; RUN: llvm-bcanalyzer -dump %t/2.o.thinlto.bc | FileCheck %s -DP2=%t/2.o --check-prefix=BACKEND2
; RUN: not test -e %t/4

;; Ensure lld generates an index even if the file is wrapped in --start-lib/--end-lib.
; RUN: rm -f %t/2.o.thinlto.bc %t/4
; RUN: %lld --thinlto-index-only -dylib %t/1.o %t/3.o --start-lib %t/2.o --end-lib -o %t/4
; RUN: llvm-dis < %t/2.o.thinlto.bc | grep -q '\^0 = module:'
; RUN: not test -e %t/4

;; Test that LLD generates an empty index even for lazy object file that is not added to link.
;; Test LLD generates empty imports file either because of thinlto-emit-imports-files option.
; RUN: rm -f %t/1.o.thinlto.bc %t/1.o.imports
; RUN: %lld --thinlto-index-only -dylib %t/2.o --start-lib %t/1.o --end-lib \
; RUN:      --thinlto-emit-imports-files -o %t/3
; RUN: ls %t/1.o.thinlto.bc
; RUN: ls %t/1.o.imports

;; Ensure LLD generates an empty index for each bitcode file even if all bitcode files are lazy.
; RUN: rm -f %t/dummy.o %t/1.o.thinlto.bc
; RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin /dev/null -o %t/dummy.o
; RUN: %lld --thinlto-index-only -dylib %t/dummy.o --start-lib %t/1.o --end-lib -o /dev/null
; RUN: ls %t/1.o.thinlto.bc

;; Ensure when the same bitcode object is given as both lazy and non-lazy,
;; LLD does not generate an empty index for the lazy object.
; RUN: rm -f %t/2.o.thinlto.bc
; RUN: %lld --thinlto-index-only -dylib %t/1.o %t/2.o --start-lib %t/2.o --end-lib -o /dev/null
; RUN: llvm-dis < %t/2.o.thinlto.bc | grep -q '\^0 = module:'
; RUN: rm -f %t/2.o.thinlto.bc
; RUN: %lld --thinlto-index-only -dylib --start-lib %t/2.o --end-lib %t/2.o %t/1.o -o /dev/null
; RUN: llvm-dis < %t/2.o.thinlto.bc | grep -q '\^0 = module:'

;; Ensure when the same lazy bitcode object is given multiple times,
;; no empty index file is generated if one of the copies is linked.
; RUN: rm -f %t/2.o.thinlto.bc
; RUN: %lld --thinlto-index-only -dylib %t/1.o --start-lib %t/2.o --end-lib --start-lib %t/2.o --end-lib -o /dev/null
; RUN: llvm-dis < %t/2.o.thinlto.bc | grep -q '\^0 = module:'

; NM: T _f

;; The backend index for this module contains summaries from itself and
;; g.ll, as it imports from the latter.
; BACKEND1: <MODULE_STRTAB_BLOCK
; BACKEND1-NEXT: <ENTRY {{.*}} record string = '[[P1]]'
; BACKEND1-NEXT: <ENTRY {{.*}} record string = '[[P2]]'
; BACKEND1-NEXT: </MODULE_STRTAB_BLOCK
; BACKEND1: <GLOBALVAL_SUMMARY_BLOCK
; BACKEND1: <VERSION
; BACKEND1: <FLAGS
; BACKEND1: <VALUE_GUID op0={{1|2}} op1={{-3706093650706652785|-5300342847281564238}}
; BACKEND1: <VALUE_GUID op0={{1|2}} op1={{-3706093650706652785|-5300342847281564238}}
; BACKEND1: <COMBINED
; BACKEND1: <COMBINED
; BACKEND1: </GLOBALVAL_SUMMARY_BLOCK

;; The backend index for g.ll contains summaries from itself only,
;; as it does not import anything.
; BACKEND2: <MODULE_STRTAB_BLOCK
; BACKEND2-NEXT: <ENTRY {{.*}} record string = '[[P2]]'
; BACKEND2-NEXT: </MODULE_STRTAB_BLOCK
; BACKEND2-NEXT: <GLOBALVAL_SUMMARY_BLOCK
; BACKEND2-NEXT: <VERSION
; BACKEND2-NEXT: <FLAGS
; BACKEND2-NEXT: <VALUE_GUID op0=1 op1=-5300342847281564238
; BACKEND2-NEXT: <COMBINED
; BACKEND2-NEXT: <BLOCK_COUNT op0=2/>
; BACKEND2-NEXT: </GLOBALVAL_SUMMARY_BLOCK

;--- f.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}

;--- g.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

define void @g() {
entry:
  ret void
}

;--- empty.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"
