; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t

;; Mostly copied/updated from thinlto-index-only.ll
;; First ensure that the ThinLTO handling in lld handles
;; bitcode without summary sections gracefully and generates index file.
; RUN: llvm-as %t/f.ll -o %t/1.o
; RUN: llvm-as %t/g.ll -o %t/2.o
; RUN: %lld --thinlto-emit-index-files -dylib %t/1.o %t/2.o -o %t/3
; RUN: ls %t/2.o.thinlto.bc
; RUN: ls %t/3
; RUN: %lld -dylib %t/1.o %t/2.o -o %t/3
; RUN: llvm-nm %t/3 | FileCheck %s --check-prefix=NM

;; Basic ThinLTO tests.
; RUN: opt -module-summary %t/f.ll -o %t/1.o
; RUN: opt -module-summary %t/g.ll -o %t/2.o
; RUN: opt -module-summary %t/empty.ll -o %t/3.o

;; Ensure lld generates an index and also a binary if requested.
; RUN: %lld --thinlto-emit-index-files -dylib %t/1.o %t/2.o -o %t/4
; RUN: llvm-bcanalyzer -dump %t/1.o.thinlto.bc | FileCheck %s -DP1=%t/1.o -DP2=%t/2.o --check-prefix=BACKEND1
; RUN: llvm-bcanalyzer -dump %t/2.o.thinlto.bc | FileCheck %s -DP2=%t/2.o --check-prefix=BACKEND2
; RUN: ls %t/4

;; Ensure lld generates an index and not a binary if both emit-index and index-only are present.
; RUN: %lld --thinlto-emit-index-files --thinlto-index-only -dylib %t/1.o %t/2.o -o %t/5
; RUN: not ls %t/5

;; Ensure lld generates an index even if the file is wrapped in --start-lib/--end-lib
; RUN: rm -f %t/2.o.thinlto.bc
; RUN: %lld --thinlto-emit-index-files -dylib %t/1.o %t/3.o --start-lib %t/2.o --end-lib -o %t/6
; RUN: llvm-dis < %t/2.o.thinlto.bc | grep -q '\^0 = module:'
; RUN: ls %t/6

;; Test that LLD generates an empty index even for lazy object file that is not added to link.
;; Test that LLD also generates empty imports file with the --thinlto-emit-imports-files option.
; RUN: rm -f %t/1.o.thinlto.bc %t/1.o.imports
; RUN: %lld --thinlto-emit-index-files -dylib %t/2.o --start-lib %t/1.o --end-lib \
; RUN: --thinlto-emit-imports-files -o %t/7
; RUN: ls %t/7
; RUN: ls %t/1.o.thinlto.bc
; RUN: ls %t/1.o.imports

;; Ensure LLD generates an empty index for each bitcode file even if all bitcode files are lazy.
; RUN: rm -f %t/1.o.thinlto.bc
; RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin /dev/null -o dummy.o
; RUN: %lld --thinlto-emit-index-files -dylib dummy.o --start-lib %t/1.o --end-lib -o %t/8
; RUN: ls %t/8
; RUN: ls %t/1.o.thinlto.bc

;; Test that LLD errors out when run with suffix replacement, or prefix replacement
; RUN: not %lld --thinlto-emit-index-files -dylib %t/2.o --start-lib %t/1.o --end-lib \
; RUN: --thinlto-prefix-replace="abc;xyz" 2>&1 | FileCheck %s --check-prefix=ERR1
; ERR1: --thinlto-prefix-replace is not supported with --thinlto-emit-index-files

; RUN: not %lld --thinlto-emit-index-files -dylib %t/2.o --start-lib %t/1.o --end-lib \
; RUN: --thinlto-object-suffix-replace="abc;xyz" 2>&1 | FileCheck %s --check-prefix=ERR2
; ERR2: --thinlto-object-suffix-replace is not supported with --thinlto-emit-index-files

;; But not when passed with index only as well
; RUN: %lld --thinlto-emit-index-files -dylib %t/2.o --start-lib %t/1.o --end-lib \
; RUN: --thinlto-prefix-replace="abc;xyz" --thinlto-index-only

; RUN: %lld --thinlto-emit-index-files -dylib %t/2.o --start-lib %t/1.o --end-lib \
; RUN: --thinlto-object-suffix-replace="abc;xyz" --thinlto-index-only

; NM: T _f

;; The backend index for this module contains summaries from itself and
;; Inputs/thinlto.ll, as it imports from the latter.
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

;; The backend index for Input/thinlto.ll contains summaries from itself only,
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
