; REQUIRES: x86
; RUN: rm -rf %t && mkdir %t && cd %t

;; First ensure that the ThinLTO handling in lld handles
;; bitcode without summary sections gracefully and generates index file.
; RUN: llvm-as %s -o 1.o
; RUN: llvm-as %p/Inputs/thinlto.ll -o 2.o
; RUN: ld.lld --plugin-opt=thinlto-index-only -shared 1.o 2.o -o 3
; RUN: ls 2.o.thinlto.bc
; RUN: not test -e 3
; RUN: ld.lld -shared 1.o 2.o -o 3
; RUN: llvm-nm 3 | FileCheck %s --check-prefix=NM

;; Basic ThinLTO tests.
; RUN: opt -module-summary %s -o 1.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o 2.o
; RUN: opt -module-summary %p/Inputs/thinlto_empty.ll -o 3.o

;; Ensure lld generates an index and not a binary if requested.
; RUN: ld.lld --plugin-opt=thinlto-index-only -shared 1.o 2.o -o 4
; RUN: llvm-bcanalyzer -dump 1.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND1
; RUN: llvm-bcanalyzer -dump 2.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND2
; RUN: not test -e 4

;; Ensure lld generates an index even if the file is wrapped in --start-lib/--end-lib
; RUN: rm -f 2.o.thinlto.bc 4
; RUN: ld.lld --plugin-opt=thinlto-index-only -shared 1.o 3.o --start-lib 2.o --end-lib -o 4
; RUN: llvm-dis < 2.o.thinlto.bc | grep -q '\^0 = module:'
; RUN: not test -e 4

;; Test that LLD generates an empty index even for lazy object file that is not added to link.
;; Test LLD generates empty imports file either because of thinlto-emit-imports-files option.
; RUN: rm -f 1.o.thinlto.bc 1.o.imports
; RUN: ld.lld --plugin-opt=thinlto-index-only -shared 2.o --start-lib 1.o --end-lib \
; RUN: --plugin-opt=thinlto-emit-imports-files -o 3
; RUN: ls 1.o.thinlto.bc
; RUN: ls 1.o.imports

;; Ensure LLD generates an empty index for each bitcode file even if all bitcode files are lazy.
; RUN: rm -f 1.o.thinlto.bc
; RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux-gnu /dev/null -o dummy.o
; RUN: ld.lld --plugin-opt=thinlto-index-only -shared dummy.o --start-lib 1.o --end-lib -o /dev/null
; RUN: ls 1.o.thinlto.bc

;; Ensure when the same bitcode object is given as both lazy and non-lazy,
;; LLD does not generate an empty index for the lazy object.
; RUN: rm -f 2.o.thinlto.bc
; RUN: ld.lld --plugin-opt=thinlto-index-only -shared 1.o 2.o --start-lib 2.o --end-lib -o /dev/null
; RUN: llvm-dis < 2.o.thinlto.bc | grep -q '\^0 = module:'
; RUN: rm -f 2.o.thinlto.bc
; RUN: ld.lld --plugin-opt=thinlto-index-only -shared --start-lib 2.o --end-lib 2.o 1.o -o /dev/null
; RUN: llvm-dis < 2.o.thinlto.bc | grep -q '\^0 = module:'

;; Ensure when the same lazy bitcode object is given multiple times,
;; no empty index file is generated if one of the copies is linked.
; RUN: rm -f 2.o.thinlto.bc
; RUN: ld.lld --plugin-opt=thinlto-index-only -shared 1.o --start-lib 2.o --end-lib --start-lib 2.o --end-lib -o /dev/null
; RUN: llvm-dis < 2.o.thinlto.bc | grep -q '\^0 = module:'

; NM: T f

;; The backend index for this module contains summaries from itself and
;; Inputs/thinlto.ll, as it imports from the latter.
; BACKEND1: <MODULE_STRTAB_BLOCK
; BACKEND1-NEXT: <ENTRY {{.*}} record string = '1.o'
; BACKEND1-NEXT: <ENTRY {{.*}} record string = '2.o'
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
; BACKEND2-NEXT: <ENTRY {{.*}} record string = '2.o'
; BACKEND2-NEXT: </MODULE_STRTAB_BLOCK
; BACKEND2-NEXT: <GLOBALVAL_SUMMARY_BLOCK
; BACKEND2-NEXT: <VERSION
; BACKEND2-NEXT: <FLAGS
; BACKEND2-NEXT: <VALUE_GUID op0=1 op1=-5300342847281564238
; BACKEND2-NEXT: <COMBINED
; BACKEND2-NEXT: <BLOCK_COUNT op0=2/>
; BACKEND2-NEXT: </GLOBALVAL_SUMMARY_BLOCK

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
