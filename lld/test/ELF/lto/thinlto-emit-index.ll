; REQUIRES: x86

;; Mostly copied/updated from thinlto-index-only.ll
;; First ensure that the ThinLTO handling in lld handles
;; bitcode without summary sections gracefully and generates index file.
; RUN: rm -rf %t && mkdir %t && cd %t
; RUN: mkdir d
; RUN: llvm-as %s -o 1.o
; RUN: llvm-as %p/Inputs/thinlto.ll -o d/2.o
; RUN: ld.lld --thinlto-emit-index-files -shared 1.o d/2.o -o 3
; RUN: ls d/2.o.thinlto.bc
; RUN: ls 3
; RUN: ld.lld -shared 1.o d/2.o -o 3
; RUN: llvm-nm 3 | FileCheck %s --check-prefix=NM

;; Basic ThinLTO tests.
; RUN: opt -module-summary %s -o 1.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o d/2.o
; RUN: opt -module-summary %p/Inputs/thinlto_empty.ll -o 3.o
; RUN: cp 3.o 4.o

;; Ensure lld generates an index and also a binary if requested.
; RUN: ld.lld --thinlto-emit-index-files -shared 1.o --start-lib d/2.o 3.o --end-lib 4.o -o 4
; RUN: ls 4
; RUN: llvm-bcanalyzer -dump 1.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND1
; RUN: llvm-bcanalyzer -dump d/2.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND2
; RUN: llvm-dis < 3.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND3
; RUN: llvm-dis < 4.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND4

; IMPORTS1: d/2.o

;; Ensure lld generates an index and not a binary if both emit-index and index-only are present.
; RUN: ld.lld --thinlto-emit-index-files --thinlto-index-only -shared 1.o d/2.o -o 5
; RUN: not ls 5

;; Test that LLD generates an empty index even for lazy object file that is not added to link.
;; Test that LLD also generates empty imports file with the --thinlto-emit-imports-files option.
; RUN: rm -f 1.o.thinlto.bc 1.o.imports
; RUN: ld.lld --thinlto-emit-index-files -shared d/2.o --start-lib 1.o --end-lib \
; RUN: --thinlto-emit-imports-files -o 7
; RUN: ls 7
; RUN: ls 1.o.thinlto.bc
; RUN: ls 1.o.imports

;; Ensure LLD generates an empty index for each bitcode file even if all bitcode files are lazy.
; RUN: rm -f 1.o.thinlto.bc
; RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux-gnu /dev/null -o dummy.o
; RUN: ld.lld --thinlto-emit-index-files -shared dummy.o --start-lib 1.o --end-lib -o 8
; RUN: ls 8
; RUN: ls 1.o.thinlto.bc

;; Test that LLD errors out when run with suffix replacement, or prefix replacement
; RUN: not ld.lld --thinlto-emit-index-files -shared d/2.o --start-lib 1.o --end-lib \
; RUN: --thinlto-prefix-replace="abc;xyz" 2>&1 | FileCheck %s --check-prefix=ERR1
; ERR1: --thinlto-prefix-replace is not supported with --thinlto-emit-index-files

; RUN: not ld.lld --thinlto-emit-index-files -shared d/2.o --start-lib 1.o --end-lib \
; RUN: --thinlto-object-suffix-replace="abc;xyz" 2>&1 | FileCheck %s --check-prefix=ERR2
; ERR2: --thinlto-object-suffix-replace is not supported with --thinlto-emit-index-files

;; But not when passed with index only as well
; RUN: ld.lld --thinlto-emit-index-files -shared d/2.o --start-lib 1.o --end-lib \
; RUN: --thinlto-prefix-replace="abc;xyz" --thinlto-index-only

; RUN: ld.lld --thinlto-emit-index-files -shared d/2.o --start-lib 1.o --end-lib \
; RUN: --thinlto-object-suffix-replace="abc;xyz" --thinlto-index-only

; NM: T f

;; The backend index for this module contains summaries from itself and
;; Inputs/thinlto.ll, as it imports from the latter.
; BACKEND1: <MODULE_STRTAB_BLOCK
; BACKEND1-NEXT: <ENTRY {{.*}} record string = '1.o'
; BACKEND1-NEXT: <ENTRY {{.*}} record string = 'd/2.o'
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
; BACKEND2-NEXT: <ENTRY {{.*}} record string = 'd/2.o'
; BACKEND2-NEXT: </MODULE_STRTAB_BLOCK
; BACKEND2-NEXT: <GLOBALVAL_SUMMARY_BLOCK
; BACKEND2-NEXT: <VERSION
; BACKEND2-NEXT: <FLAGS
; BACKEND2-NEXT: <VALUE_GUID op0=1 op1=-5300342847281564238
; BACKEND2-NEXT: <COMBINED
; BACKEND2-NEXT: </GLOBALVAL_SUMMARY_BLOCK

; BACKEND3: ^0 = flags:

; BACKEND4: ^0 = module: (path: "4.o", hash: (0, 0, 0, 0, 0))

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
