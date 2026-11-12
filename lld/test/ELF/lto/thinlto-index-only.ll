; REQUIRES: x86
; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: mkdir d

;; First ensure that the ThinLTO handling in lld handles
;; bitcode without summary sections gracefully and generates index file.
; RUN: llvm-as 1.ll -o 1.o
; RUN: llvm-as %p/Inputs/thinlto.ll -o d/2.o
; RUN: ld.lld --plugin-opt=thinlto-index-only -shared 1.o d/2.o -o 3
; RUN: ls d/2.o.thinlto.bc
; RUN: not test -e 3
; RUN: ld.lld -shared 1.o d/2.o -o 3
; RUN: llvm-nm 3 | FileCheck %s --check-prefix=NM

;; Basic ThinLTO tests.
; RUN: llvm-as 0.ll -o 0.o
; RUN: opt -module-summary 1.ll -o 1.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o d/2.o
; RUN: opt -module-summary %p/Inputs/thinlto_empty.ll -o 3.o
; RUN: cp 3.o 4.o

;; Ensure lld doesn't generates index files when --thinlto-index-only is not enabled.
; RUN: rm -f 1.o.thinlto.bc d/2.o.thinlto.bc
; RUN: ld.lld -shared 1.o d/2.o -o /dev/null
; RUN: not ls 1.o.thinlto.bc
; RUN: not ls d/2.o.thinlto.bc

;; Ensure lld generates an index and not a binary if requested.
; RUN: ld.lld --plugin-opt=thinlto-index-only -shared 1.o --start-lib d/2.o 3.o --end-lib 4.o -o 4
; RUN: not test -e 4
; RUN: llvm-bcanalyzer -dump 1.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND1
; RUN: llvm-bcanalyzer -dump d/2.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND2
; RUN: llvm-dis < 3.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND3
; RUN: llvm-dis < 4.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND4

; RUN: rm -f 1.o.thinlto.bc d/2.o.thinlto.bc 3.o.thinlto.bc 4.o.thinlto.bc
; RUN: ld.lld --plugin-opt=thinlto-index-only=4.txt --plugin-opt=thinlto-emit-imports-files -shared 1.o --start-lib d/2.o 3.o --end-lib 4.o -o 4
; RUN: not test -e 4
; RUN: FileCheck %s --check-prefix=RSP --implicit-check-not={{.}} < 4.txt
; RUN: llvm-bcanalyzer -dump 1.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND1
; RUN: llvm-bcanalyzer -dump d/2.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND2
; RUN: llvm-dis < 3.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND3
; RUN: llvm-dis < 4.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND4
; RUN: FileCheck %s --check-prefix=IMPORTS1 --implicit-check-not={{.}} < 1.o.imports
; RUN: count 0 < d/2.o.imports
;; Test that LLD generates an empty index even for lazy object file that is not added to link.
; RUN: count 0 < 3.o.imports
; RUN: count 0 < 4.o.imports

;; Test the other spelling --thinlto-index-only= --thinlto-emit-imports-files and the interaction with --save-temps.
; RUN: rm -f 4.txt 1.o.thinlto.bc d/2.o.thinlto.bc 3.o.thinlto.bc 4.o.thinlto.bc
; RUN: ld.lld --thinlto-index-only=4.txt --thinlto-emit-imports-files --save-temps -shared 0.o 1.o --start-lib d/2.o 3.o --end-lib 4.o -o t
; RUN: not test -e 4
; RUN: FileCheck %s --check-prefix=RSP --implicit-check-not={{.}} < 4.txt
; RUN: llvm-bcanalyzer -dump 1.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND1
; RUN: FileCheck %s --check-prefix=IMPORTS1 --implicit-check-not={{.}} < 1.o.imports
; RUN: FileCheck %s --check-prefix=RESOLUTION < t.resolution.txt
; RUN: llvm-dis < t.index.bc | FileCheck %s --check-prefix=INDEX-BC

; RSP:      1.o
; RSP-NEXT: d/2.o
; RSP-NEXT: 4.o

; IMPORTS1: d/2.o

; RESOLUTION:      0.o
; RESOLUTION-NEXT: -r=0.o,foo,px
; RESOLUTION-NEXT: 1.o

; INDEX-BC:      ^0 = module: (path: "1.o", hash: (0, 0, 0, 0, 0))
; INDEX-BC-NEXT: ^1 = module: (path: "4.o", hash: (0, 0, 0, 0, 0))
; INDEX-BC-NEXT: ^2 = module: (path: "d/2.o", hash: (0, 0, 0, 0, 0))

;; Ensure LLD generates an empty index for each bitcode file even if all bitcode files are lazy.
; RUN: rm -f 1.o.thinlto.bc
; RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux-gnu /dev/null -o dummy.o
; RUN: ld.lld --plugin-opt=thinlto-index-only -shared dummy.o --start-lib 1.o --end-lib -o /dev/null
; RUN: ls 1.o.thinlto.bc

;; Ensure when the same bitcode object is given as both lazy and non-lazy,
;; LLD does not generate an empty index for the lazy object.
; RUN: rm -f d/2.o.thinlto.bc
; RUN: ld.lld --plugin-opt=thinlto-index-only -shared 1.o d/2.o --start-lib d/2.o --end-lib -o /dev/null
; RUN: llvm-dis < d/2.o.thinlto.bc | grep -q '\^0 = module:'
; RUN: rm -f d/2.o.thinlto.bc
; RUN: ld.lld --plugin-opt=thinlto-index-only -shared --start-lib d/2.o --end-lib d/2.o 1.o -o /dev/null
; RUN: llvm-dis < d/2.o.thinlto.bc | grep -q '\^0 = module:'

;; Ensure when the same lazy bitcode object is given multiple times,
;; no empty index file is generated if one of the copies is linked.
; RUN: rm -f d/2.o.thinlto.bc
; RUN: ld.lld --plugin-opt=thinlto-index-only -shared 1.o --start-lib d/2.o --end-lib --start-lib d/2.o --end-lib -o /dev/null
; RUN: llvm-dis < d/2.o.thinlto.bc | grep -q '\^0 = module:'

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
; BACKEND1: <VALUE_GUID {{.*}} op0={{1|2}} {{op1=3060885059 op2=1207956914|op1=3432075125 op2=3712786831}}
; BACKEND1: <VALUE_GUID {{.*}} op0={{1|2}} {{op1=3060885059 op2=1207956914|op1=3432075125 op2=3712786831}}
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
; BACKEND2-NEXT: <VALUE_GUID {{.*}} op0=1 op1=3060885059 op2=1207956914
; BACKEND2-NEXT: <COMBINED
; BACKEND2-NEXT: </GLOBALVAL_SUMMARY_BLOCK

; BACKEND3: ^0 = flags:

; BACKEND4: ^0 = module: (path: "4.o", hash: (0, 0, 0, 0, 0))

;--- 0.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() {
  ret void
}

;--- 1.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
