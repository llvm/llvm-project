; REQUIRES: x86
;; Test --thinlto-index= for distributed ThinLTO.

; RUN: rm -rf %t && split-file %s %t && mkdir %t/thin %t/index && cd %t
; RUN: llvm-mc -filetype=obj -triple=x86_64 main.s -o main.o
; RUN: llvm-mc -filetype=obj -triple=x86_64 c.s -o c.o
; RUN: opt -thinlto-bc a.ll -o thin/a.o -thin-link-bitcode-file=thin/a.min.o
; RUN: opt -thinlto-bc bc.ll -o thin/bc.o -thin-link-bitcode-file=thin/bc.min.o
; RUN: opt -thinlto-bc %p/Inputs/thinlto_empty.ll -o thin/empty.o -thin-link-bitcode-file=thin/empty.min.o
; RUN: opt -thinlto-bc %p/Inputs/thinlto_empty.ll -o thin/empty1.o -thin-link-bitcode-file=thin/empty1.min.o

; RUN: ld.lld --thinlto-index=1.map --thinlto-emit-imports-files --thinlto-prefix-replace='thin;obj' \
; RUN:   --thinlto-object-suffix-replace='.min.o;.o' main.o thin/a.min.o \
; RUN:   --start-lib thin/bc.min.o thin/empty.min.o --end-lib thin/empty1.min.o --start-lib c.o --end-lib -o 1
; RUN: FileCheck --input-file=1.map %s --implicit-check-not={{.}} --match-full-lines --strict-whitespace

;; No entry for empty.min.o which is not extracted. empty1.min.o is present,
;; otherwise the final link may try in-process ThinLTO backend compilation.
;      CHECK:thin/a.min.o=obj/a.o
; CHECK-NEXT:thin/bc.min.o=obj/bc.o
; CHECK-NEXT:thin/empty1.min.o=obj/empty1.o
; CHECK-NEXT:thin/empty.min.o=/dev/null

;; Nevertheless, empty.o.{imports,thinlto.bc} exist to meet the build system requirement.
; RUN: ls obj/a.o.imports obj/bc.o.imports obj/empty.o.imports obj/empty1.o.imports

; RUN: llvm-bcanalyzer -dump obj/a.o.thinlto.bc | FileCheck %s --check-prefix=BACKENDA
; RUN: llvm-bcanalyzer -dump obj/bc.o.thinlto.bc | FileCheck %s --check-prefix=BACKENDB
; RUN: llvm-bcanalyzer -dump obj/empty.o.thinlto.bc | FileCheck %s --check-prefix=BACKENDE
; RUN: llvm-bcanalyzer -dump obj/empty1.o.thinlto.bc | FileCheck %s --check-prefix=BACKENDE1

; BACKENDA:       <MODULE_STRTAB_BLOCK
; BACKENDA-NEXT:  <ENTRY {{.*}} record string = 'thin/a.o'
; BACKENDA-NEXT:  <HASH
; BACKENDA-NEXT:  <ENTRY {{.*}} record string = 'thin/bc.o'
; BACKENDA-NEXT:  <HASH
; BACKENDA-NEXT:  </MODULE_STRTAB_BLOCK>

; BACKENDB:       <MODULE_STRTAB_BLOCK
; BACKENDB-NEXT:  <ENTRY {{.*}} record string = 'thin/bc.o'
; BACKENDB-NEXT:  <HASH
; BACKENDB-NEXT:  </MODULE_STRTAB_BLOCK>

; BACKENDE:       <MODULE_STRTAB_BLOCK
; BACKENDE-NEXT:  </MODULE_STRTAB_BLOCK>

; BACKENDE1:      <MODULE_STRTAB_BLOCK
; BACKENDE1-NEXT: <ENTRY {{.*}} record string = 'thin/empty1.o'
; BACKENDE1-NEXT: <HASH
; BACKENDE1-NEXT: </MODULE_STRTAB_BLOCK>

;; Thin archives can be used as well.
; RUN: llvm-ar rcTS thin.a thin/bc.min.o thin/empty.min.o
; RUN: ld.lld --thinlto-index=2.map --thinlto-prefix-replace='thin;obj' \
; RUN:   --thinlto-object-suffix-replace='.min.o;.o' main.o \
; RUN:   thin/a.min.o thin.a thin/empty1.min.o -o 2
; RUN: FileCheck --input-file=2.map %s --implicit-check-not={{.}} --match-full-lines --strict-whitespace

;; For regular archives, the filename may be insufficient to locate the archive and the particular member.
; RUN: llvm-ar rcS bc.a thin/bc.min.o thin/empty.min.o
; RUN: ld.lld --thinlto-index=3.map --thinlto-prefix-replace='thin;obj' \
; RUN:   --thinlto-object-suffix-replace='.min.o;.o' main.o \
; RUN:   thin/a.min.o bc.a thin/empty1.min.o -o 3
; RUN: FileCheck --input-file=3.map %s --check-prefix=ARCHIVE --implicit-check-not={{.}} --match-full-lines --strict-whitespace

;      ARCHIVE:thin/a.min.o=obj/a.o
; ARCHIVE-NEXT:bc.min.o=bc.o
; ARCHIVE-NEXT:thin/empty1.min.o=obj/empty1.o
; ARCHIVE-NEXT:empty.min.o=/dev/null

; RUN: ld.lld --thinlto-index=4.map --thinlto-emit-imports-files --thinlto-prefix-replace='thin;index;obj' \
; RUN:   --thinlto-object-suffix-replace='.min.o;.o' main.o thin/a.min.o \
; RUN:   --start-lib thin/bc.min.o thin/empty.min.o --end-lib thin/empty1.min.o --start-lib c.o --end-lib -o 4
; RUN: FileCheck --input-file=4.map %s --implicit-check-not={{.}} --match-full-lines --strict-whitespace
; RUN: ls index/a.o.thinlto.bc index/a.o.imports

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @b()
declare void @c()

define void @a() {
  call void () @b()
  call void () @c()
  ret void
}

;--- bc.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @b() {
  ret void
}
define void @c() {
  ret void
}

;--- main.s
.globl _start
_start:
  call a

;--- c.s
.globl c
c:
