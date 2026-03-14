; REQUIRES: x86
; RUN: rm -fr %t && mkdir %t && cd %t
; RUN: opt -thinlto-bc -o main.o %s
; RUN: opt -thinlto-bc -o thin1.o %S/Inputs/thin1.ll
; RUN: opt -thinlto-bc -o thin2.o %S/Inputs/thin2.ll
; RUN: llvm-ar qcT thin.a thin1.o thin2.o

;; --thinlto-single-module=main.o should result in only main.o compiled, of which
;; the object code is saved in single1.o1. Note that single1.o is always the dummy
;; output, aka ld-temp.o. There should be no more object files generated.
; RUN: ld.lld main.o thin.a --thinlto-single-module=main.o --lto-obj-path=single1.o
; RUN: llvm-readelf -S -s single1.o | FileCheck %s --check-prefix=DEFAULT
; RUN: llvm-readelf -S -s single1.o1 | FileCheck %s --check-prefix=MAIN
; RUN: not ls single1.o2
; RUN: not ls a.out

; DEFAULT:       Value        Size Type Bind   Vis     Ndx Name
; DEFAULT:   0000000000000000    0 FILE LOCAL  DEFAULT ABS ld-temp.o
; MAIN:          Value        Size Type Bind   Vis     Ndx Name
; MAIN:      0000000000000000    0 FILE LOCAL  DEFAULT ABS thinlto-single-module.ll
; MAIN-NEXT: 0000000000000000    3 FUNC GLOBAL DEFAULT   3 _start

;; --thinlto-single-module=thin.a should result in only thin1.o and thin2.o compiled.
; RUN: ld.lld main.o thin.a --thinlto-single-module=thin.a --lto-obj-path=single2.o
; RUN: llvm-readelf -S -s single2.o | FileCheck %s --check-prefix=DEFAULT
; RUN: llvm-readelf -S -s single2.o1 | FileCheck %s --check-prefix=FOO
; RUN: llvm-readelf -S -s single2.o2 | FileCheck %s --check-prefix=BLAH
; RUN: not ls single1.o3

;; Multiple --thinlto-single-module uses should result in a combination of inputs compiled.
; RUN: ld.lld main.o thin.a --thinlto-single-module=main.o --thinlto-single-module=thin2.o --lto-obj-path=single4.o
; RUN: llvm-readelf -S -s single4.o | FileCheck %s --check-prefix=DEFAULT
; RUN: llvm-readelf -S -s single4.o1 | FileCheck %s --check-prefix=MAIN
; RUN: llvm-readelf -S -s single4.o2 | FileCheck %s --check-prefix=BLAH
; RUN: not ls single4.o3

; FOO:           Value        Size Type Bind   Vis     Ndx Name
; FOO:       0000000000000000    0 FILE LOCAL  DEFAULT ABS thin1.ll
; FOO-NEXT:  0000000000000000    6 FUNC GLOBAL DEFAULT   3 foo
; BLAH:          Value        Size Type Bind   Vis     Ndx Name
; BLAH:      0000000000000000    0 FILE LOCAL  DEFAULT ABS thin2.ll
; BLAH-NEXT: 0000000000000000    4 FUNC GLOBAL DEFAULT   3 blah

;; Check only main.o is in the result thin index file.
;; Also check a *.thinlto.bc file generated for main.o only.
; RUN: ld.lld main.o thin.a --thinlto-single-module=main.o --thinlto-index-only=single5.idx
; RUN: ls main.o.thinlto.bc
; RUN: ls | FileCheck --implicit-check-not='thin.{{.*}}.thinlto.bc' /dev/null
; RUN: FileCheck %s --check-prefix=IDX < single5.idx
; RUN: count 1 < single5.idx

; IDX: main.o

;; Check temporary output generated for main.o only.
; RUN: ld.lld main.o thin.a --thinlto-single-module=main.o --save-temps
; RUN: ls main.o.0.preopt.bc
; RUN: not ls thin.*.0.preopt.bc

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-scei-ps4"

declare i32 @blah(i32 %meh)
declare i32 @foo(i32 %goo)

define i32 @_start() {
  call i32 @foo(i32 0)
  call i32 @blah(i32 0)
  ret i32 0
}
