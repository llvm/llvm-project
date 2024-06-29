;; This test is similar to llvm/test/ThinLTO/X86/selective-save-temps.ll

; REQUIRES: x86
; UNSUPPORTED: system-windows
;; Unsupported on Windows due to difficulty with escaping "opt" across platforms.
;; lit substitutes 'opt' with /path/to/opt.

; RUN: rm -rf %t && mkdir %t && cd %t
; RUN: mkdir all all2 all3 build subset subset2 && cd build

; RUN: opt -thinlto-bc -o main.o %s
; RUN: opt -thinlto-bc -o thin1.o %S/Inputs/thinlto.ll

;; Create the .all dir with save-temps saving everything, this will be used to compare
;; with the output from individualized save-temps later
; RUN: ld.lld main.o thin1.o --save-temps -o %t/all/a.out
; RUN: mv a.out.lto.* *.o.*.bc %t/all
;; Sanity check that everything got moved
; RUN: ls | count 2

;; Check precedence if both --save-temps and --save-temps= are present
; RUN: ld.lld main.o thin1.o --save-temps=preopt --save-temps --save-temps=\opt -o %t/all2/a.out
; RUN: cmp %t/all2/a.out %t/all/a.out
; RUN: mv a.out.lto.* *.o.* %t/all2
; RUN: ls | count 2
; RUN: diff -r %t/all %t/all2

;; The next 9 blocks follow this structure:
;; for each option of save-temps=
;;   Run linker and generate files
;;   Make sure a.out exists and is correct (by diff-ing)
;;     this is the only file that should recur between runs
;;   (Also, for some stages, copy the generated files to %t/subset2 to check composability later)
;;   Move files that were expected to be generated to %t/all3
;;   Make sure there's no unexpected extra files
;; After that, we'll diff %t/all and %t/all3 to make sure all contents are identical

;; Check preopt
; RUN: ld.lld main.o thin1.o --save-temps=preopt
; RUN: cmp %t/all/a.out a.out && rm -f a.out
; RUN: cp *.0.preopt.* %t/subset2
; RUN: mv *.0.preopt.* %t/all3
; RUN: ls | count 2

;; Check promote
; RUN: ld.lld main.o thin1.o --save-temps=promote
; RUN: cmp %t/all/a.out a.out && rm -f a.out
; RUN: mv *.1.promote* %t/all3
; RUN: ls | count 2

;; Check internalize
; RUN: ld.lld main.o thin1.o --save-temps=internalize
; RUN: cmp %t/all/a.out a.out && rm -f a.out
; RUN: mv *.2.internalize* %t/all3
; RUN: ls | count 2

;; Check import
; RUN: ld.lld main.o thin1.o --save-temps=import
; RUN: cmp %t/all/a.out a.out && rm -f a.out
; RUN: mv *.3.import* %t/all3
; RUN: ls | count 2

;; Check opt
; RUN: ld.lld main.o thin1.o --save-temps=\opt
; RUN: cmp %t/all/a.out a.out && rm -f a.out
; RUN: cp *.4.opt* %t/subset2
; RUN: mv *.4.opt* %t/all3
; RUN: ls | count 2

;; Check precodegen
; RUN: ld.lld main.o thin1.o --save-temps=precodegen
; RUN: cmp %t/all/a.out a.out && rm -f a.out
; RUN: mv *.5.precodegen* %t/all3
; RUN: ls | count 2

;; Check combinedindex
; RUN: ld.lld main.o thin1.o --save-temps=combinedindex
; RUN: cmp %t/all/a.out a.out && rm -f a.out
; RUN: mv *.index.bc %t/all3
; RUN: mv *.index.dot %t/all3
; RUN: ls | count 2

;; Check prelink
; RUN: ld.lld main.o thin1.o --save-temps=prelink
; RUN: cmp %t/all/a.out a.out && rm -f a.out
; RUN: cp a.out.lto.*.o %t/subset2
; RUN: mv a.out.lto.*.o %t/all3
; RUN: ls | count 2

;; Check resolution
; RUN: ld.lld main.o thin1.o --save-temps=resolution
;; %t/all3 needs at least 1 copy of a.out, move it over now since its the last block
; RUN: mv a.out %t/all3
; RUN: mv *.resolution.txt %t/all3
; RUN: ls | count 2

;; If no files were left out from individual stages, the .all3 dir should be identical to .all
; RUN: diff -r %t/all %t/all3

;; Check multi-stage composability
;; Similar to the above, but do it with a subset instead.
;; .all -> .subset, .all3 -> .subset2
; RUN: ld.lld main.o thin1.o --save-temps=preopt --save-temps=prelink --save-temps=\opt
; RUN: cmp %t/all/a.out a.out && rm -f a.out
; RUN: mv *.0.preopt.* %t/subset
; RUN: mv *.4.opt* %t/subset
; RUN: mv a.out.lto.*.o %t/subset
; RUN: ls | count 2
; RUN: diff -r %t/subset2 %t/subset

;; Check error message
; RUN: not ld.lld --save-temps=prelink --save-temps=\opt --save-temps=notastage 2>&1 \
; RUN: | FileCheck %s
; CHECK: unknown --save-temps value: notastage

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g()

define i32 @_start() {
  call void @g()
  ret i32 0
}
