; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t

; Generate summary sections and test lld handling.
; RUN: opt -module-summary %t/f.ll -o %t1.o
; RUN: opt -module-summary %t/g.ll -o %t2.o

; Include a file with an empty module summary index, to ensure that the expected
; output files are created regardless, for a distributed build system.
; RUN: opt -module-summary %t/empty.ll -o %t3.o

; Ensure lld generates imports files if requested for distributed backends.
; RUN: rm -f %t3.o.imports %t3.o.thinlto.bc
; RUN: %lld --thinlto-index-only --thinlto-emit-imports-files -dylib %t1.o %t2.o %t3.o -o %t4

; The imports file for this module contains the bitcode file for %t/g.ll
; RUN: count 1 < %t1.o.imports
; RUN: FileCheck %s --check-prefix=IMPORTS1 < %t1.o.imports
; IMPORTS1: thinlto-emit-imports.ll.tmp2.o

; The imports file for g.ll is empty as it does not import anything.
; RUN: count 0 < %t2.o.imports

; The imports file for empty.ll is empty but should exist.
; RUN: count 0 < %t3.o.imports

; The index file should be created even for the input with an empty summary.
; RUN: ls %t3.o.thinlto.bc

; Ensure lld generates error if unable to write to imports file.
; RUN: rm -f %t3.o.imports
; RUN: touch %t3.o.imports
; RUN: chmod 400 %t3.o.imports
; RUN: not %lld --thinlto-index-only --thinlto-emit-imports-files -dylib %t1.o %t2.o %t3.o -o /dev/null 2>&1 \
; RUN:     | FileCheck -DMSG=%errc_EACCES %s --check-prefix=ERR
; ERR: cannot open {{.*}}3.o.imports: [[MSG]]

; Ensure lld doesn't generate import files when thinlto-index-only is not enabled
; RUN: rm -f %t1.o.imports
; RUN: rm -f %t2.o.imports
; RUN: rm -f %t3.o.imports
; RUN: %lld --thinlto-emit-imports-files -dylib %t1.o %t2.o %t3.o -o %t4
; RUN: not ls %t1.o.imports
; RUN: not ls %t2.o.imports
; RUN: not ls %t3.o.imports

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
