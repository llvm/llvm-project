; REQUIRES: x86, non-root-user
;; Test a few properties not tested by thinlto-index-only.ll

; RUN: opt -module-summary %s -o %t1.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o %t2.o
; RUN: opt -module-summary %p/Inputs/thinlto_empty.ll -o %t3.o

; Ensure lld generates error if unable to write to imports file.
; RUN: rm -f %t3.o.imports
; RUN: touch %t3.o.imports
; RUN: chmod 400 %t3.o.imports
; RUN: not ld.lld --plugin-opt=thinlto-index-only --plugin-opt=thinlto-emit-imports-files -shared %t1.o %t2.o %t3.o -o /dev/null 2>&1 | FileCheck -DMSG=%errc_EACCES %s --check-prefix=ERR
; ERR: cannot open {{.*}}3.o.imports: [[MSG]]

; RUN: rm -f %t1.o.imports %t2.o.imports rm -f %t3.o.imports
; RUN: ld.lld --plugin-opt=thinlto-emit-imports-files -shared %t1.o %t2.o %t3.o -o %t4
; RUN: not ls %t1.o.imports
; RUN: not ls %t2.o.imports
; RUN: not ls %t3.o.imports

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
