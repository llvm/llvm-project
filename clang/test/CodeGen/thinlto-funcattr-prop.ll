; REQUIRES: x86-registered-target

; Test that FunctionAttr Propagation is generating correct summaries

; RUN: split-file %s %t
; RUN: opt -module-summary %t/a.ll -o %t/a.bc
; RUN: opt -module-summary %t/b.ll -o %t/b.bc

; RUN: llvm-lto2 run -disable-thinlto-funcattrs=0 %t/a.bc %t/b.bc -o %t1.o -save-temps \
; RUN:   -r=%t/a.bc,call_extern,plx \
; RUN:   -r=%t/a.bc,extern, \
; RUN:   -r=%t/b.bc,extern,p

; RUN: llvm-dis %t1.o.index.bc -o - | FileCheck %s --check-prefix=CHECK-INDEX
; RUN: llvm-dis %t1.o.1.1.promote.bc -o - | FileCheck %s --check-prefix=CHECK-IR

;; Summary for call_extern. Note that llvm-lto2 writes out the index before propagation occurs so call_extern doesn't have its flags updated.
; CHECK-INDEX: ^2 = gv: (guid: 13959900437860518209, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0), insts: 2, calls: ((callee: ^3)))))
;; Summary for extern
; CHECK-INDEX: ^3 = gv: (guid: 14959766916849974397, summaries: (function: (module: ^1, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 0, canAutoHide: 0), insts: 1, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 1, returnDoesNotAlias: 0, noInline: 0, alwaysInline: 0, noUnwind: 1, mayThrow: 0, hasUnknownCall: 0))))

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @extern()

; CHECK-IR: Function Attrs: norecurse nounwind
; CHECK-IR-NEXT: define dso_local void @call_extern()
define void @call_extern() {
    call void @extern()
    ret void
}

;--- b.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

attributes #0 = { nounwind norecurse }

define void @extern() #0 {
  ret void
}
