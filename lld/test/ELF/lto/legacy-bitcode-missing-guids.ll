; REQUIRES: aarch64

;; Bitcode produced before GUID assignment was added may have a summary but no
;; GUIDs on definitions. WPD can require those GUIDs without
;; --lto-whole-program-visibility when IR vcall visibility permits
;; devirtualization.
; RUN: split-file %s %t
; RUN: opt --thinlto-bc --thinlto-split-lto-unit -o %t/thin.o %t/ThinLTO.ll
; RUN: llvm-as -o %t/regular.o %t/RegularLTO.ll
; RUN: ld.lld %t/thin.o %t/regular.o -o %t/a.out \
; RUN:   -mllvm -pass-remarks=wholeprogramdevirt 2>&1 | FileCheck %s

; CHECK: single-impl: devirtualized a call to vf.llvm.merged

;--- ThinLTO.ll
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define void @_start(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"T")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  call void %fptr(ptr %obj)
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

;--- RegularLTO.ll
source_filename = "regular.ll"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

@vt = constant ptr @vf, !type !0, !vcall_visibility !1
@llvm.used = appending global [1 x ptr] [ptr @vt], section "llvm.metadata"

define internal void @vf(ptr %obj) {
  ret void
}

!llvm.module.flags = !{!2, !3}

!0 = !{i64 0, !"T"}
!1 = !{i64 1}
!2 = !{i32 1, !"ThinLTO", i32 0}
!3 = !{i32 1, !"EnableSplitLTOUnit", i32 1}

^0 = module: (path: "regular.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (name: "vf", summaries: (function: (module: ^0, flags: (linkage: internal, visibility: default, notEligibleToImport: 1, live: 0, dsoLocal: 1, canAutoHide: 0, importType: definition, noRenameOnPromotion: 0), insts: 1))) ; guid = 4583073392578861042
^2 = gv: (name: "vt", summaries: (variable: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 1, live: 0, dsoLocal: 0, canAutoHide: 0, importType: definition, noRenameOnPromotion: 0), varFlags: (readonly: 1, writeonly: 0, constant: 1), refs: (^1)))) ; guid = 10027328872143257552
^3 = gv: (name: "llvm.used", summaries: (variable: (module: ^0, flags: (linkage: appending, visibility: default, notEligibleToImport: 1, live: 1, dsoLocal: 0, canAutoHide: 0, importType: definition, noRenameOnPromotion: 0), varFlags: (readonly: 0, writeonly: 0, constant: 0), refs: (^2)))) ; guid = 15665353970260777610
^4 = flags: 8
^5 = blockcount: 0
