; RUN: aiir-translate --import-llvm %s | FileCheck %s

%Domain = type { ptr, ptr }

; CHECK: llvm.aiir.global external @D()
; CHECK-SAME: !llvm.struct<"Domain", (ptr, ptr)>
; CHECK: %[[E0:.+]] = llvm.aiir.zero : !llvm.ptr
; CHECK: %[[RES:.+]] = llvm.aiir.zero : !llvm.struct<"Domain", (ptr, ptr)>
; CHECK: llvm.return %[[RES]] : !llvm.struct<"Domain", (ptr, ptr)>
@D = global %Domain zeroinitializer
