target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

; This function, when compiled will generate a new undefined reference to
; memcpy, that is not present in the bitcode object
define void @func_with_libcall(ptr %a, ptr %b) {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %a, ptr %b, i64 1024, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1)
