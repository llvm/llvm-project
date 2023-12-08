target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-emscripten"

declare void @llvm.wasm.throw(i32, ptr)

define void @foo(ptr %p) {
  call void @llvm.wasm.throw(i32 0, ptr %p)
  ret void
}
