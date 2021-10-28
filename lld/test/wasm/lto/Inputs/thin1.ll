; Copied from lld/test/ELF/lto/Inputs/thin1.ll

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define i32 @foo(i32 %goo) {
entry:
  %goo.addr = alloca i32, align 4
  store i32 %goo, i32* %goo.addr, align 4
  %0 = load i32, i32* %goo.addr, align 4
  %1 = load i32, i32* %goo.addr, align 4
  %mul = mul nsw i32 %0, %1
  ret i32 %mul
}
