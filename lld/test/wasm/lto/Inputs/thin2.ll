; Copied from lld/test/ELF/lto/Inputs/thin2.ll

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

define i32 @blah(i32 %meh) #0 {
entry:
  %meh.addr = alloca i32, align 4
  store i32 %meh, ptr %meh.addr, align 4
  %0 = load i32, ptr %meh.addr, align 4
  %sub = sub nsw i32 %0, 48
  ret i32 %sub
}
