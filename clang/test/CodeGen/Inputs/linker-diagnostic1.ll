target triple = "armv4-none-unknown-eabi"
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

declare i32 @foo(i32)

define i32 @bar(i32 %x) {
  %1 = tail call i32 @foo(i32 %x)
  ret i32 %1
}
