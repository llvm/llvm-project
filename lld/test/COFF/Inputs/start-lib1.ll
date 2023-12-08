target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare i32 @bar()

define i32 @foo() {
  %1 = call i32 () @bar()
  %2 = add i32 %1, 1
  ret i32 %2
}

!llvm.linker.options = !{!0}
!0 = !{!"/INCLUDE:foo"}
