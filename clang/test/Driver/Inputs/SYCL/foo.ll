target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @foo_func1(i32 %a, i32 %b) {
entry:
  %call = tail call spir_func i32 @addFive(i32 %b)
  %res = tail call spir_func i32 @bar_func1(i32 %a, i32 %call)
  ret i32 %res
}

declare spir_func i32 @bar_func1(i32, i32)

declare spir_func i32 @addFive(i32)

define spir_func i32 @foo_func2(i32 %c, i32 %d, i32 %e) {
entry:
  %call = tail call spir_func i32 @foo_func1(i32 %c, i32 %d)
  %res = mul nsw i32 %call, %e
  ret i32 %res
}
