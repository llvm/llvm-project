target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @addFive(i32 %a) {
entry:
  %res = add nsw i32 %a, 5
  ret i32 %res
}

define spir_func i32 @unusedFunc(i32 %a) {
entry:
  %res = mul nsw i32 %a, 5
  ret i32 %res
}
