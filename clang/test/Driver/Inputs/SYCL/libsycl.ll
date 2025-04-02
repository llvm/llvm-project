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
