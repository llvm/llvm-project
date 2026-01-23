target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @bar_func1(i32 %a, i32 %b) {
entry:
  %mul = shl nsw i32 %a, 1
  %res = add nsw i32 %mul, %b
  ret i32 %res
}

define spir_func i32 @baz_func1(i32 %a) {
entry:
  %add = add nsw i32 %a, 5
  %res = tail call spir_func i32 @bar_func1(i32 %a, i32 %add)
  ret i32 %res
}
