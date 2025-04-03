target triple = "spirv64"

define spir_func i32 @bar_func1(i32 %a, i32 %b) {
entry:
  %res = add nsw i32 %b, %a
  ret i32 %res
}
