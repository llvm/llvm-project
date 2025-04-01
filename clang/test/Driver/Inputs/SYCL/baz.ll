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
