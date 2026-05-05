target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @helper_shared(i32 %a) {
entry:
  %r = add nsw i32 %a, 1
  ret i32 %r
}

define spir_kernel void @kernel_a(ptr addrspace(1) %out, i32 %a) {
entry:
  %r = tail call spir_func i32 @helper_shared(i32 %a)
  store i32 %r, ptr addrspace(1) %out, align 4
  ret void
}

define spir_kernel void @kernel_b(ptr addrspace(1) %out, i32 %a, i32 %b) {
entry:
  %h = tail call spir_func i32 @helper_shared(i32 %a)
  %r = mul nsw i32 %h, %b
  store i32 %r, ptr addrspace(1) %out, align 4
  ret void
}
