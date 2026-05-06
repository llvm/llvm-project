target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

; A kernel from TU1 and a sycl_external function from TU2.

define spir_func i32 @ext_fn(i32 %a) #1 {
entry:
  %r = add nsw i32 %a, 2
  ret i32 %r
}

define spir_kernel void @k(ptr addrspace(1) %out) #0 {
entry:
  store i32 42, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { "sycl-module-id"="TU1.cpp" }
attributes #1 = { "sycl-module-id"="TU2.cpp" }
