target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @helper() {
  ret i32 0
}

define spir_kernel void @kernel_a() #0 {
  %r = call spir_func i32 @helper()
  ret void
}

define spir_kernel void @kernel_b() #1 {
  %r = call spir_func i32 @helper()
  ret void
}

define spir_func i32 @ext_fn() #2 {
  %r = call spir_func i32 @helper()
  ret i32 0
}

attributes #0 = { "sycl-module-id"="TU1.cpp" }
attributes #1 = { "sycl-module-id"="TU2.cpp" }
attributes #2 = { "sycl-module-id"="TU3.cpp" }
