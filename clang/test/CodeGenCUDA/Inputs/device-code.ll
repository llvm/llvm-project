; Simple bit of IR to mimic CUDA's libdevice. We want to be
; able to link with it and we need to make sure all __nvvm_reflect
; calls are eliminated by the time PTX has been produced.

target triple = "nvptx-unknown-cuda"

declare i32 @__nvvm_reflect(ptr)

@"$str" = private addrspace(1) constant [8 x i8] c"USE_MUL\00"

define void @unused_subfunc(float %a) {
       ret void
}

define void @used_subfunc(float %a) {
       ret void
}

define float @_Z17device_mul_or_addff(float %a, float %b) {
  %reflect = call i32 @__nvvm_reflect(ptr addrspacecast (ptr addrspace(1) @"$str" to ptr))
  %cmp = icmp ne i32 %reflect, 0
  br i1 %cmp, label %use_mul, label %use_add

use_mul:
  %ret1 = fmul float %a, %b
  br label %exit

use_add:
  %ret2 = fadd float %a, %b
  br label %exit

exit:
  %ret = phi float [%ret1, %use_mul], [%ret2, %use_add]

  call void @used_subfunc(float %ret)

  ret float %ret
}
