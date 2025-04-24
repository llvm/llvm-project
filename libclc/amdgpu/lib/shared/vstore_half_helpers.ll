;;===----------------------------------------------------------------------===;;
;
; Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
; See https://llvm.org/LICENSE.txt for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;
;;===----------------------------------------------------------------------===;;

define void @__clc_vstore_half_float_helper__private(float %data, half addrspace(0)* nocapture %ptr) nounwind alwaysinline {
  %res = fptrunc float %data to half
  store half %res, half addrspace(0)* %ptr
  ret void
}

define void @__clc_vstore_half_float_helper__global(float %data, half addrspace(1)* nocapture %ptr) nounwind alwaysinline {
  %res = fptrunc float %data to half
  store half %res, half addrspace(1)* %ptr
  ret void
}

define void @__clc_vstore_half_float_helper__local(float %data, half addrspace(3)* nocapture %ptr) nounwind alwaysinline {
  %res = fptrunc float %data to half
  store half %res, half addrspace(3)* %ptr
  ret void
}

define void @__clc_vstore_half_double_helper__private(double %data, half addrspace(0)* nocapture %ptr) nounwind alwaysinline {
  %res = fptrunc double %data to half
  store half %res, half addrspace(0)* %ptr
  ret void
}

define void @__clc_vstore_half_double_helper__global(double %data, half addrspace(1)* nocapture %ptr) nounwind alwaysinline {
  %res = fptrunc double %data to half
  store half %res, half addrspace(1)* %ptr
  ret void
}

define void @__clc_vstore_half_double_helper__local(double %data, half addrspace(3)* nocapture %ptr) nounwind alwaysinline {
  %res = fptrunc double %data to half
  store half %res, half addrspace(3)* %ptr
  ret void
}
