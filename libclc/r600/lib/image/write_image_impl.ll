;;===----------------------------------------------------------------------===;;
;
; Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
; See https://llvm.org/LICENSE.txt for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;
;;===----------------------------------------------------------------------===;;

%opencl.image2d_t = type opaque
%opencl.image3d_t = type opaque

declare i32 @llvm.OpenCL.image.get.resource.id.2d(
  %opencl.image2d_t addrspace(1)*) nounwind readnone
declare i32 @llvm.OpenCL.image.get.resource.id.3d(
  %opencl.image3d_t addrspace(1)*) nounwind readnone

declare void @llvm.r600.rat.store.typed(<4 x i32> %color, <4 x i32> %coord, i32 %rat_id)

define void @__clc_write_imageui_2d(
    %opencl.image2d_t addrspace(1)* nocapture %img,
    <2 x i32> %coord, <4 x i32> %color) #0 {

  ; Coordinate int2 -> int4.
  %e0 = extractelement <2 x i32> %coord, i32 0
  %e1 = extractelement <2 x i32> %coord, i32 1
  %coord.0 = insertelement <4 x i32> undef,    i32 %e0, i32 0
  %coord.1 = insertelement <4 x i32> %coord.0, i32 %e1, i32 1
  %coord.2 = insertelement <4 x i32> %coord.1, i32 0,  i32 2
  %coord.3 = insertelement <4 x i32> %coord.2, i32 0,  i32 3

  ; Get RAT ID.
  %img_id = call i32 @llvm.OpenCL.image.get.resource.id.2d(
      %opencl.image2d_t addrspace(1)* %img)
  %rat_id = add i32 %img_id, 1

  ; Call store intrinsic.
  call void @llvm.r600.rat.store.typed(<4 x i32> %color, <4 x i32> %coord.3, i32 %rat_id)
  ret void
}

define void @__clc_write_imagei_2d(
    %opencl.image2d_t addrspace(1)* nocapture %img,
    <2 x i32> %coord, <4 x i32> %color) #0 {
  call void @__clc_write_imageui_2d(
      %opencl.image2d_t addrspace(1)* nocapture %img,
      <2 x i32> %coord, <4 x i32> %color)
  ret void
}

define void @__clc_write_imagef_2d(
    %opencl.image2d_t addrspace(1)* nocapture %img,
    <2 x i32> %coord, <4 x float> %color) #0 {
  %color.i32 = bitcast <4 x float> %color to <4 x i32>
  call void @__clc_write_imageui_2d(
      %opencl.image2d_t addrspace(1)* nocapture %img,
      <2 x i32> %coord, <4 x i32> %color.i32)
  ret void
}

attributes #0 = { alwaysinline }
