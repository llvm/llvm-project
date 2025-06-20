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

declare [3 x i32] @llvm.OpenCL.image.get.size.2d(
  %opencl.image2d_t addrspace(1)*) nounwind readnone
declare [3 x i32] @llvm.OpenCL.image.get.size.3d(
  %opencl.image3d_t addrspace(1)*) nounwind readnone

declare [2 x i32] @llvm.OpenCL.image.get.format.2d(
  %opencl.image2d_t addrspace(1)*) nounwind readnone
declare [2 x i32] @llvm.OpenCL.image.get.format.3d(
  %opencl.image3d_t addrspace(1)*) nounwind readnone

define i32 @__clc_get_image_width_2d(
                          %opencl.image2d_t addrspace(1)* nocapture %img) #0 {
  %1 = tail call [3 x i32] @llvm.OpenCL.image.get.size.2d(
    %opencl.image2d_t addrspace(1)* %img)
  %2 = extractvalue [3 x i32] %1, 0
  ret i32 %2
}
define i32 @__clc_get_image_width_3d(
                          %opencl.image3d_t addrspace(1)* nocapture %img) #0 {
  %1 = tail call [3 x i32] @llvm.OpenCL.image.get.size.3d(
    %opencl.image3d_t addrspace(1)* %img)
  %2 = extractvalue [3 x i32] %1, 0
  ret i32 %2
}

define i32 @__clc_get_image_height_2d(
                          %opencl.image2d_t addrspace(1)* nocapture %img) #0 {
  %1 = tail call [3 x i32] @llvm.OpenCL.image.get.size.2d(
    %opencl.image2d_t addrspace(1)* %img)
  %2 = extractvalue [3 x i32] %1, 1
  ret i32 %2
}
define i32 @__clc_get_image_height_3d(
                          %opencl.image3d_t addrspace(1)* nocapture %img) #0 {
  %1 = tail call [3 x i32] @llvm.OpenCL.image.get.size.3d(
    %opencl.image3d_t addrspace(1)* %img)
  %2 = extractvalue [3 x i32] %1, 1
  ret i32 %2
}

define i32 @__clc_get_image_depth_3d(
                          %opencl.image3d_t addrspace(1)* nocapture %img) #0 {
  %1 = tail call [3 x i32] @llvm.OpenCL.image.get.size.3d(
    %opencl.image3d_t addrspace(1)* %img)
  %2 = extractvalue [3 x i32] %1, 2
  ret i32 %2
}

define i32 @__clc_get_image_channel_data_type_2d(
                          %opencl.image2d_t addrspace(1)* nocapture %img) #0 {
  %1 = tail call [2 x i32] @llvm.OpenCL.image.get.format.2d(
    %opencl.image2d_t addrspace(1)* %img)
  %2 = extractvalue [2 x i32] %1, 0
  ret i32 %2
}
define i32 @__clc_get_image_channel_data_type_3d(
                          %opencl.image3d_t addrspace(1)* nocapture %img) #0 {
  %1 = tail call [2 x i32] @llvm.OpenCL.image.get.format.3d(
    %opencl.image3d_t addrspace(1)* %img)
  %2 = extractvalue [2 x i32] %1, 0
  ret i32 %2
}

define i32 @__clc_get_image_channel_order_2d(
                          %opencl.image2d_t addrspace(1)* nocapture %img) #0 {
  %1 = tail call [2 x i32] @llvm.OpenCL.image.get.format.2d(
    %opencl.image2d_t addrspace(1)* %img)
  %2 = extractvalue [2 x i32] %1, 1
  ret i32 %2
}
define i32 @__clc_get_image_channel_order_3d(
                          %opencl.image3d_t addrspace(1)* nocapture %img) #0 {
  %1 = tail call [2 x i32] @llvm.OpenCL.image.get.format.3d(
    %opencl.image3d_t addrspace(1)* %img)
  %2 = extractvalue [2 x i32] %1, 1
  ret i32 %2
}

attributes #0 = { nounwind readnone alwaysinline }
