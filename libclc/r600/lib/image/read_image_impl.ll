target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"

%opencl.image2d_t = type opaque

declare <4 x float> @llvm.R600.tex(<4 x float>, i32, i32, i32, i32, i32, i32,
                                   i32, i32, i32) readnone
declare i32 @llvm.OpenCL.image.get.resource.id.2d(
  %opencl.image2d_t addrspace(1)*) nounwind readnone
declare i32 @llvm.OpenCL.sampler.get.resource.id(i32) readnone

define <4 x float> @__clc_v4f_from_v2f(<2 x float> %v) alwaysinline {
  %e0 = extractelement <2 x float> %v, i32 0
  %e1 = extractelement <2 x float> %v, i32 1
  %res.0 = insertelement <4 x float> undef,  float %e0, i32 0
  %res.1 = insertelement <4 x float> %res.0, float %e1, i32 1
  %res.2 = insertelement <4 x float> %res.1, float 0.0, i32 2
  %res.3 = insertelement <4 x float> %res.2, float 0.0, i32 3
  ret <4 x float> %res.3
}

define <4 x float> @__clc_read_imagef_tex(
    %opencl.image2d_t addrspace(1)* nocapture %img,
    i32 %sampler, <2 x float> %coord) alwaysinline {
entry:
  %coord_v4 = call <4 x float> @__clc_v4f_from_v2f(<2 x float> %coord)
  %smp_id = call i32 @llvm.OpenCL.sampler.get.resource.id(i32 %sampler)
  %img_id = call i32 @llvm.OpenCL.image.get.resource.id.2d(
      %opencl.image2d_t addrspace(1)* %img)
  %tex_id = add i32 %img_id, 2    ; First 2 IDs are reserved.

  %coord_norm = and i32 %sampler, 1
  %is_norm = icmp eq i32 %coord_norm, 1
  br i1 %is_norm, label %NormCoord, label %UnnormCoord
NormCoord:
  %data.norm = call <4 x float> @llvm.R600.tex(
      <4 x float> %coord_v4,
      i32 0, i32 0, i32 0,        ; Offset.
      i32 2, i32 %smp_id,
      i32 1, i32 1, i32 1, i32 1) ; Normalized coords.
  ret <4 x float> %data.norm
UnnormCoord:
  %data.unnorm = call <4 x float> @llvm.R600.tex(
      <4 x float> %coord_v4,
      i32 0, i32 0, i32 0,        ; Offset.
      i32 %tex_id, i32 %smp_id,
      i32 0, i32 0, i32 0, i32 0) ; Unnormalized coords.
  ret <4 x float> %data.unnorm
}
