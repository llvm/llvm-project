target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"
target triple = "amdgcn-amd-amdhsa"

define <4 x float> @__llvm_amdgcn_image_load_1d_v4f32_i32(i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %6 = tail call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 %0, i32 %1, <8 x i32> %2, i32 %3, i32 %4) #1
  ret <4 x float> %6
}

declare <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32, i32, <8 x i32>, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_load_2d_v4f32_i32(i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %7 = tail call <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32 %0, i32 %1, i32 %2, <8 x i32> %3, i32 %4, i32 %5) #1
  ret <4 x float> %7
}

declare <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_load_3d_v4f32_i32(i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %8 = tail call <4 x float> @llvm.amdgcn.image.load.3d.v4f32.i32(i32 %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #1
  ret <4 x float> %8
}

declare <4 x float> @llvm.amdgcn.image.load.3d.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_load_cube_v4f32_i32(i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %8 = tail call <4 x float> @llvm.amdgcn.image.load.cube.v4f32.i32(i32 %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #1
  ret <4 x float> %8
}

declare <4 x float> @llvm.amdgcn.image.load.cube.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_load_1darray_v4f32_i32(i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %7 = tail call <4 x float> @llvm.amdgcn.image.load.1darray.v4f32.i32(i32 %0, i32 %1, i32 %2, <8 x i32> %3, i32 %4, i32 %5) #1
  ret <4 x float> %7
}

declare <4 x float> @llvm.amdgcn.image.load.1darray.v4f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_load_2darray_v4f32_i32(i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %8 = tail call <4 x float> @llvm.amdgcn.image.load.2darray.v4f32.i32(i32 %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #1
  ret <4 x float> %8
}

declare <4 x float> @llvm.amdgcn.image.load.2darray.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_load_2dmsaa_v4f32_i32(i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %8 = tail call <4 x float> @llvm.amdgcn.image.load.2dmsaa.v4f32.i32(i32 %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #1
  ret <4 x float> %8
}

declare <4 x float> @llvm.amdgcn.image.load.2dmsaa.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_load_2darraymsaa_v4f32_i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %9 = tail call <4 x float> @llvm.amdgcn.image.load.2darraymsaa.v4f32.i32(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #1
  ret <4 x float> %9
}

declare <4 x float> @llvm.amdgcn.image.load.2darraymsaa.v4f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_load_mip_1d_v4f32_i32(i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %7 = tail call <4 x float> @llvm.amdgcn.image.load.mip.1d.v4f32.i32(i32 %0, i32 %1, i32 %2, <8 x i32> %3, i32 %4, i32 %5) #1
  ret <4 x float> %7
}

declare <4 x float> @llvm.amdgcn.image.load.mip.1d.v4f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_load_mip_2d_v4f32_i32(i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %8 = tail call <4 x float> @llvm.amdgcn.image.load.mip.2d.v4f32.i32(i32 %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #1
  ret <4 x float> %8
}

declare <4 x float> @llvm.amdgcn.image.load.mip.2d.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_load_mip_3d_v4f32_i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %9 = tail call <4 x float> @llvm.amdgcn.image.load.mip.3d.v4f32.i32(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #1
  ret <4 x float> %9
}

declare <4 x float> @llvm.amdgcn.image.load.mip.3d.v4f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_load_mip_cube_v4f32_i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %9 = tail call <4 x float> @llvm.amdgcn.image.load.mip.cube.v4f32.i32(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #1
  ret <4 x float> %9
}

declare <4 x float> @llvm.amdgcn.image.load.mip.cube.v4f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_load_mip_1darray_v4f32_i32(i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %8 = tail call <4 x float> @llvm.amdgcn.image.load.mip.1darray.v4f32.i32(i32 %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #1
  ret <4 x float> %8
}

declare <4 x float> @llvm.amdgcn.image.load.mip.1darray.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_load_mip_2darray_v4f32_i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %9 = tail call <4 x float> @llvm.amdgcn.image.load.mip.2darray.v4f32.i32(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #1
  ret <4 x float> %9
}

declare <4 x float> @llvm.amdgcn.image.load.mip.2darray.v4f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_load_1d_v4f16_i32(i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %6 = tail call <4 x half> @llvm.amdgcn.image.load.1d.v4f16.i32(i32 %0, i32 %1, <8 x i32> %2, i32 %3, i32 %4) #1
  ret <4 x half> %6
}

declare <4 x half> @llvm.amdgcn.image.load.1d.v4f16.i32(i32, i32, <8 x i32>, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_load_2d_v4f16_i32(i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %7 = tail call <4 x half> @llvm.amdgcn.image.load.2d.v4f16.i32(i32 %0, i32 %1, i32 %2, <8 x i32> %3, i32 %4, i32 %5) #1
  ret <4 x half> %7
}

declare <4 x half> @llvm.amdgcn.image.load.2d.v4f16.i32(i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_load_3d_v4f16_i32(i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %8 = tail call <4 x half> @llvm.amdgcn.image.load.3d.v4f16.i32(i32 %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #1
  ret <4 x half> %8
}

declare <4 x half> @llvm.amdgcn.image.load.3d.v4f16.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_load_cube_v4f16_i32(i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %8 = tail call <4 x half> @llvm.amdgcn.image.load.cube.v4f16.i32(i32 %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #1
  ret <4 x half> %8
}

declare <4 x half> @llvm.amdgcn.image.load.cube.v4f16.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_load_1darray_v4f16_i32(i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %7 = tail call <4 x half> @llvm.amdgcn.image.load.1darray.v4f16.i32(i32 %0, i32 %1, i32 %2, <8 x i32> %3, i32 %4, i32 %5) #1
  ret <4 x half> %7
}

declare <4 x half> @llvm.amdgcn.image.load.1darray.v4f16.i32(i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_load_2darray_v4f16_i32(i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %8 = tail call <4 x half> @llvm.amdgcn.image.load.2darray.v4f16.i32(i32 %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #1
  ret <4 x half> %8
}

declare <4 x half> @llvm.amdgcn.image.load.2darray.v4f16.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_load_2dmsaa_v4f16_i32(i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %8 = tail call <4 x half> @llvm.amdgcn.image.load.2dmsaa.v4f16.i32(i32 %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #1
  ret <4 x half> %8
}

declare <4 x half> @llvm.amdgcn.image.load.2dmsaa.v4f16.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_load_2darraymsaa_v4f16_i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %9 = tail call <4 x half> @llvm.amdgcn.image.load.2darraymsaa.v4f16.i32(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #1
  ret <4 x half> %9
}

declare <4 x half> @llvm.amdgcn.image.load.2darraymsaa.v4f16.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_load_mip_1d_v4f16_i32(i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %7 = tail call <4 x half> @llvm.amdgcn.image.load.mip.1d.v4f16.i32(i32 %0, i32 %1, i32 %2, <8 x i32> %3, i32 %4, i32 %5) #1
  ret <4 x half> %7
}

declare <4 x half> @llvm.amdgcn.image.load.mip.1d.v4f16.i32(i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_load_mip_2d_v4f16_i32(i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %8 = tail call <4 x half> @llvm.amdgcn.image.load.mip.2d.v4f16.i32(i32 %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #1
  ret <4 x half> %8
}

declare <4 x half> @llvm.amdgcn.image.load.mip.2d.v4f16.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_load_mip_3d_v4f16_i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %9 = tail call <4 x half> @llvm.amdgcn.image.load.mip.3d.v4f16.i32(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #1
  ret <4 x half> %9
}

declare <4 x half> @llvm.amdgcn.image.load.mip.3d.v4f16.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_load_mip_cube_v4f16_i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %9 = tail call <4 x half> @llvm.amdgcn.image.load.mip.cube.v4f16.i32(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #1
  ret <4 x half> %9
}

declare <4 x half> @llvm.amdgcn.image.load.mip.cube.v4f16.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_load_mip_1darray_v4f16_i32(i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %8 = tail call <4 x half> @llvm.amdgcn.image.load.mip.1darray.v4f16.i32(i32 %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #1
  ret <4 x half> %8
}

declare <4 x half> @llvm.amdgcn.image.load.mip.1darray.v4f16.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_load_mip_2darray_v4f16_i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %9 = tail call <4 x half> @llvm.amdgcn.image.load.mip.2darray.v4f16.i32(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #1
  ret <4 x half> %9
}

declare <4 x half> @llvm.amdgcn.image.load.mip.2darray.v4f16.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define float @__llvm_amdgcn_image_load_1d_f32_i32(i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %6 = tail call float @llvm.amdgcn.image.load.1d.f32.i32(i32 %0, i32 %1, <8 x i32> %2, i32 %3, i32 %4) #1
  ret float %6
}

declare float @llvm.amdgcn.image.load.1d.f32.i32(i32, i32, <8 x i32>, i32, i32) #1

define float @__llvm_amdgcn_image_load_2d_f32_i32(i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %7 = tail call float @llvm.amdgcn.image.load.2d.f32.i32(i32 %0, i32 %1, i32 %2, <8 x i32> %3, i32 %4, i32 %5) #1
  ret float %7
}

declare float @llvm.amdgcn.image.load.2d.f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1

define float @__llvm_amdgcn_image_load_3d_f32_i32(i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %8 = tail call float @llvm.amdgcn.image.load.3d.f32.i32(i32 %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #1
  ret float %8
}

declare float @llvm.amdgcn.image.load.3d.f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define float @__llvm_amdgcn_image_load_cube_f32_i32(i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %8 = tail call float @llvm.amdgcn.image.load.cube.f32.i32(i32 %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #1
  ret float %8
}

declare float @llvm.amdgcn.image.load.cube.f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define float @__llvm_amdgcn_image_load_1darray_f32_i32(i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %7 = tail call float @llvm.amdgcn.image.load.1darray.f32.i32(i32 %0, i32 %1, i32 %2, <8 x i32> %3, i32 %4, i32 %5) #1
  ret float %7
}

declare float @llvm.amdgcn.image.load.1darray.f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1

define float @__llvm_amdgcn_image_load_2darray_f32_i32(i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %8 = tail call float @llvm.amdgcn.image.load.2darray.f32.i32(i32 %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #1
  ret float %8
}

declare float @llvm.amdgcn.image.load.2darray.f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define float @__llvm_amdgcn_image_load_2dmsaa_f32_i32(i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %8 = tail call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #1
  ret float %8
}

declare float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define float @__llvm_amdgcn_image_load_2darraymsaa_f32_i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %9 = tail call float @llvm.amdgcn.image.load.2darraymsaa.f32.i32(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #1
  ret float %9
}

declare float @llvm.amdgcn.image.load.2darraymsaa.f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define float @__llvm_amdgcn_image_load_mip_1d_f32_i32(i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %7 = tail call float @llvm.amdgcn.image.load.mip.1d.f32.i32(i32 %0, i32 %1, i32 %2, <8 x i32> %3, i32 %4, i32 %5) #1
  ret float %7
}

declare float @llvm.amdgcn.image.load.mip.1d.f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1

define float @__llvm_amdgcn_image_load_mip_2d_f32_i32(i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %8 = tail call float @llvm.amdgcn.image.load.mip.2d.f32.i32(i32 %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #1
  ret float %8
}

declare float @llvm.amdgcn.image.load.mip.2d.f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define float @__llvm_amdgcn_image_load_mip_3d_f32_i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %9 = tail call float @llvm.amdgcn.image.load.mip.3d.f32.i32(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #1
  ret float %9
}

declare float @llvm.amdgcn.image.load.mip.3d.f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define float @__llvm_amdgcn_image_load_mip_cube_f32_i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %9 = tail call float @llvm.amdgcn.image.load.mip.cube.f32.i32(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #1
  ret float %9
}

declare float @llvm.amdgcn.image.load.mip.cube.f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define float @__llvm_amdgcn_image_load_mip_1darray_f32_i32(i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %8 = tail call float @llvm.amdgcn.image.load.mip.1darray.f32.i32(i32 %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #1
  ret float %8
}

declare float @llvm.amdgcn.image.load.mip.1darray.f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define float @__llvm_amdgcn_image_load_mip_2darray_f32_i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #0 {
  %9 = tail call float @llvm.amdgcn.image.load.mip.2darray.f32.i32(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #1
  ret float %9
}

declare float @llvm.amdgcn.image.load.mip.2darray.f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

define void @__llvm_amdgcn_image_store_1d_v4f32_i32(<4 x float>, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %0, i32 %1, i32 %2, <8 x i32> %3, i32 %4, i32 %5) #3
  ret void
}

declare void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float>, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_2d_v4f32_i32(<4 x float>, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float> %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #3
  ret void
}

declare void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float>, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_3d_v4f32_i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.3d.v4f32.i32(<4 x float> %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #3
  ret void
}

declare void @llvm.amdgcn.image.store.3d.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_cube_v4f32_i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.cube.v4f32.i32(<4 x float> %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #3
  ret void
}

declare void @llvm.amdgcn.image.store.cube.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_1darray_v4f32_i32(<4 x float>, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.1darray.v4f32.i32(<4 x float> %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #3
  ret void
}

declare void @llvm.amdgcn.image.store.1darray.v4f32.i32(<4 x float>, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_2darray_v4f32_i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.2darray.v4f32.i32(<4 x float> %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #3
  ret void
}

declare void @llvm.amdgcn.image.store.2darray.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_2dmsaa_v4f32_i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.2dmsaa.v4f32.i32(<4 x float> %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #3
  ret void
}

declare void @llvm.amdgcn.image.store.2dmsaa.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_2darraymsaa_v4f32_i32(<4 x float>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.2darraymsaa.v4f32.i32(<4 x float> %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, <8 x i32> %6, i32 %7, i32 %8) #3
  ret void
}

declare void @llvm.amdgcn.image.store.2darraymsaa.v4f32.i32(<4 x float>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_mip_1d_v4f32_i32(<4 x float>, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.mip.1d.v4f32.i32(<4 x float> %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #3
  ret void
}

declare void @llvm.amdgcn.image.store.mip.1d.v4f32.i32(<4 x float>, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_mip_2d_v4f32_i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.mip.2d.v4f32.i32(<4 x float> %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #3
  ret void
}

declare void @llvm.amdgcn.image.store.mip.2d.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_mip_3d_v4f32_i32(<4 x float>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.mip.3d.v4f32.i32(<4 x float> %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, <8 x i32> %6, i32 %7, i32 %8) #3
  ret void
}

declare void @llvm.amdgcn.image.store.mip.3d.v4f32.i32(<4 x float>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_mip_cube_v4f32_i32(<4 x float>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.mip.cube.v4f32.i32(<4 x float> %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, <8 x i32> %6, i32 %7, i32 %8) #3
  ret void
}

declare void @llvm.amdgcn.image.store.mip.cube.v4f32.i32(<4 x float>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_mip_1darray_v4f32_i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.mip.1darray.v4f32.i32(<4 x float> %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #3
  ret void
}

declare void @llvm.amdgcn.image.store.mip.1darray.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_mip_2darray_v4f32_i32(<4 x float>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.mip.2darray.v4f32.i32(<4 x float> %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, <8 x i32> %6, i32 %7, i32 %8) #3
  ret void
}

declare void @llvm.amdgcn.image.store.mip.2darray.v4f32.i32(<4 x float>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_1d_v4f16_i32(<4 x half>, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.1d.v4f16.i32(<4 x half> %0, i32 %1, i32 %2, <8 x i32> %3, i32 %4, i32 %5) #3
  ret void
}

declare void @llvm.amdgcn.image.store.1d.v4f16.i32(<4 x half>, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_2d_v4f16_i32(<4 x half>, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.2d.v4f16.i32(<4 x half> %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #3
  ret void
}

declare void @llvm.amdgcn.image.store.2d.v4f16.i32(<4 x half>, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_3d_v4f16_i32(<4 x half>, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.3d.v4f16.i32(<4 x half> %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #3
  ret void
}

declare void @llvm.amdgcn.image.store.3d.v4f16.i32(<4 x half>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_cube_v4f16_i32(<4 x half>, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.cube.v4f16.i32(<4 x half> %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #3
  ret void
}

declare void @llvm.amdgcn.image.store.cube.v4f16.i32(<4 x half>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_1darray_v4f16_i32(<4 x half>, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.1darray.v4f16.i32(<4 x half> %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #3
  ret void
}

declare void @llvm.amdgcn.image.store.1darray.v4f16.i32(<4 x half>, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_2darray_v4f16_i32(<4 x half>, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.2darray.v4f16.i32(<4 x half> %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #3
  ret void
}

declare void @llvm.amdgcn.image.store.2darray.v4f16.i32(<4 x half>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_2dmsaa_v4f16_i32(<4 x half>, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.2dmsaa.v4f16.i32(<4 x half> %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #3
  ret void
}

declare void @llvm.amdgcn.image.store.2dmsaa.v4f16.i32(<4 x half>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_2darraymsaa_v4f16_i32(<4 x half>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.2darraymsaa.v4f16.i32(<4 x half> %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, <8 x i32> %6, i32 %7, i32 %8) #3
  ret void
}

declare void @llvm.amdgcn.image.store.2darraymsaa.v4f16.i32(<4 x half>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_mip_1d_v4f16_i32(<4 x half>, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.mip.1d.v4f16.i32(<4 x half> %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #3
  ret void
}

declare void @llvm.amdgcn.image.store.mip.1d.v4f16.i32(<4 x half>, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_mip_2d_v4f16_i32(<4 x half>, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.mip.2d.v4f16.i32(<4 x half> %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #3
  ret void
}

declare void @llvm.amdgcn.image.store.mip.2d.v4f16.i32(<4 x half>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_mip_3d_v4f16_i32(<4 x half>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.mip.3d.v4f16.i32(<4 x half> %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, <8 x i32> %6, i32 %7, i32 %8) #3
  ret void
}

declare void @llvm.amdgcn.image.store.mip.3d.v4f16.i32(<4 x half>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_mip_cube_v4f16_i32(<4 x half>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.mip.cube.v4f16.i32(<4 x half> %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, <8 x i32> %6, i32 %7, i32 %8) #3
  ret void
}

declare void @llvm.amdgcn.image.store.mip.cube.v4f16.i32(<4 x half>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_mip_1darray_v4f16_i32(<4 x half>, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.mip.1darray.v4f16.i32(<4 x half> %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #3
  ret void
}

declare void @llvm.amdgcn.image.store.mip.1darray.v4f16.i32(<4 x half>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_mip_2darray_v4f16_i32(<4 x half>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.mip.2darray.v4f16.i32(<4 x half> %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, <8 x i32> %6, i32 %7, i32 %8) #3
  ret void
}

declare void @llvm.amdgcn.image.store.mip.2darray.v4f16.i32(<4 x half>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_1d_f32_i32(float, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.1d.f32.i32(float %0, i32 %1, i32 %2, <8 x i32> %3, i32 %4, i32 %5) #3
  ret void
}

declare void @llvm.amdgcn.image.store.1d.f32.i32(float, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_2d_f32_i32(float, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.2d.f32.i32(float %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #3
  ret void
}

declare void @llvm.amdgcn.image.store.2d.f32.i32(float, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_3d_f32_i32(float, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.3d.f32.i32(float %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #3
  ret void
}

declare void @llvm.amdgcn.image.store.3d.f32.i32(float, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_cube_f32_i32(float, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.cube.f32.i32(float %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #3
  ret void
}

declare void @llvm.amdgcn.image.store.cube.f32.i32(float, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_1darray_f32_i32(float, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.1darray.f32.i32(float %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #3
  ret void
}

declare void @llvm.amdgcn.image.store.1darray.f32.i32(float, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_2darray_f32_i32(float, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.2darray.f32.i32(float %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #3
  ret void
}

declare void @llvm.amdgcn.image.store.2darray.f32.i32(float, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_2dmsaa_f32_i32(float, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.2dmsaa.f32.i32(float %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #3
  ret void
}

declare void @llvm.amdgcn.image.store.2dmsaa.f32.i32(float, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_2darraymsaa_f32_i32(float, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.2darraymsaa.f32.i32(float %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, <8 x i32> %6, i32 %7, i32 %8) #3
  ret void
}

declare void @llvm.amdgcn.image.store.2darraymsaa.f32.i32(float, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_mip_1d_f32_i32(float, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.mip.1d.f32.i32(float %0, i32 %1, i32 %2, i32 %3, <8 x i32> %4, i32 %5, i32 %6) #3
  ret void
}

declare void @llvm.amdgcn.image.store.mip.1d.f32.i32(float, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_mip_2d_f32_i32(float, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.mip.2d.f32.i32(float %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #3
  ret void
}

declare void @llvm.amdgcn.image.store.mip.2d.f32.i32(float, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_mip_3d_f32_i32(float, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.mip.3d.f32.i32(float %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, <8 x i32> %6, i32 %7, i32 %8) #3
  ret void
}

declare void @llvm.amdgcn.image.store.mip.3d.f32.i32(float, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_mip_cube_f32_i32(float, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.mip.cube.f32.i32(float %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, <8 x i32> %6, i32 %7, i32 %8) #3
  ret void
}

declare void @llvm.amdgcn.image.store.mip.cube.f32.i32(float, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_mip_1darray_f32_i32(float, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.mip.1darray.f32.i32(float %0, i32 %1, i32 %2, i32 %3, i32 %4, <8 x i32> %5, i32 %6, i32 %7) #3
  ret void
}

declare void @llvm.amdgcn.image.store.mip.1darray.f32.i32(float, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define void @__llvm_amdgcn_image_store_mip_2darray_f32_i32(float, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) local_unnamed_addr #2 {
  tail call void @llvm.amdgcn.image.store.mip.2darray.f32.i32(float %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, <8 x i32> %6, i32 %7, i32 %8) #3
  ret void
}

declare void @llvm.amdgcn.image.store.mip.2darray.f32.i32(float, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

define <4 x float> @__llvm_amdgcn_image_sample_lz_1d_v4f32_f32(i32, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %8 = tail call <4 x float> @llvm.amdgcn.image.sample.lz.1d.v4f32.f32(i32 %0, float %1, <8 x i32> %2, <4 x i32> %3, i1 zeroext %4, i32 %5, i32 %6) #1
  ret <4 x float> %8
}

declare <4 x float> @llvm.amdgcn.image.sample.lz.1d.v4f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_sample_l_1d_v4f32_f32(i32, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %9 = tail call <4 x float> @llvm.amdgcn.image.sample.l.1d.v4f32.f32(i32 %0, float %1, float %2, <8 x i32> %3, <4 x i32> %4, i1 zeroext %5, i32 %6, i32 %7) #1
  ret <4 x float> %9
}

declare <4 x float> @llvm.amdgcn.image.sample.l.1d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_sample_d_1d_v4f32_f32_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call <4 x float> @llvm.amdgcn.image.sample.d.1d.v4f32.f32.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret <4 x float> %10
}

declare <4 x float> @llvm.amdgcn.image.sample.d.1d.v4f32.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_sample_lz_2d_v4f32_f32(i32, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %9 = tail call <4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f32(i32 %0, float %1, float %2, <8 x i32> %3, <4 x i32> %4, i1 zeroext %5, i32 %6, i32 %7) #1
  ret <4 x float> %9
}

declare <4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_sample_l_2d_v4f32_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call <4 x float> @llvm.amdgcn.image.sample.l.2d.v4f32.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret <4 x float> %10
}

declare <4 x float> @llvm.amdgcn.image.sample.l.2d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_sample_d_2d_v4f32_f32_f32(i32, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %13 = tail call <4 x float> @llvm.amdgcn.image.sample.d.2d.v4f32.f32.f32(i32 %0, float %1, float %2, float %3, float %4, float %5, float %6, <8 x i32> %7, <4 x i32> %8, i1 zeroext %9, i32 %10, i32 %11) #1
  ret <4 x float> %13
}

declare <4 x float> @llvm.amdgcn.image.sample.d.2d.v4f32.f32.f32(i32, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_sample_lz_3d_v4f32_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call <4 x float> @llvm.amdgcn.image.sample.lz.3d.v4f32.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret <4 x float> %10
}

declare <4 x float> @llvm.amdgcn.image.sample.lz.3d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_sample_l_3d_v4f32_f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %11 = tail call <4 x float> @llvm.amdgcn.image.sample.l.3d.v4f32.f32(i32 %0, float %1, float %2, float %3, float %4, <8 x i32> %5, <4 x i32> %6, i1 zeroext %7, i32 %8, i32 %9) #1
  ret <4 x float> %11
}

declare <4 x float> @llvm.amdgcn.image.sample.l.3d.v4f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_sample_d_3d_v4f32_f32_f32(i32, float, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %16 = tail call <4 x float> @llvm.amdgcn.image.sample.d.3d.v4f32.f32.f32(i32 %0, float %1, float %2, float %3, float %4, float %5, float %6, float %7, float %8, float %9, <8 x i32> %10, <4 x i32> %11, i1 zeroext %12, i32 %13, i32 %14) #1
  ret <4 x float> %16
}

declare <4 x float> @llvm.amdgcn.image.sample.d.3d.v4f32.f32.f32(i32, float, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_sample_lz_cube_v4f32_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call <4 x float> @llvm.amdgcn.image.sample.lz.cube.v4f32.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret <4 x float> %10
}

declare <4 x float> @llvm.amdgcn.image.sample.lz.cube.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_sample_l_cube_v4f32_f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %11 = tail call <4 x float> @llvm.amdgcn.image.sample.l.cube.v4f32.f32(i32 %0, float %1, float %2, float %3, float %4, <8 x i32> %5, <4 x i32> %6, i1 zeroext %7, i32 %8, i32 %9) #1
  ret <4 x float> %11
}

declare <4 x float> @llvm.amdgcn.image.sample.l.cube.v4f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_sample_d_cube_v4f32_f32_f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %14 = tail call <4 x float> @llvm.amdgcn.image.sample.d.cube.v4f32.f32.f32(i32 %0, float %1, float %2, float %3, float %4, float %5, float %6, float %7, <8 x i32> %8, <4 x i32> %9, i1 zeroext %10, i32 %11, i32 %12) #1
  ret <4 x float> %14
}

declare <4 x float> @llvm.amdgcn.image.sample.d.cube.v4f32.f32.f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_sample_lz_1darray_v4f32_f32(i32, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %9 = tail call <4 x float> @llvm.amdgcn.image.sample.lz.1darray.v4f32.f32(i32 %0, float %1, float %2, <8 x i32> %3, <4 x i32> %4, i1 zeroext %5, i32 %6, i32 %7) #1
  ret <4 x float> %9
}

declare <4 x float> @llvm.amdgcn.image.sample.lz.1darray.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_sample_l_1darray_v4f32_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call <4 x float> @llvm.amdgcn.image.sample.l.1darray.v4f32.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret <4 x float> %10
}

declare <4 x float> @llvm.amdgcn.image.sample.l.1darray.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_sample_d_1darray_v4f32_f32_f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %11 = tail call <4 x float> @llvm.amdgcn.image.sample.d.1darray.v4f32.f32.f32(i32 %0, float %1, float %2, float %3, float %4, <8 x i32> %5, <4 x i32> %6, i1 zeroext %7, i32 %8, i32 %9) #1
  ret <4 x float> %11
}

declare <4 x float> @llvm.amdgcn.image.sample.d.1darray.v4f32.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_sample_lz_2darray_v4f32_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call <4 x float> @llvm.amdgcn.image.sample.lz.2darray.v4f32.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret <4 x float> %10
}

declare <4 x float> @llvm.amdgcn.image.sample.lz.2darray.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_sample_l_2darray_v4f32_f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %11 = tail call <4 x float> @llvm.amdgcn.image.sample.l.2darray.v4f32.f32(i32 %0, float %1, float %2, float %3, float %4, <8 x i32> %5, <4 x i32> %6, i1 zeroext %7, i32 %8, i32 %9) #1
  ret <4 x float> %11
}

declare <4 x float> @llvm.amdgcn.image.sample.l.2darray.v4f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_sample_d_2darray_v4f32_f32_f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %14 = tail call <4 x float> @llvm.amdgcn.image.sample.d.2darray.v4f32.f32.f32(i32 %0, float %1, float %2, float %3, float %4, float %5, float %6, float %7, <8 x i32> %8, <4 x i32> %9, i1 zeroext %10, i32 %11, i32 %12) #1
  ret <4 x float> %14
}

declare <4 x float> @llvm.amdgcn.image.sample.d.2darray.v4f32.f32.f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_sample_lz_1d_v4f16_f32(i32, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %8 = tail call <4 x half> @llvm.amdgcn.image.sample.lz.1d.v4f16.f32(i32 %0, float %1, <8 x i32> %2, <4 x i32> %3, i1 zeroext %4, i32 %5, i32 %6) #1
  ret <4 x half> %8
}

declare <4 x half> @llvm.amdgcn.image.sample.lz.1d.v4f16.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_sample_l_1d_v4f16_f32(i32, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %9 = tail call <4 x half> @llvm.amdgcn.image.sample.l.1d.v4f16.f32(i32 %0, float %1, float %2, <8 x i32> %3, <4 x i32> %4, i1 zeroext %5, i32 %6, i32 %7) #1
  ret <4 x half> %9
}

declare <4 x half> @llvm.amdgcn.image.sample.l.1d.v4f16.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_sample_d_1d_v4f16_f32_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call <4 x half> @llvm.amdgcn.image.sample.d.1d.v4f16.f32.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret <4 x half> %10
}

declare <4 x half> @llvm.amdgcn.image.sample.d.1d.v4f16.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_sample_lz_2d_v4f16_f32(i32, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %9 = tail call <4 x half> @llvm.amdgcn.image.sample.lz.2d.v4f16.f32(i32 %0, float %1, float %2, <8 x i32> %3, <4 x i32> %4, i1 zeroext %5, i32 %6, i32 %7) #1
  ret <4 x half> %9
}

declare <4 x half> @llvm.amdgcn.image.sample.lz.2d.v4f16.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_sample_l_2d_v4f16_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call <4 x half> @llvm.amdgcn.image.sample.l.2d.v4f16.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret <4 x half> %10
}

declare <4 x half> @llvm.amdgcn.image.sample.l.2d.v4f16.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_sample_d_2d_v4f16_f32_f32(i32, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %13 = tail call <4 x half> @llvm.amdgcn.image.sample.d.2d.v4f16.f32.f32(i32 %0, float %1, float %2, float %3, float %4, float %5, float %6, <8 x i32> %7, <4 x i32> %8, i1 zeroext %9, i32 %10, i32 %11) #1
  ret <4 x half> %13
}

declare <4 x half> @llvm.amdgcn.image.sample.d.2d.v4f16.f32.f32(i32, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_sample_lz_3d_v4f16_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call <4 x half> @llvm.amdgcn.image.sample.lz.3d.v4f16.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret <4 x half> %10
}

declare <4 x half> @llvm.amdgcn.image.sample.lz.3d.v4f16.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_sample_l_3d_v4f16_f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %11 = tail call <4 x half> @llvm.amdgcn.image.sample.l.3d.v4f16.f32(i32 %0, float %1, float %2, float %3, float %4, <8 x i32> %5, <4 x i32> %6, i1 zeroext %7, i32 %8, i32 %9) #1
  ret <4 x half> %11
}

declare <4 x half> @llvm.amdgcn.image.sample.l.3d.v4f16.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_sample_d_3d_v4f16_f32_f32(i32, float, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %16 = tail call <4 x half> @llvm.amdgcn.image.sample.d.3d.v4f16.f32.f32(i32 %0, float %1, float %2, float %3, float %4, float %5, float %6, float %7, float %8, float %9, <8 x i32> %10, <4 x i32> %11, i1 zeroext %12, i32 %13, i32 %14) #1
  ret <4 x half> %16
}

declare <4 x half> @llvm.amdgcn.image.sample.d.3d.v4f16.f32.f32(i32, float, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_sample_lz_cube_v4f16_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call <4 x half> @llvm.amdgcn.image.sample.lz.cube.v4f16.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret <4 x half> %10
}

declare <4 x half> @llvm.amdgcn.image.sample.lz.cube.v4f16.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_sample_l_cube_v4f16_f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %11 = tail call <4 x half> @llvm.amdgcn.image.sample.l.cube.v4f16.f32(i32 %0, float %1, float %2, float %3, float %4, <8 x i32> %5, <4 x i32> %6, i1 zeroext %7, i32 %8, i32 %9) #1
  ret <4 x half> %11
}

declare <4 x half> @llvm.amdgcn.image.sample.l.cube.v4f16.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_sample_d_cube_v4f16_f32_f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %14 = tail call <4 x half> @llvm.amdgcn.image.sample.d.cube.v4f16.f32.f32(i32 %0, float %1, float %2, float %3, float %4, float %5, float %6, float %7, <8 x i32> %8, <4 x i32> %9, i1 zeroext %10, i32 %11, i32 %12) #1
  ret <4 x half> %14
}

declare <4 x half> @llvm.amdgcn.image.sample.d.cube.v4f16.f32.f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_sample_lz_1darray_v4f16_f32(i32, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %9 = tail call <4 x half> @llvm.amdgcn.image.sample.lz.1darray.v4f16.f32(i32 %0, float %1, float %2, <8 x i32> %3, <4 x i32> %4, i1 zeroext %5, i32 %6, i32 %7) #1
  ret <4 x half> %9
}

declare <4 x half> @llvm.amdgcn.image.sample.lz.1darray.v4f16.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_sample_l_1darray_v4f16_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call <4 x half> @llvm.amdgcn.image.sample.l.1darray.v4f16.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret <4 x half> %10
}

declare <4 x half> @llvm.amdgcn.image.sample.l.1darray.v4f16.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_sample_d_1darray_v4f16_f32_f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %11 = tail call <4 x half> @llvm.amdgcn.image.sample.d.1darray.v4f16.f32.f32(i32 %0, float %1, float %2, float %3, float %4, <8 x i32> %5, <4 x i32> %6, i1 zeroext %7, i32 %8, i32 %9) #1
  ret <4 x half> %11
}

declare <4 x half> @llvm.amdgcn.image.sample.d.1darray.v4f16.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_sample_lz_2darray_v4f16_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call <4 x half> @llvm.amdgcn.image.sample.lz.2darray.v4f16.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret <4 x half> %10
}

declare <4 x half> @llvm.amdgcn.image.sample.lz.2darray.v4f16.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_sample_l_2darray_v4f16_f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %11 = tail call <4 x half> @llvm.amdgcn.image.sample.l.2darray.v4f16.f32(i32 %0, float %1, float %2, float %3, float %4, <8 x i32> %5, <4 x i32> %6, i1 zeroext %7, i32 %8, i32 %9) #1
  ret <4 x half> %11
}

declare <4 x half> @llvm.amdgcn.image.sample.l.2darray.v4f16.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x half> @__llvm_amdgcn_image_sample_d_2darray_v4f16_f32_f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %14 = tail call <4 x half> @llvm.amdgcn.image.sample.d.2darray.v4f16.f32.f32(i32 %0, float %1, float %2, float %3, float %4, float %5, float %6, float %7, <8 x i32> %8, <4 x i32> %9, i1 zeroext %10, i32 %11, i32 %12) #1
  ret <4 x half> %14
}

declare <4 x half> @llvm.amdgcn.image.sample.d.2darray.v4f16.f32.f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define float @__llvm_amdgcn_image_sample_lz_1d_f32_f32(i32, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %8 = tail call float @llvm.amdgcn.image.sample.lz.1d.f32.f32(i32 %0, float %1, <8 x i32> %2, <4 x i32> %3, i1 zeroext %4, i32 %5, i32 %6) #1
  ret float %8
}

declare float @llvm.amdgcn.image.sample.lz.1d.f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define float @__llvm_amdgcn_image_sample_l_1d_f32_f32(i32, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %9 = tail call float @llvm.amdgcn.image.sample.l.1d.f32.f32(i32 %0, float %1, float %2, <8 x i32> %3, <4 x i32> %4, i1 zeroext %5, i32 %6, i32 %7) #1
  ret float %9
}

declare float @llvm.amdgcn.image.sample.l.1d.f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define float @__llvm_amdgcn_image_sample_d_1d_f32_f32_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call float @llvm.amdgcn.image.sample.d.1d.f32.f32.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret float %10
}

declare float @llvm.amdgcn.image.sample.d.1d.f32.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define float @__llvm_amdgcn_image_sample_lz_2d_f32_f32(i32, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %9 = tail call float @llvm.amdgcn.image.sample.lz.2d.f32.f32(i32 %0, float %1, float %2, <8 x i32> %3, <4 x i32> %4, i1 zeroext %5, i32 %6, i32 %7) #1
  ret float %9
}

declare float @llvm.amdgcn.image.sample.lz.2d.f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define float @__llvm_amdgcn_image_sample_l_2d_f32_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call float @llvm.amdgcn.image.sample.l.2d.f32.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret float %10
}

declare float @llvm.amdgcn.image.sample.l.2d.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define float @__llvm_amdgcn_image_sample_d_2d_f32_f32_f32(i32, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %13 = tail call float @llvm.amdgcn.image.sample.d.2d.f32.f32.f32(i32 %0, float %1, float %2, float %3, float %4, float %5, float %6, <8 x i32> %7, <4 x i32> %8, i1 zeroext %9, i32 %10, i32 %11) #1
  ret float %13
}

declare float @llvm.amdgcn.image.sample.d.2d.f32.f32.f32(i32, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define float @__llvm_amdgcn_image_sample_lz_3d_f32_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call float @llvm.amdgcn.image.sample.lz.3d.f32.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret float %10
}

declare float @llvm.amdgcn.image.sample.lz.3d.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define float @__llvm_amdgcn_image_sample_l_3d_f32_f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %11 = tail call float @llvm.amdgcn.image.sample.l.3d.f32.f32(i32 %0, float %1, float %2, float %3, float %4, <8 x i32> %5, <4 x i32> %6, i1 zeroext %7, i32 %8, i32 %9) #1
  ret float %11
}

declare float @llvm.amdgcn.image.sample.l.3d.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define float @__llvm_amdgcn_image_sample_d_3d_f32_f32_f32(i32, float, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %16 = tail call float @llvm.amdgcn.image.sample.d.3d.f32.f32.f32(i32 %0, float %1, float %2, float %3, float %4, float %5, float %6, float %7, float %8, float %9, <8 x i32> %10, <4 x i32> %11, i1 zeroext %12, i32 %13, i32 %14) #1
  ret float %16
}

declare float @llvm.amdgcn.image.sample.d.3d.f32.f32.f32(i32, float, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define float @__llvm_amdgcn_image_sample_lz_cube_f32_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call float @llvm.amdgcn.image.sample.lz.cube.f32.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret float %10
}

declare float @llvm.amdgcn.image.sample.lz.cube.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define float @__llvm_amdgcn_image_sample_l_cube_f32_f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %11 = tail call float @llvm.amdgcn.image.sample.l.cube.f32.f32(i32 %0, float %1, float %2, float %3, float %4, <8 x i32> %5, <4 x i32> %6, i1 zeroext %7, i32 %8, i32 %9) #1
  ret float %11
}

declare float @llvm.amdgcn.image.sample.l.cube.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define float @__llvm_amdgcn_image_sample_d_cube_f32_f32_f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %14 = tail call float @llvm.amdgcn.image.sample.d.cube.f32.f32.f32(i32 %0, float %1, float %2, float %3, float %4, float %5, float %6, float %7, <8 x i32> %8, <4 x i32> %9, i1 zeroext %10, i32 %11, i32 %12) #1
  ret float %14
}

declare float @llvm.amdgcn.image.sample.d.cube.f32.f32.f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define float @__llvm_amdgcn_image_sample_lz_1darray_f32_f32(i32, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %9 = tail call float @llvm.amdgcn.image.sample.lz.1darray.f32.f32(i32 %0, float %1, float %2, <8 x i32> %3, <4 x i32> %4, i1 zeroext %5, i32 %6, i32 %7) #1
  ret float %9
}

declare float @llvm.amdgcn.image.sample.lz.1darray.f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define float @__llvm_amdgcn_image_sample_l_1darray_f32_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call float @llvm.amdgcn.image.sample.l.1darray.f32.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret float %10
}

declare float @llvm.amdgcn.image.sample.l.1darray.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define float @__llvm_amdgcn_image_sample_d_1darray_f32_f32_f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %11 = tail call float @llvm.amdgcn.image.sample.d.1darray.f32.f32.f32(i32 %0, float %1, float %2, float %3, float %4, <8 x i32> %5, <4 x i32> %6, i1 zeroext %7, i32 %8, i32 %9) #1
  ret float %11
}

declare float @llvm.amdgcn.image.sample.d.1darray.f32.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define float @__llvm_amdgcn_image_sample_lz_2darray_f32_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call float @llvm.amdgcn.image.sample.lz.2darray.f32.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret float %10
}

declare float @llvm.amdgcn.image.sample.lz.2darray.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define float @__llvm_amdgcn_image_sample_l_2darray_f32_f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %11 = tail call float @llvm.amdgcn.image.sample.l.2darray.f32.f32(i32 %0, float %1, float %2, float %3, float %4, <8 x i32> %5, <4 x i32> %6, i1 zeroext %7, i32 %8, i32 %9) #1
  ret float %11
}

declare float @llvm.amdgcn.image.sample.l.2darray.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define float @__llvm_amdgcn_image_sample_d_2darray_f32_f32_f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %14 = tail call float @llvm.amdgcn.image.sample.d.2darray.f32.f32.f32(i32 %0, float %1, float %2, float %3, float %4, float %5, float %6, float %7, <8 x i32> %8, <4 x i32> %9, i1 zeroext %10, i32 %11, i32 %12) #1
  ret float %14
}

declare float @llvm.amdgcn.image.sample.d.2darray.f32.f32.f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_gather4_lz_2d_v4f32_f32(i32, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %9 = tail call <4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f32(i32 %0, float %1, float %2, <8 x i32> %3, <4 x i32> %4, i1 zeroext %5, i32 %6, i32 %7) #1
  ret <4 x float> %9
}

declare <4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_gather4_l_2d_v4f32_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call <4 x float> @llvm.image.amdgcn.gather4.l.2d.v4f32.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret <4 x float> %10
}

declare <4 x float> @llvm.image.amdgcn.gather4.l.2d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_gather4_lz_cube_v4f32_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call <4 x float> @llvm.amdgcn.image.gather4.lz.cube.v4f32.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret <4 x float> %10
}

declare <4 x float> @llvm.amdgcn.image.gather4.lz.cube.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_gather4_l_cube_v4f32_f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %11 = tail call <4 x float> @llvm.amdgcn.image.gather4.l.cube.v4f32.f32(i32 %0, float %1, float %2, float %3, float %4, <8 x i32> %5, <4 x i32> %6, i1 zeroext %7, i32 %8, i32 %9) #1
  ret <4 x float> %11
}

declare <4 x float> @llvm.amdgcn.image.gather4.l.cube.v4f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_gather4_lz_2darray_v4f32_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call <4 x float> @llvm.amdgcn.image.gather4.lz.2darray.v4f32.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret <4 x float> %10
}

declare <4 x float> @llvm.amdgcn.image.gather4.lz.2darray.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_image_gather4_l_2darray_v4f32_f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %11 = tail call <4 x float> @llvm.amdgcn.image.gather4.l.2darray.v4f32.f32(i32 %0, float %1, float %2, float %3, float %4, <8 x i32> %5, <4 x i32> %6, i1 zeroext %7, i32 %8, i32 %9) #1
  ret <4 x float> %11
}

declare <4 x float> @llvm.amdgcn.image.gather4.l.2darray.v4f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

define <4 x float> @__llvm_amdgcn_gather_4h_2d_v4f32_f32(i32, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %9 = tail call <4 x float> @llvm.amdgcn.image.gather.4h.2d.v4f32.f32(i32 %0, float %1, float %2, <8 x i32> %3, <4 x i32> %4, i1 zeroext %5, i32 %6, i32 %7) #1
  ret <4 x float> %9
}

declare <4 x float> @llvm.amdgcn.image.gather.4h.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) #1

define <4 x float> @__llvm_amdgcn_gather_4h_cube_v4f32_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call <4 x float> @llvm.image.amdgcn.gather.4h.cube.v4f32.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret <4 x float> %10
}

declare <4 x float> @llvm.image.amdgcn.gather.4h.cube.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) #1

define <4 x float> @__llvm_amdgcn_gather_4h_2darray_v4f32_f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) local_unnamed_addr #0 {
  %10 = tail call <4 x float> @llvm.image.amdgcn.gather.4h.2darray.v4f32.f32(i32 %0, float %1, float %2, float %3, <8 x i32> %4, <4 x i32> %5, i1 zeroext %6, i32 %7, i32 %8) #1
  ret <4 x float> %10
}

declare <4 x float> @llvm.image.amdgcn.gather.4h.2darray.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1 zeroext, i32, i32) #1

attributes #0 = { alwaysinline nounwind readonly }
attributes #1 = { nounwind readonly }
attributes #2 = { alwaysinline nounwind writeonly }
attributes #3 = { nounwind writeonly }

