target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8"
target triple = "amdgcn-amd-amdhsa"

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_load_1d_v4f32_i32(i32 %arg1, <8 x i32> %arg2) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 noundef 15, i32 %arg1, <8 x i32> %arg2, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 immarg, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_load_2d_v4f32_i32(i32 %arg1, i32 %arg2, <8 x i32> %arg3) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32 noundef 15, i32 %arg1, i32 %arg2, <8 x i32> %arg3, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32 immarg, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_load_3d_v4f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.3d.v4f32.i32(i32 noundef 15, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.load.3d.v4f32.i32(i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_load_cube_v4f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.cube.v4f32.i32(i32 noundef 15, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.load.cube.v4f32.i32(i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_load_1darray_v4f32_i32(i32 %arg1, i32 %arg2, <8 x i32> %arg3) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.1darray.v4f32.i32(i32 noundef 15, i32 %arg1, i32 %arg2, <8 x i32> %arg3, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.load.1darray.v4f32.i32(i32 immarg, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_load_2darray_v4f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.2darray.v4f32.i32(i32 noundef 15, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.load.2darray.v4f32.i32(i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_load_mip_1d_v4f32_i32(i32 %arg1, i32 %arg2, <8 x i32> %arg3) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.mip.1d.v4f32.i32(i32 noundef 15, i32 %arg1, i32 %arg2, <8 x i32> %arg3, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.load.mip.1d.v4f32.i32(i32 immarg, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_load_mip_2d_v4f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.mip.2d.v4f32.i32(i32 noundef 15, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.load.mip.2d.v4f32.i32(i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_load_mip_3d_v4f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.mip.3d.v4f32.i32(i32 noundef 15, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.load.mip.3d.v4f32.i32(i32 immarg, i32, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_load_mip_cube_v4f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.mip.cube.v4f32.i32(i32 noundef 15, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.load.mip.cube.v4f32.i32(i32 immarg, i32, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_load_mip_1darray_v4f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.mip.1darray.v4f32.i32(i32 noundef 15, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.load.mip.1darray.v4f32.i32(i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_load_mip_2darray_v4f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.mip.2darray.v4f32.i32(i32 noundef 15, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.load.mip.2darray.v4f32.i32(i32 immarg, i32, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_load_1d_v4f16_i32(i32 %arg1, <8 x i32> %arg2) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.1d.v4f16.i32(i32 noundef 15, i32 %arg1, <8 x i32> %arg2, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.load.1d.v4f16.i32(i32 immarg, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_load_2d_v4f16_i32(i32 %arg1, i32 %arg2, <8 x i32> %arg3) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.2d.v4f16.i32(i32 noundef 15, i32 %arg1, i32 %arg2, <8 x i32> %arg3, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.load.2d.v4f16.i32(i32 immarg, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_load_3d_v4f16_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.3d.v4f16.i32(i32 noundef 15, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.load.3d.v4f16.i32(i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_load_cube_v4f16_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.cube.v4f16.i32(i32 noundef 15, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.load.cube.v4f16.i32(i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_load_1darray_v4f16_i32(i32 %arg1, i32 %arg2, <8 x i32> %arg3) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.1darray.v4f16.i32(i32 noundef 15, i32 %arg1, i32 %arg2, <8 x i32> %arg3, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.load.1darray.v4f16.i32(i32 immarg, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_load_2darray_v4f16_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.2darray.v4f16.i32(i32 noundef 15, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.load.2darray.v4f16.i32(i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_load_mip_1d_v4f16_i32(i32 %arg1, i32 %arg2, <8 x i32> %arg3) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.mip.1d.v4f16.i32(i32 noundef 15, i32 %arg1, i32 %arg2, <8 x i32> %arg3, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.load.mip.1d.v4f16.i32(i32 immarg, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_load_mip_2d_v4f16_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.mip.2d.v4f16.i32(i32 noundef 15, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.load.mip.2d.v4f16.i32(i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_load_mip_3d_v4f16_i32(i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.mip.3d.v4f16.i32(i32 noundef 15, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.load.mip.3d.v4f16.i32(i32 immarg, i32, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_load_mip_cube_v4f16_i32(i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.mip.cube.v4f16.i32(i32 noundef 15, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.load.mip.cube.v4f16.i32(i32 immarg, i32, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_load_mip_1darray_v4f16_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.mip.1darray.v4f16.i32(i32 noundef 15, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.load.mip.1darray.v4f16.i32(i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_load_mip_2darray_v4f16_i32(i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.mip.2darray.v4f16.i32(i32 noundef 15, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.load.mip.2darray.v4f16.i32(i32 immarg, i32, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected float @__llvm_amdgcn_image_load_2d_f32_i32(i32 %arg1, i32 %arg2, <8 x i32> %arg3) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.load.2d.f32.i32(i32 noundef 1, i32 %arg1, i32 %arg2, <8 x i32> %arg3, i32 noundef 0, i32 noundef 0)
  ret float %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare float @llvm.amdgcn.image.load.2d.f32.i32(i32 immarg, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected float @__llvm_amdgcn_image_load_2darray_f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.load.2darray.f32.i32(i32 noundef 1, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 noundef 0, i32 noundef 0)
  ret float %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare float @llvm.amdgcn.image.load.2darray.f32.i32(i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected float @__llvm_amdgcn_image_load_mip_2d_f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.load.mip.2d.f32.i32(i32 noundef 1, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 noundef 0, i32 noundef 0)
  ret float %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare float @llvm.amdgcn.image.load.mip.2d.f32.i32(i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected float @__llvm_amdgcn_image_load_mip_2darray_f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.load.mip.2darray.f32.i32(i32 noundef 1, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 noundef 0, i32 noundef 0)
  ret float %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare float @llvm.amdgcn.image.load.mip.2darray.f32.i32(i32 immarg, i32, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_1d_v4f32_i32(<4 x float> %arg, i32 %arg2, <8 x i32> %arg3) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %arg, i32 noundef 15, i32 %arg2, <8 x i32> %arg3, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float>, i32 immarg, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_2d_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float>, i32 immarg, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_3d_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.3d.v4f32.i32(<4 x float> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.3d.v4f32.i32(<4 x float>, i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_cube_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.cube.v4f32.i32(<4 x float> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.cube.v4f32.i32(<4 x float>, i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_1darray_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.1darray.v4f32.i32(<4 x float> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.1darray.v4f32.i32(<4 x float>, i32 immarg, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_2darray_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.2darray.v4f32.i32(<4 x float> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.2darray.v4f32.i32(<4 x float>, i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_mip_1d_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.1d.v4f32.i32(<4 x float> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.mip.1d.v4f32.i32(<4 x float>, i32 immarg, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_mip_2d_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.2d.v4f32.i32(<4 x float> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.mip.2d.v4f32.i32(<4 x float>, i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_mip_3d_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.3d.v4f32.i32(<4 x float> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.mip.3d.v4f32.i32(<4 x float>, i32 immarg, i32, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_mip_cube_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.cube.v4f32.i32(<4 x float> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.mip.cube.v4f32.i32(<4 x float>, i32 immarg, i32, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_mip_1darray_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.1darray.v4f32.i32(<4 x float> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.mip.1darray.v4f32.i32(<4 x float>, i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_mip_2darray_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.2darray.v4f32.i32(<4 x float> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.mip.2darray.v4f32.i32(<4 x float>, i32 immarg, i32, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_1d_v4f16_i32(<4 x half> %arg, i32 %arg2, <8 x i32> %arg3) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.1d.v4f16.i32(<4 x half> %arg, i32 noundef 15, i32 %arg2, <8 x i32> %arg3, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.1d.v4f16.i32(<4 x half>, i32 immarg, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_2d_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.2d.v4f16.i32(<4 x half> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.2d.v4f16.i32(<4 x half>, i32 immarg, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_3d_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.3d.v4f16.i32(<4 x half> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.3d.v4f16.i32(<4 x half>, i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_cube_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.cube.v4f16.i32(<4 x half> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.cube.v4f16.i32(<4 x half>, i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_1darray_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.1darray.v4f16.i32(<4 x half> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.1darray.v4f16.i32(<4 x half>, i32 immarg, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_2darray_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.2darray.v4f16.i32(<4 x half> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.2darray.v4f16.i32(<4 x half>, i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_mip_1d_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.1d.v4f16.i32(<4 x half> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.mip.1d.v4f16.i32(<4 x half>, i32 immarg, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_mip_2d_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.2d.v4f16.i32(<4 x half> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.mip.2d.v4f16.i32(<4 x half>, i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_mip_3d_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.3d.v4f16.i32(<4 x half> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.mip.3d.v4f16.i32(<4 x half>, i32 immarg, i32, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_mip_cube_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.cube.v4f16.i32(<4 x half> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.mip.cube.v4f16.i32(<4 x half>, i32 immarg, i32, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_mip_1darray_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.1darray.v4f16.i32(<4 x half> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.mip.1darray.v4f16.i32(<4 x half>, i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_mip_2darray_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.2darray.v4f16.i32(<4 x half> %arg, i32 noundef 15, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.mip.2darray.v4f16.i32(<4 x half>, i32 immarg, i32, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_2d_f32_i32(float %arg, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.2d.f32.i32(float %arg, i32 noundef 15, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.2d.f32.i32(float, i32 immarg, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_2darray_f32_i32(float %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.2darray.f32.i32(float %arg, i32 noundef 1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.2darray.f32.i32(float, i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_mip_2d_f32_i32(float %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.2d.f32.i32(float %arg, i32 noundef 1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.mip.2d.f32.i32(float, i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(write)
define protected void @__llvm_amdgcn_image_store_mip_2darray_f32_i32(float %arg, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.2darray.f32.i32(float %arg, i32 noundef 1, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6, i32 noundef 0, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.amdgcn.image.store.mip.2darray.f32.i32(float, i32 immarg, i32, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_1d_v4f32_f32(float %arg1, <8 x i32> %arg2, <4 x i32> %arg3) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 noundef 15, float %arg1, <8 x i32> %arg2, <4 x i32> %arg3, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 immarg, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_2d_v4f32_f32(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32 noundef 15, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_3d_v4f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.3d.v4f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.3d.v4f32.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_cube_v4f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.cube.v4f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.cube.v4f32.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_1darray_v4f32_f32(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.1darray.v4f32.f32(i32 noundef 15, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.1darray.v4f32.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_2darray_v4f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.2darray.v4f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.2darray.v4f32.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_1d_v4f16_f32(float %arg1, <8 x i32> %arg2, <4 x i32> %arg3) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.1d.v4f16.f32(i32 noundef 15, float %arg1, <8 x i32> %arg2, <4 x i32> %arg3, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.1d.v4f16.f32(i32 immarg, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_2d_v4f16_f32(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.2d.v4f16.f32(i32 noundef 15, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.2d.v4f16.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_3d_v4f16_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.3d.v4f16.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.3d.v4f16.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_cube_v4f16_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.cube.v4f16.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.cube.v4f16.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_1darray_v4f16_f32(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.1darray.v4f16.f32(i32 noundef 15, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.1darray.v4f16.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_2darray_v4f16_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.2darray.v4f16.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.2darray.v4f16.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected float @__llvm_amdgcn_image_sample_2d_f32_f32(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.sample.2d.f32.f32(i32 noundef 1, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret float %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare float @llvm.amdgcn.image.sample.2d.f32.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected float @__llvm_amdgcn_image_sample_2darray_f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.sample.2darray.f32.f32(i32 noundef 1, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret float %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare float @llvm.amdgcn.image.sample.2darray.f32.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

attributes #0 = { nofree norecurse nosync nounwind willreturn memory(read) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(read) }
attributes #2 = { nofree norecurse nosync nounwind willreturn memory(write) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(write) }
