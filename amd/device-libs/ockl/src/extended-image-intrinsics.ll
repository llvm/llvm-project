target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8"
target triple = "amdgcn-amd-amdhsa"

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_lz_1d_v4f32_f32(float %arg1, <8 x i32> %arg2, <4 x i32> %arg3) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.lz.1d.v4f32.f32(i32 noundef 15, float %arg1, <8 x i32> %arg2, <4 x i32> %arg3, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.lz.1d.v4f32.f32(i32 immarg, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_l_1d_v4f32_f32(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.l.1d.v4f32.f32(i32 noundef 15, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.l.1d.v4f32.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_d_1d_v4f32_f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.d.1d.v4f32.f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.d.1d.v4f32.f32.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_lz_2d_v4f32_f32(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f32(i32 noundef 15, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_l_2d_v4f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.l.2d.v4f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.l.2d.v4f32.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_d_2d_v4f32_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, <8 x i32> %arg7, <4 x i32> %arg8) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.d.2d.v4f32.f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, <8 x i32> %arg7, <4 x i32> %arg8, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.d.2d.v4f32.f32.f32(i32 immarg, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_lz_3d_v4f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.lz.3d.v4f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.lz.3d.v4f32.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_l_3d_v4f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.l.3d.v4f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.l.3d.v4f32.f32(i32 immarg, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_d_3d_v4f32_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, float %arg8, float %arg9, <8 x i32> %arg10, <4 x i32> %arg11) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.d.3d.v4f32.f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, float %arg8, float %arg9, <8 x i32> %arg10, <4 x i32> %arg11, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.d.3d.v4f32.f32.f32(i32 immarg, float, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_lz_cube_v4f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.lz.cube.v4f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.lz.cube.v4f32.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_l_cube_v4f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.l.cube.v4f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.l.cube.v4f32.f32(i32 immarg, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_lz_1darray_v4f32_f32(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.lz.1darray.v4f32.f32(i32 noundef 15, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.lz.1darray.v4f32.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_l_1darray_v4f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.l.1darray.v4f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.l.1darray.v4f32.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_d_1darray_v4f32_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.d.1darray.v4f32.f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.d.1darray.v4f32.f32.f32(i32 immarg, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_lz_2darray_v4f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.lz.2darray.v4f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.lz.2darray.v4f32.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_l_2darray_v4f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.l.2darray.v4f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.l.2darray.v4f32.f32(i32 immarg, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_sample_d_2darray_v4f32_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, <8 x i32> %arg8, <4 x i32> %arg9) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.d.2darray.v4f32.f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, <8 x i32> %arg8, <4 x i32> %arg9, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.sample.d.2darray.v4f32.f32.f32(i32 immarg, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_lz_1d_v4f16_f32(float %arg1, <8 x i32> %arg2, <4 x i32> %arg3) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.lz.1d.v4f16.f32(i32 noundef 15, float %arg1, <8 x i32> %arg2, <4 x i32> %arg3, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.lz.1d.v4f16.f32(i32 immarg, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_l_1d_v4f16_f32(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.l.1d.v4f16.f32(i32 noundef 15, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.l.1d.v4f16.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_d_1d_v4f16_f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.d.1d.v4f16.f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.d.1d.v4f16.f32.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_lz_2d_v4f16_f32(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.lz.2d.v4f16.f32(i32 noundef 15, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.lz.2d.v4f16.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_l_2d_v4f16_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.l.2d.v4f16.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.l.2d.v4f16.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_d_2d_v4f16_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, <8 x i32> %arg7, <4 x i32> %arg8) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.d.2d.v4f16.f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, <8 x i32> %arg7, <4 x i32> %arg8, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.d.2d.v4f16.f32.f32(i32 immarg, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_lz_3d_v4f16_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.lz.3d.v4f16.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.lz.3d.v4f16.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_l_3d_v4f16_f32(float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.l.3d.v4f16.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.l.3d.v4f16.f32(i32 immarg, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_d_3d_v4f16_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, float %arg8, float %arg9, <8 x i32> %arg10, <4 x i32> %arg11, i32 %arg13, i32 %arg14) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.d.3d.v4f16.f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, float %arg8, float %arg9, <8 x i32> %arg10, <4 x i32> %arg11, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.d.3d.v4f16.f32.f32(i32 immarg, float, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_lz_cube_v4f16_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.lz.cube.v4f16.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.lz.cube.v4f16.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_l_cube_v4f16_f32(float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.l.cube.v4f16.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.l.cube.v4f16.f32(i32 immarg, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_lz_1darray_v4f16_f32(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.lz.1darray.v4f16.f32(i32 noundef 15, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.lz.1darray.v4f16.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_l_1darray_v4f16_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.l.1darray.v4f16.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.l.1darray.v4f16.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_d_1darray_v4f16_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.d.1darray.v4f16.f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.d.1darray.v4f16.f32.f32(i32 immarg, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_lz_2darray_v4f16_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.lz.2darray.v4f16.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.lz.2darray.v4f16.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_l_2darray_v4f16_f32(float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.l.2darray.v4f16.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.l.2darray.v4f16.f32(i32 immarg, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x half> @__llvm_amdgcn_image_sample_d_2darray_v4f16_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, <8 x i32> %arg8, <4 x i32> %arg9) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.d.2darray.v4f16.f32.f32(i32 noundef 15, float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, <8 x i32> %arg8, <4 x i32> %arg9, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x half> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x half> @llvm.amdgcn.image.sample.d.2darray.v4f16.f32.f32(i32 immarg, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected float @__llvm_amdgcn_image_sample_lz_2d_f32_f32(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.sample.lz.2d.f32.f32(i32 noundef 1, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret float %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare float @llvm.amdgcn.image.sample.lz.2d.f32.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected float @__llvm_amdgcn_image_sample_l_2d_f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.sample.l.2d.f32.f32(i32 noundef 1, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret float %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare float @llvm.amdgcn.image.sample.l.2d.f32.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected float @__llvm_amdgcn_image_sample_d_2d_f32_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, <8 x i32> %arg7, <4 x i32> %arg8) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.sample.d.2d.f32.f32.f32(i32 noundef 1, float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, <8 x i32> %arg7, <4 x i32> %arg8, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret float %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare float @llvm.amdgcn.image.sample.d.2d.f32.f32.f32(i32 immarg, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected float @__llvm_amdgcn_image_sample_lz_2darray_f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.sample.lz.2darray.f32.f32(i32 noundef 1, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret float %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare float @llvm.amdgcn.image.sample.lz.2darray.f32.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected float @__llvm_amdgcn_image_sample_l_2darray_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.sample.l.2darray.f32.f32(i32 noundef 1, float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret float %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare float @llvm.amdgcn.image.sample.l.2darray.f32.f32(i32 immarg, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected float @__llvm_amdgcn_image_sample_d_2darray_f32_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, <8 x i32> %arg8, <4 x i32> %arg9, i32 %arg11, i32 %arg12) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.sample.d.2darray.f32.f32.f32(i32 noundef 1, float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, <8 x i32> %arg8, <4 x i32> %arg9, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret float %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare float @llvm.amdgcn.image.sample.d.2darray.f32.f32.f32(i32 immarg, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_gather4_lz_2d_v4f32_f32_r(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f32(i32 noundef 1, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_gather4_lz_2d_v4f32_f32_g(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f32(i32 noundef 2, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_gather4_lz_2d_v4f32_f32_b(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f32(i32 noundef 4, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nofree norecurse nosync nounwind willreturn memory(read)
define protected <4 x float> @__llvm_amdgcn_image_gather4_lz_2d_v4f32_f32_a(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f32(i32 noundef 8, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 noundef false, i32 noundef 0, i32 noundef 0)
  ret <4 x float> %tmp
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

attributes #0 = { nofree norecurse nosync nounwind willreturn memory(read) "target-features"="+extended-image-insts" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(read) }
