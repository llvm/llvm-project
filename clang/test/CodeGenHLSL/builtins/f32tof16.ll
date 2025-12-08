; ModuleID = 'f32tof16.hlsl'
source_filename = "f32tof16.hlsl"
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64-v48:16:16-v96:32:32-v192:64:64"
target triple = "dxilv1.3-pc-shadermodel6.3-library"

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define hidden noundef i32 @_Z11test_scalarf(float noundef nofpclass(nan inf) %p0) #0 {
entry:
  %p0.addr = alloca float, align 4
  store float %p0, ptr %p0.addr, align 4
  %0 = load float, ptr %p0.addr, align 4
  %hlsl.f32tof16 = call i32 @llvm.dx.legacyf32tof16.f32(float %0)
  ret i32 %hlsl.f32tof16
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.dx.legacyf32tof16.f32(float) #1

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define hidden noundef <2 x i32> @_Z10test_uint2Dv2_f(<2 x float> noundef nofpclass(nan inf) %p0) #0 {
entry:
  %p0.addr = alloca <2 x float>, align 8
  store <2 x float> %p0, ptr %p0.addr, align 8
  %0 = load <2 x float>, ptr %p0.addr, align 8
  %hlsl.f32tof16 = call <2 x i32> @llvm.dx.legacyf32tof16.v2f32(<2 x float> %0)
  ret <2 x i32> %hlsl.f32tof16
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <2 x i32> @llvm.dx.legacyf32tof16.v2f32(<2 x float>) #1

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define hidden noundef <3 x i32> @_Z10test_uint3Dv3_f(<3 x float> noundef nofpclass(nan inf) %p0) #0 {
entry:
  %p0.addr = alloca <3 x float>, align 16
  store <3 x float> %p0, ptr %p0.addr, align 16
  %0 = load <3 x float>, ptr %p0.addr, align 16
  %hlsl.f32tof16 = call <3 x i32> @llvm.dx.legacyf32tof16.v3f32(<3 x float> %0)
  ret <3 x i32> %hlsl.f32tof16
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <3 x i32> @llvm.dx.legacyf32tof16.v3f32(<3 x float>) #1

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define hidden noundef <4 x i32> @_Z10test_uint4Dv4_f(<4 x float> noundef nofpclass(nan inf) %p0) #0 {
entry:
  %p0.addr = alloca <4 x float>, align 16
  store <4 x float> %p0, ptr %p0.addr, align 16
  %0 = load <4 x float>, ptr %p0.addr, align 16
  %hlsl.f32tof16 = call <4 x i32> @llvm.dx.legacyf32tof16.v4f32(<4 x float> %0)
  ret <4 x i32> %hlsl.f32tof16
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <4 x i32> @llvm.dx.legacyf32tof16.v4f32(<4 x float>) #1

attributes #0 = { alwaysinline convergent mustprogress norecurse nounwind "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0}
!dx.valver = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 8}
!2 = !{!"clang version 22.0.0git (https://github.com/llvm/llvm-project.git a5e8e77f7ccd15945eb432a3619e57f9600c142a)"}
