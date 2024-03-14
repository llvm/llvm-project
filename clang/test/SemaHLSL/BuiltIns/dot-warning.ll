; ModuleID = 'D:\projects\llvm-project\clang\test\SemaHLSL\BuiltIns\dot-warning.hlsl'
source_filename = "D:\\projects\\llvm-project\\clang\\test\\SemaHLSL\\BuiltIns\\dot-warning.hlsl"
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-pc-shadermodel6.3-library"

; Function Attrs: noinline nounwind optnone
define noundef float @"?test_dot_builtin_vector_elem_size_reduction@@YAMT?$__vector@J$01@__clang@@M@Z"(<2 x i64> noundef %p0, float noundef %p1) #0 {
entry:
  %p1.addr = alloca float, align 4
  %p0.addr = alloca <2 x i64>, align 16
  store float %p1, ptr %p1.addr, align 4
  store <2 x i64> %p0, ptr %p0.addr, align 16
  %0 = load <2 x i64>, ptr %p0.addr, align 16
  %conv = sitofp <2 x i64> %0 to <2 x float>
  %1 = load float, ptr %p1.addr, align 4
  %splat.splatinsert = insertelement <2 x float> poison, float %1, i64 0
  %splat.splat = shufflevector <2 x float> %splat.splatinsert, <2 x float> poison, <2 x i32> zeroinitializer
  %dx.dot = call float @llvm.dx.dot.v2f32(<2 x float> %conv, <2 x float> %splat.splat)
  ret float %dx.dot
}

; Function Attrs: nounwind willreturn memory(none)
declare float @llvm.dx.dot.v2f32(<2 x float>, <2 x float>) #1

; Function Attrs: noinline nounwind optnone
define noundef float @"?test_dot_builtin_int_vector_elem_size_reduction@@YAMT?$__vector@H$01@__clang@@M@Z"(<2 x i32> noundef %p0, float noundef %p1) #0 {
entry:
  %p1.addr = alloca float, align 4
  %p0.addr = alloca <2 x i32>, align 8
  store float %p1, ptr %p1.addr, align 4
  store <2 x i32> %p0, ptr %p0.addr, align 8
  %0 = load <2 x i32>, ptr %p0.addr, align 8
  %conv = sitofp <2 x i32> %0 to <2 x float>
  %1 = load float, ptr %p1.addr, align 4
  %splat.splatinsert = insertelement <2 x float> poison, float %1, i64 0
  %splat.splat = shufflevector <2 x float> %splat.splatinsert, <2 x float> poison, <2 x i32> zeroinitializer
  %dx.dot = call float @llvm.dx.dot.v2f32(<2 x float> %conv, <2 x float> %splat.splat)
  ret float %dx.dot
}

attributes #0 = { noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{!"clang version 19.0.0git (https://github.com/farzonl/llvm-project.git f40562c7b4224e00da2ff2e13d175abfaac68532)"}
