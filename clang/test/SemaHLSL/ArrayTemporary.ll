; ModuleID = '/Users/cbieneman/dev/llvm-project/clang/test/SemaHLSL/ArrayTemporary.hlsl'
source_filename = "/Users/cbieneman/dev/llvm-project/clang/test/SemaHLSL/ArrayTemporary.hlsl"
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-pc-shadermodel6.3-library"

%struct.Obj = type { float, i32 }

@"__const.?call3@@YAXXZ.Arr" = private unnamed_addr constant [2 x [2 x float]] [[2 x float] zeroinitializer, [2 x float] [float 1.000000e+00, float 1.000000e+00]], align 4

; Function Attrs: noinline nounwind optnone
define void @"?fn@@YAXY01M@Z"(ptr noundef byval([2 x float]) align 4 %x) #0 {
entry:
  ret void
}

; Function Attrs: noinline nounwind optnone
define void @"?call@@YAXXZ"() #0 {
entry:
  %Arr = alloca [2 x float], align 4
  %agg.tmp = alloca [2 x float], align 4
  call void @llvm.memset.p0.i32(ptr align 4 %Arr, i8 0, i32 8, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %agg.tmp, ptr align 4 %Arr, i32 8, i1 false)
  call void @"?fn@@YAXY01M@Z"(ptr noundef byval([2 x float]) align 4 %agg.tmp)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1 immarg) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i32(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg) #2

; Function Attrs: noinline nounwind optnone
define void @"?fn2@@YAXY03UObj@@@Z"(ptr noundef byval([4 x %struct.Obj]) align 4 %O) #0 {
entry:
  ret void
}

; Function Attrs: noinline nounwind optnone
define void @"?call2@@YAXXZ"() #0 {
entry:
  %Arr = alloca [4 x %struct.Obj], align 4
  %agg.tmp = alloca [4 x %struct.Obj], align 4
  call void @llvm.memset.p0.i32(ptr align 4 %Arr, i8 0, i32 32, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %agg.tmp, ptr align 4 %Arr, i32 32, i1 false)
  call void @"?fn2@@YAXY03UObj@@@Z"(ptr noundef byval([4 x %struct.Obj]) align 4 %agg.tmp)
  ret void
}

; Function Attrs: noinline nounwind optnone
define void @"?fn3@@YAXY111M@Z"(ptr noundef byval([2 x [2 x float]]) align 4 %x) #0 {
entry:
  ret void
}

; Function Attrs: noinline nounwind optnone
define void @"?call3@@YAXXZ"() #0 {
entry:
  %Arr = alloca [2 x [2 x float]], align 4
  %agg.tmp = alloca [2 x [2 x float]], align 4
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %Arr, ptr align 4 @"__const.?call3@@YAXXZ.Arr", i32 16, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %agg.tmp, ptr align 4 %Arr, i32 16, i1 false)
  call void @"?fn3@@YAXY111M@Z"(ptr noundef byval([2 x [2 x float]]) align 4 %agg.tmp)
  ret void
}

attributes #0 = { noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{!"clang version 19.0.0git (git@github.com:llvm/llvm-project.git 64e1c15c520cf11114ef2ddd887e76560903db2b)"}
