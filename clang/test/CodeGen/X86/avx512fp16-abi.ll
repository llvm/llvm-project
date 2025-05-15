; ModuleID = '/home/vorrtt3x/dev/llvm-project/clang/test/CodeGen/X86/avx512fp16-abi.c'
source_filename = "/home/vorrtt3x/dev/llvm-project/clang/test/CodeGen/X86/avx512fp16-abi.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux"

%struct.half1 = type { half }
%struct.half2 = type { half, half }
%struct.half3 = type { half, half, half }
%struct.half4 = type { half, half, half, half }
%struct.floathalf = type { float, half }
%struct.floathalf2 = type { float, half, half }
%struct.halffloat = type { half, float }
%struct.half2float = type { half, half, float }
%struct.floathalf3 = type { float, half, half, half }
%struct.half5 = type { half, half, half, half, half }
%struct.float2 = type { [4 x i8], float, float }
%struct.float3 = type { float, [4 x i8], float }
%struct.shalf2 = type { [2 x i8], half, half }
%struct.fsd = type { float, double }
%struct.hsd = type { half, double }
%struct.hsf = type { half, float }
%struct.fds = type { float, double, [8 x i8] }

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local half @_Z2h1DF16_(half noundef %a) #0 {
entry:
  %retval = alloca %struct.half1, align 2
  %a.addr = alloca half, align 2
  store half %a, ptr %a.addr, align 2
  %0 = load half, ptr %a.addr, align 2
  %a1 = getelementptr inbounds nuw %struct.half1, ptr %retval, i32 0, i32 0
  store half %0, ptr %a1, align 2
  %coerce.dive = getelementptr inbounds nuw %struct.half1, ptr %retval, i32 0, i32 0
  %1 = load half, ptr %coerce.dive, align 2
  ret half %1
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local <2 x half> @_Z2h2DF16_DF16_(half noundef %a, half noundef %b) #1 {
entry:
  %retval = alloca %struct.half2, align 2
  %a.addr = alloca half, align 2
  %b.addr = alloca half, align 2
  store half %a, ptr %a.addr, align 2
  store half %b, ptr %b.addr, align 2
  %0 = load half, ptr %a.addr, align 2
  %a1 = getelementptr inbounds nuw %struct.half2, ptr %retval, i32 0, i32 0
  store half %0, ptr %a1, align 2
  %1 = load half, ptr %b.addr, align 2
  %b2 = getelementptr inbounds nuw %struct.half2, ptr %retval, i32 0, i32 1
  store half %1, ptr %b2, align 2
  %2 = load <2 x half>, ptr %retval, align 2
  ret <2 x half> %2
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local <4 x half> @_Z2h3DF16_DF16_DF16_(half noundef %a, half noundef %b, half noundef %c) #2 {
entry:
  %retval = alloca %struct.half3, align 2
  %a.addr = alloca half, align 2
  %b.addr = alloca half, align 2
  %c.addr = alloca half, align 2
  %retval.coerce = alloca <4 x half>, align 8
  store half %a, ptr %a.addr, align 2
  store half %b, ptr %b.addr, align 2
  store half %c, ptr %c.addr, align 2
  %0 = load half, ptr %a.addr, align 2
  %a1 = getelementptr inbounds nuw %struct.half3, ptr %retval, i32 0, i32 0
  store half %0, ptr %a1, align 2
  %1 = load half, ptr %b.addr, align 2
  %b2 = getelementptr inbounds nuw %struct.half3, ptr %retval, i32 0, i32 1
  store half %1, ptr %b2, align 2
  %2 = load half, ptr %c.addr, align 2
  %c3 = getelementptr inbounds nuw %struct.half3, ptr %retval, i32 0, i32 2
  store half %2, ptr %c3, align 2
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %retval.coerce, ptr align 2 %retval, i64 6, i1 false)
  %3 = load <4 x half>, ptr %retval.coerce, align 8
  ret <4 x half> %3
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local <4 x half> @_Z2h4DF16_DF16_DF16_DF16_(half noundef %a, half noundef %b, half noundef %c, half noundef %d) #2 {
entry:
  %retval = alloca %struct.half4, align 2
  %a.addr = alloca half, align 2
  %b.addr = alloca half, align 2
  %c.addr = alloca half, align 2
  %d.addr = alloca half, align 2
  store half %a, ptr %a.addr, align 2
  store half %b, ptr %b.addr, align 2
  store half %c, ptr %c.addr, align 2
  store half %d, ptr %d.addr, align 2
  %0 = load half, ptr %a.addr, align 2
  %a1 = getelementptr inbounds nuw %struct.half4, ptr %retval, i32 0, i32 0
  store half %0, ptr %a1, align 2
  %1 = load half, ptr %b.addr, align 2
  %b2 = getelementptr inbounds nuw %struct.half4, ptr %retval, i32 0, i32 1
  store half %1, ptr %b2, align 2
  %2 = load half, ptr %c.addr, align 2
  %c3 = getelementptr inbounds nuw %struct.half4, ptr %retval, i32 0, i32 2
  store half %2, ptr %c3, align 2
  %3 = load half, ptr %d.addr, align 2
  %d4 = getelementptr inbounds nuw %struct.half4, ptr %retval, i32 0, i32 3
  store half %3, ptr %d4, align 2
  %4 = load <4 x half>, ptr %retval, align 2
  ret <4 x half> %4
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local <4 x half> @_Z2fhfDF16_(float noundef %a, half noundef %b) #2 {
entry:
  %retval = alloca %struct.floathalf, align 4
  %a.addr = alloca float, align 4
  %b.addr = alloca half, align 2
  store float %a, ptr %a.addr, align 4
  store half %b, ptr %b.addr, align 2
  %0 = load float, ptr %a.addr, align 4
  %a1 = getelementptr inbounds nuw %struct.floathalf, ptr %retval, i32 0, i32 0
  store float %0, ptr %a1, align 4
  %1 = load half, ptr %b.addr, align 2
  %b2 = getelementptr inbounds nuw %struct.floathalf, ptr %retval, i32 0, i32 1
  store half %1, ptr %b2, align 4
  %2 = load <4 x half>, ptr %retval, align 4
  ret <4 x half> %2
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local <4 x half> @_Z3fh2fDF16_DF16_(float noundef %a, half noundef %b, half noundef %c) #2 {
entry:
  %retval = alloca %struct.floathalf2, align 4
  %a.addr = alloca float, align 4
  %b.addr = alloca half, align 2
  %c.addr = alloca half, align 2
  store float %a, ptr %a.addr, align 4
  store half %b, ptr %b.addr, align 2
  store half %c, ptr %c.addr, align 2
  %0 = load float, ptr %a.addr, align 4
  %a1 = getelementptr inbounds nuw %struct.floathalf2, ptr %retval, i32 0, i32 0
  store float %0, ptr %a1, align 4
  %1 = load half, ptr %b.addr, align 2
  %b2 = getelementptr inbounds nuw %struct.floathalf2, ptr %retval, i32 0, i32 1
  store half %1, ptr %b2, align 4
  %2 = load half, ptr %c.addr, align 2
  %c3 = getelementptr inbounds nuw %struct.floathalf2, ptr %retval, i32 0, i32 2
  store half %2, ptr %c3, align 2
  %3 = load <4 x half>, ptr %retval, align 4
  ret <4 x half> %3
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local <4 x half> @_Z2hfDF16_f(half noundef %a, float noundef %b) #2 {
entry:
  %retval = alloca %struct.halffloat, align 4
  %a.addr = alloca half, align 2
  %b.addr = alloca float, align 4
  store half %a, ptr %a.addr, align 2
  store float %b, ptr %b.addr, align 4
  %0 = load half, ptr %a.addr, align 2
  %a1 = getelementptr inbounds nuw %struct.halffloat, ptr %retval, i32 0, i32 0
  store half %0, ptr %a1, align 4
  %1 = load float, ptr %b.addr, align 4
  %b2 = getelementptr inbounds nuw %struct.halffloat, ptr %retval, i32 0, i32 1
  store float %1, ptr %b2, align 4
  %2 = load <4 x half>, ptr %retval, align 4
  ret <4 x half> %2
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local <4 x half> @_Z3h2fDF16_DF16_f(half noundef %a, half noundef %b, float noundef %c) #2 {
entry:
  %retval = alloca %struct.half2float, align 4
  %a.addr = alloca half, align 2
  %b.addr = alloca half, align 2
  %c.addr = alloca float, align 4
  store half %a, ptr %a.addr, align 2
  store half %b, ptr %b.addr, align 2
  store float %c, ptr %c.addr, align 4
  %0 = load half, ptr %a.addr, align 2
  %a1 = getelementptr inbounds nuw %struct.half2float, ptr %retval, i32 0, i32 0
  store half %0, ptr %a1, align 4
  %1 = load half, ptr %b.addr, align 2
  %b2 = getelementptr inbounds nuw %struct.half2float, ptr %retval, i32 0, i32 1
  store half %1, ptr %b2, align 2
  %2 = load float, ptr %c.addr, align 4
  %c3 = getelementptr inbounds nuw %struct.half2float, ptr %retval, i32 0, i32 2
  store float %2, ptr %c3, align 4
  %3 = load <4 x half>, ptr %retval, align 4
  ret <4 x half> %3
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local { <4 x half>, half } @_Z3fh3fDF16_DF16_DF16_(float noundef %a, half noundef %b, half noundef %c, half noundef %d) #0 {
entry:
  %retval = alloca %struct.floathalf3, align 4
  %a.addr = alloca float, align 4
  %b.addr = alloca half, align 2
  %c.addr = alloca half, align 2
  %d.addr = alloca half, align 2
  %retval.coerce = alloca { <4 x half>, half }, align 8
  store float %a, ptr %a.addr, align 4
  store half %b, ptr %b.addr, align 2
  store half %c, ptr %c.addr, align 2
  store half %d, ptr %d.addr, align 2
  %0 = load float, ptr %a.addr, align 4
  %a1 = getelementptr inbounds nuw %struct.floathalf3, ptr %retval, i32 0, i32 0
  store float %0, ptr %a1, align 4
  %1 = load half, ptr %b.addr, align 2
  %b2 = getelementptr inbounds nuw %struct.floathalf3, ptr %retval, i32 0, i32 1
  store half %1, ptr %b2, align 4
  %2 = load half, ptr %c.addr, align 2
  %c3 = getelementptr inbounds nuw %struct.floathalf3, ptr %retval, i32 0, i32 2
  store half %2, ptr %c3, align 2
  %3 = load half, ptr %d.addr, align 2
  %d4 = getelementptr inbounds nuw %struct.floathalf3, ptr %retval, i32 0, i32 3
  store half %3, ptr %d4, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %retval.coerce, ptr align 4 %retval, i64 12, i1 false)
  %4 = load { <4 x half>, half }, ptr %retval.coerce, align 8
  ret { <4 x half>, half } %4
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local { <4 x half>, half } @_Z2h5DF16_DF16_DF16_DF16_DF16_(half noundef %a, half noundef %b, half noundef %c, half noundef %d, half noundef %e) #0 {
entry:
  %retval = alloca %struct.half5, align 2
  %a.addr = alloca half, align 2
  %b.addr = alloca half, align 2
  %c.addr = alloca half, align 2
  %d.addr = alloca half, align 2
  %e.addr = alloca half, align 2
  %retval.coerce = alloca { <4 x half>, half }, align 8
  store half %a, ptr %a.addr, align 2
  store half %b, ptr %b.addr, align 2
  store half %c, ptr %c.addr, align 2
  store half %d, ptr %d.addr, align 2
  store half %e, ptr %e.addr, align 2
  %0 = load half, ptr %a.addr, align 2
  %a1 = getelementptr inbounds nuw %struct.half5, ptr %retval, i32 0, i32 0
  store half %0, ptr %a1, align 2
  %1 = load half, ptr %b.addr, align 2
  %b2 = getelementptr inbounds nuw %struct.half5, ptr %retval, i32 0, i32 1
  store half %1, ptr %b2, align 2
  %2 = load half, ptr %c.addr, align 2
  %c3 = getelementptr inbounds nuw %struct.half5, ptr %retval, i32 0, i32 2
  store half %2, ptr %c3, align 2
  %3 = load half, ptr %d.addr, align 2
  %d4 = getelementptr inbounds nuw %struct.half5, ptr %retval, i32 0, i32 3
  store half %3, ptr %d4, align 2
  %4 = load half, ptr %e.addr, align 2
  %e5 = getelementptr inbounds nuw %struct.half5, ptr %retval, i32 0, i32 4
  store half %4, ptr %e5, align 2
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %retval.coerce, ptr align 2 %retval, i64 10, i1 false)
  %5 = load { <4 x half>, half }, ptr %retval.coerce, align 8
  ret { <4 x half>, half } %5
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local noundef float @_Z7pr518136float2(double %s.coerce0, float %s.coerce1) #0 {
entry:
  %s = alloca %struct.float2, align 4
  %coerce = alloca { double, float }, align 4
  %0 = getelementptr inbounds nuw { double, float }, ptr %coerce, i32 0, i32 0
  store double %s.coerce0, ptr %0, align 4
  %1 = getelementptr inbounds nuw { double, float }, ptr %coerce, i32 0, i32 1
  store float %s.coerce1, ptr %1, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %s, ptr align 4 %coerce, i64 12, i1 false)
  %a = getelementptr inbounds nuw %struct.float2, ptr %s, i32 0, i32 1
  %2 = load float, ptr %a, align 4
  ret float %2
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local noundef float @_Z9pr51813_26float3(double %s.coerce0, float %s.coerce1) #0 {
entry:
  %s = alloca %struct.float3, align 4
  %coerce = alloca { double, float }, align 4
  %0 = getelementptr inbounds nuw { double, float }, ptr %coerce, i32 0, i32 0
  store double %s.coerce0, ptr %0, align 4
  %1 = getelementptr inbounds nuw { double, float }, ptr %coerce, i32 0, i32 1
  store float %s.coerce1, ptr %1, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %s, ptr align 4 %coerce, i64 12, i1 false)
  %a = getelementptr inbounds nuw %struct.float3, ptr %s, i32 0, i32 0
  %2 = load float, ptr %a, align 4
  ret float %2
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local noundef half @_Z3sf26shalf2(double %s.coerce) #0 {
entry:
  %s = alloca %struct.shalf2, align 2
  %tmp.coerce = alloca double, align 8
  store double %s.coerce, ptr %tmp.coerce, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %s, ptr align 8 %tmp.coerce, i64 6, i1 false)
  %a = getelementptr inbounds nuw %struct.shalf2, ptr %s, i32 0, i32 1
  %0 = load half, ptr %a, align 2
  ret half %0
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local noundef half @_Z3fs26shalf2(double %s.coerce) #0 {
entry:
  %s = alloca %struct.shalf2, align 2
  %tmp.coerce = alloca double, align 8
  store double %s.coerce, ptr %tmp.coerce, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %s, ptr align 8 %tmp.coerce, i64 6, i1 false)
  %a = getelementptr inbounds nuw %struct.shalf2, ptr %s, i32 0, i32 1
  %0 = load half, ptr %a, align 2
  ret half %0
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local { float, double } @_Z7pr52011v() #0 {
entry:
  %retval = alloca %struct.fsd, align 8
  %0 = load { float, double }, ptr %retval, align 8
  ret { float, double } %0
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local { half, double } @_Z9pr52011_2v() #0 {
entry:
  %retval = alloca %struct.hsd, align 8
  %0 = load { half, double }, ptr %retval, align 8
  ret { half, double } %0
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local <4 x half> @_Z9pr52011_3v() #2 {
entry:
  %retval = alloca %struct.hsf, align 4
  %0 = load <4 x half>, ptr %retval, align 4
  ret <4 x half> %0
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z9pr52011_4v(ptr dead_on_unwind noalias writable sret(%struct.fds) align 8 %agg.result) #0 {
entry:
  ret void
}

attributes #0 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+avx,+avx2,+avx512bw,+avx512f,+avx512fp16,+crc32,+cx8,+evex512,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" }
attributes #1 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="32" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+avx,+avx2,+avx512bw,+avx512f,+avx512fp16,+crc32,+cx8,+evex512,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" }
attributes #2 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="64" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+avx,+avx2,+avx512bw,+avx512f,+avx512fp16,+crc32,+cx8,+evex512,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 22.0.0git (git@github.com:vortex73/llvm-project.git 4f3a663fc730ec8750c2ae810b557e1c833f8007)"}
