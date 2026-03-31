; ModuleID = 'D:\llvm-project\clang\test\SemaHLSL\Resources\Local-Resources\expression_init.hlsl'
source_filename = "D:\\llvm-project\\clang\\test\\SemaHLSL\\Resources\\Local-Resources\\expression_init.hlsl"
target datalayout = "e-m:e-ve-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxilv1.6-pc-shadermodel6.6-compute"

%"class.hlsl::RWByteAddressBuffer" = type { target("dx.RawBuffer", i8, 1, 0) }

@_ZL5gBuf0 = internal global %"class.hlsl::RWByteAddressBuffer" poison, align 4
@.str = private unnamed_addr constant [6 x i8] c"gBuf0\00", align 1
@_ZL5gBuf1 = internal global %"class.hlsl::RWByteAddressBuffer" poison, align 4
@.str.2 = private unnamed_addr constant [6 x i8] c"gBuf1\00", align 1
@_ZL4gOut = internal global %"class.hlsl::RWByteAddressBuffer" poison, align 4
@.str.4 = private unnamed_addr constant [5 x i8] c"gOut\00", align 1

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define hidden noundef i32 @_Z7DoStoreN4hlsl19RWByteAddressBufferEjj(ptr noundef byval(%"class.hlsl::RWByteAddressBuffer") align 4 %buf, i32 noundef %offset, i32 noundef %value) #0 {
entry:
  %this.addr.i = alloca ptr, align 4
  %Index.addr.i = alloca i32, align 4
  %Value.addr.i = alloca i32, align 4
  %offset.addr = alloca i32, align 4
  %value.addr = alloca i32, align 4
  store i32 %offset, ptr %offset.addr, align 4
  store i32 %value, ptr %value.addr, align 4
  %0 = load i32, ptr %offset.addr, align 4
  %1 = load i32, ptr %value.addr, align 4
  store ptr %buf, ptr %this.addr.i, align 4
  store i32 %0, ptr %Index.addr.i, align 4
  store i32 %1, ptr %Value.addr.i, align 4
  %this1.i = load ptr, ptr %this.addr.i, align 4
  %2 = load i32, ptr %Value.addr.i, align 4
  %3 = load target("dx.RawBuffer", i8, 1, 0), ptr %this1.i, align 4
  %4 = load i32, ptr %Index.addr.i, align 4
  %5 = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %3, i32 %4)
  store i32 %2, ptr %5, align 4
  %6 = load i32, ptr %value.addr, align 4
  ret i32 %6
}

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define hidden noundef i32 @_Z19Pass_ExpressionInitj(i32 noundef %idx) #0 {
entry:
  %this.addr.i10 = alloca ptr, align 4
  %other.addr.i11 = alloca ptr, align 4
  %this.addr.i7 = alloca ptr, align 4
  %other.addr.i8 = alloca ptr, align 4
  %this.addr.i4 = alloca ptr, align 4
  %other.addr.i5 = alloca ptr, align 4
  %this.addr.i2 = alloca ptr, align 4
  %other.addr.i = alloca ptr, align 4
  %this.addr.i = alloca ptr, align 4
  %Index.addr.i = alloca i32, align 4
  %Value.addr.i = alloca i32, align 4
  %offset.addr.i = alloca i32, align 4
  %value.addr.i = alloca i32, align 4
  %agg.tmp1 = alloca %"class.hlsl::RWByteAddressBuffer", align 8
  %idx.addr = alloca i32, align 4
  %buf = alloca %"class.hlsl::RWByteAddressBuffer", align 4
  %agg.tmp = alloca %"class.hlsl::RWByteAddressBuffer", align 4
  store i32 %idx, ptr %idx.addr, align 4
  store ptr %buf, ptr %this.addr.i4, align 4
  store ptr @_ZL5gBuf0, ptr %other.addr.i5, align 4
  %this1.i6 = load ptr, ptr %this.addr.i4, align 4
  %0 = load ptr, ptr %other.addr.i5, align 4
  store ptr %this1.i6, ptr %this.addr.i7, align 4
  store ptr %0, ptr %other.addr.i8, align 4
  %this1.i9 = load ptr, ptr %this.addr.i7, align 4
  %1 = load ptr, ptr %other.addr.i8, align 4, !nonnull !3, !align !4
  %2 = load target("dx.RawBuffer", i8, 1, 0), ptr %1, align 4
  store target("dx.RawBuffer", i8, 1, 0) %2, ptr %this1.i9, align 4
  store ptr %agg.tmp, ptr %this.addr.i2, align 4
  store ptr %buf, ptr %other.addr.i, align 4
  %this1.i3 = load ptr, ptr %this.addr.i2, align 4
  %3 = load ptr, ptr %other.addr.i, align 4
  store ptr %this1.i3, ptr %this.addr.i10, align 4
  store ptr %3, ptr %other.addr.i11, align 4
  %this1.i12 = load ptr, ptr %this.addr.i10, align 4
  %4 = load ptr, ptr %other.addr.i11, align 4, !nonnull !3, !align !4
  %5 = load target("dx.RawBuffer", i8, 1, 0), ptr %4, align 4
  store target("dx.RawBuffer", i8, 1, 0) %5, ptr %this1.i12, align 4
  %6 = load i32, ptr %idx.addr, align 4
  %mul = mul i32 %6, 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.tmp1, ptr align 4 %agg.tmp, i64 4, i1 false)
  store i32 %mul, ptr %offset.addr.i, align 4
  store i32 3, ptr %value.addr.i, align 4
  %7 = load i32, ptr %offset.addr.i, align 4
  %8 = load i32, ptr %value.addr.i, align 4
  store ptr %agg.tmp1, ptr %this.addr.i, align 4
  store i32 %7, ptr %Index.addr.i, align 4
  store i32 %8, ptr %Value.addr.i, align 4
  %this1.i = load ptr, ptr %this.addr.i, align 4
  %9 = load i32, ptr %Value.addr.i, align 4
  %10 = load target("dx.RawBuffer", i8, 1, 0), ptr %this1.i, align 4
  %11 = load i32, ptr %Index.addr.i, align 4
  %12 = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %10, i32 %11)
  store i32 %9, ptr %12, align 4
  %13 = load i32, ptr %value.addr.i, align 4
  ret i32 %13
}

; Function Attrs: convergent noinline norecurse
define void @main() #1 {
entry:
  %this.addr.i19.i = alloca ptr, align 4
  %other.addr.i20.i = alloca ptr, align 4
  %this.addr.i16.i = alloca ptr, align 4
  %other.addr.i17.i = alloca ptr, align 4
  %this.addr.i13.i = alloca ptr, align 4
  %other.addr.i14.i = alloca ptr, align 4
  %this.addr.i10.i = alloca ptr, align 4
  %other.addr.i11.i = alloca ptr, align 4
  %this.addr.i7.i = alloca ptr, align 4
  %other.addr.i8.i = alloca ptr, align 4
  %this.addr.i.i4 = alloca ptr, align 4
  %other.addr.i.i5 = alloca ptr, align 4
  %registerNo.addr.i.i1.i = alloca i32, align 4
  %spaceNo.addr.i.i2.i = alloca i32, align 4
  %range.addr.i.i3.i = alloca i32, align 4
  %index.addr.i.i4.i = alloca i32, align 4
  %name.addr.i.i5.i = alloca ptr, align 4
  %tmp.i.i6.i = alloca %"class.hlsl::RWByteAddressBuffer", align 4
  %registerNo.addr.i.i.i = alloca i32, align 4
  %spaceNo.addr.i.i.i = alloca i32, align 4
  %range.addr.i.i.i = alloca i32, align 4
  %index.addr.i.i.i = alloca i32, align 4
  %name.addr.i.i.i = alloca ptr, align 4
  %tmp.i.i.i = alloca %"class.hlsl::RWByteAddressBuffer", align 4
  %registerNo.addr.i.i = alloca i32, align 4
  %spaceNo.addr.i.i = alloca i32, align 4
  %range.addr.i.i = alloca i32, align 4
  %index.addr.i.i = alloca i32, align 4
  %name.addr.i.i = alloca ptr, align 4
  %tmp.i.i = alloca %"class.hlsl::RWByteAddressBuffer", align 4
  %this.addr.i1 = alloca ptr, align 4
  %other.addr.i2 = alloca ptr, align 4
  %this.addr.i = alloca ptr, align 4
  %other.addr.i = alloca ptr, align 4
  %this.addr.i1.i = alloca ptr, align 4
  %other.addr.i2.i = alloca ptr, align 4
  %this.addr.i.i = alloca ptr, align 4
  %other.addr.i.i = alloca ptr, align 4
  %this.addr.i.i.i = alloca ptr, align 4
  %Index.addr.i.i.i = alloca i32, align 4
  %Value.addr.i.i.i = alloca i32, align 4
  %offset.addr.i.i.i = alloca i32, align 4
  %value.addr.i.i.i = alloca i32, align 4
  %agg.tmp1.i.i = alloca %"class.hlsl::RWByteAddressBuffer", align 8
  %idx.addr.i.i = alloca i32, align 4
  %buf.i.i = alloca %"class.hlsl::RWByteAddressBuffer", align 4
  %agg.tmp.i.i = alloca %"class.hlsl::RWByteAddressBuffer", align 4
  %tid.addr.i = alloca <3 x i32>, align 4
  %idx.i = alloca i32, align 4
  call void @llvm.experimental.noalias.scope.decl(metadata !5)
  store i32 0, ptr %registerNo.addr.i.i, align 4, !noalias !5
  store i32 0, ptr %spaceNo.addr.i.i, align 4, !noalias !5
  store i32 1, ptr %range.addr.i.i, align 4, !noalias !5
  store i32 0, ptr %index.addr.i.i, align 4, !noalias !5
  store ptr @.str, ptr %name.addr.i.i, align 4, !noalias !5
  %0 = load i32, ptr %registerNo.addr.i.i, align 4, !noalias !5
  %1 = load i32, ptr %spaceNo.addr.i.i, align 4, !noalias !5
  %2 = load i32, ptr %range.addr.i.i, align 4, !noalias !5
  %3 = load i32, ptr %index.addr.i.i, align 4, !noalias !5
  %4 = load ptr, ptr %name.addr.i.i, align 4, !noalias !5
  %5 = call target("dx.RawBuffer", i8, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(i32 %1, i32 %0, i32 %2, i32 %3, ptr %4)
  store target("dx.RawBuffer", i8, 1, 0) %5, ptr %tmp.i.i, align 4, !noalias !5
  store ptr @_ZL5gBuf0, ptr %this.addr.i10.i, align 4
  store ptr %tmp.i.i, ptr %other.addr.i11.i, align 4
  %this1.i12.i = load ptr, ptr %this.addr.i10.i, align 4
  %6 = load ptr, ptr %other.addr.i11.i, align 4
  store ptr %this1.i12.i, ptr %this.addr.i13.i, align 4
  store ptr %6, ptr %other.addr.i14.i, align 4
  %this1.i15.i = load ptr, ptr %this.addr.i13.i, align 4
  %7 = load ptr, ptr %other.addr.i14.i, align 4, !nonnull !3, !align !4
  %8 = load target("dx.RawBuffer", i8, 1, 0), ptr %7, align 4
  store target("dx.RawBuffer", i8, 1, 0) %8, ptr %this1.i15.i, align 4
  call void @llvm.experimental.noalias.scope.decl(metadata !8)
  store i32 1, ptr %registerNo.addr.i.i.i, align 4, !noalias !8
  store i32 0, ptr %spaceNo.addr.i.i.i, align 4, !noalias !8
  store i32 1, ptr %range.addr.i.i.i, align 4, !noalias !8
  store i32 0, ptr %index.addr.i.i.i, align 4, !noalias !8
  store ptr @.str.2, ptr %name.addr.i.i.i, align 4, !noalias !8
  %9 = load i32, ptr %registerNo.addr.i.i.i, align 4, !noalias !8
  %10 = load i32, ptr %spaceNo.addr.i.i.i, align 4, !noalias !8
  %11 = load i32, ptr %range.addr.i.i.i, align 4, !noalias !8
  %12 = load i32, ptr %index.addr.i.i.i, align 4, !noalias !8
  %13 = load ptr, ptr %name.addr.i.i.i, align 4, !noalias !8
  %14 = call target("dx.RawBuffer", i8, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(i32 %10, i32 %9, i32 %11, i32 %12, ptr %13)
  store target("dx.RawBuffer", i8, 1, 0) %14, ptr %tmp.i.i.i, align 4, !noalias !8
  store ptr @_ZL5gBuf1, ptr %this.addr.i7.i, align 4
  store ptr %tmp.i.i.i, ptr %other.addr.i8.i, align 4
  %this1.i9.i = load ptr, ptr %this.addr.i7.i, align 4
  %15 = load ptr, ptr %other.addr.i8.i, align 4
  store ptr %this1.i9.i, ptr %this.addr.i16.i, align 4
  store ptr %15, ptr %other.addr.i17.i, align 4
  %this1.i18.i = load ptr, ptr %this.addr.i16.i, align 4
  %16 = load ptr, ptr %other.addr.i17.i, align 4, !nonnull !3, !align !4
  %17 = load target("dx.RawBuffer", i8, 1, 0), ptr %16, align 4
  store target("dx.RawBuffer", i8, 1, 0) %17, ptr %this1.i18.i, align 4
  call void @llvm.experimental.noalias.scope.decl(metadata !11)
  store i32 3, ptr %registerNo.addr.i.i1.i, align 4, !noalias !11
  store i32 0, ptr %spaceNo.addr.i.i2.i, align 4, !noalias !11
  store i32 1, ptr %range.addr.i.i3.i, align 4, !noalias !11
  store i32 0, ptr %index.addr.i.i4.i, align 4, !noalias !11
  store ptr @.str.4, ptr %name.addr.i.i5.i, align 4, !noalias !11
  %18 = load i32, ptr %registerNo.addr.i.i1.i, align 4, !noalias !11
  %19 = load i32, ptr %spaceNo.addr.i.i2.i, align 4, !noalias !11
  %20 = load i32, ptr %range.addr.i.i3.i, align 4, !noalias !11
  %21 = load i32, ptr %index.addr.i.i4.i, align 4, !noalias !11
  %22 = load ptr, ptr %name.addr.i.i5.i, align 4, !noalias !11
  %23 = call target("dx.RawBuffer", i8, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(i32 %19, i32 %18, i32 %20, i32 %21, ptr %22)
  store target("dx.RawBuffer", i8, 1, 0) %23, ptr %tmp.i.i6.i, align 4, !noalias !11
  store ptr @_ZL4gOut, ptr %this.addr.i.i4, align 4
  store ptr %tmp.i.i6.i, ptr %other.addr.i.i5, align 4
  %this1.i.i6 = load ptr, ptr %this.addr.i.i4, align 4
  %24 = load ptr, ptr %other.addr.i.i5, align 4
  store ptr %this1.i.i6, ptr %this.addr.i19.i, align 4
  store ptr %24, ptr %other.addr.i20.i, align 4
  %this1.i21.i = load ptr, ptr %this.addr.i19.i, align 4
  %25 = load ptr, ptr %other.addr.i20.i, align 4, !nonnull !3, !align !4
  %26 = load target("dx.RawBuffer", i8, 1, 0), ptr %25, align 4
  store target("dx.RawBuffer", i8, 1, 0) %26, ptr %this1.i21.i, align 4
  %27 = call i32 @llvm.dx.thread.id(i32 0)
  %28 = insertelement <3 x i32> poison, i32 %27, i64 0
  %29 = call i32 @llvm.dx.thread.id(i32 1)
  %30 = insertelement <3 x i32> %28, i32 %29, i64 1
  %31 = call i32 @llvm.dx.thread.id(i32 2)
  %32 = insertelement <3 x i32> %30, i32 %31, i64 2
  store <3 x i32> %32, ptr %tid.addr.i, align 4
  %33 = load <3 x i32>, ptr %tid.addr.i, align 4
  %34 = extractelement <3 x i32> %33, i32 0
  %35 = load <3 x i32>, ptr %tid.addr.i, align 4
  %36 = extractelement <3 x i32> %35, i32 1
  %mul.i = mul i32 %36, 8
  %add.i = add i32 %34, %mul.i
  store i32 %add.i, ptr %idx.i, align 4
  %37 = load i32, ptr %idx.i, align 4
  store i32 %37, ptr %idx.addr.i.i, align 4
  store ptr %buf.i.i, ptr %this.addr.i1.i, align 4
  store ptr @_ZL5gBuf0, ptr %other.addr.i2.i, align 4
  %this1.i3.i = load ptr, ptr %this.addr.i1.i, align 4
  %38 = load ptr, ptr %other.addr.i2.i, align 4
  store ptr %this1.i3.i, ptr %this.addr.i1, align 4
  store ptr %38, ptr %other.addr.i2, align 4
  %this1.i3 = load ptr, ptr %this.addr.i1, align 4
  %39 = load ptr, ptr %other.addr.i2, align 4, !nonnull !3, !align !4
  %40 = load target("dx.RawBuffer", i8, 1, 0), ptr %39, align 4
  store target("dx.RawBuffer", i8, 1, 0) %40, ptr %this1.i3, align 4
  store ptr %agg.tmp.i.i, ptr %this.addr.i.i, align 4
  store ptr %buf.i.i, ptr %other.addr.i.i, align 4
  %this1.i.i = load ptr, ptr %this.addr.i.i, align 4
  %41 = load ptr, ptr %other.addr.i.i, align 4
  store ptr %this1.i.i, ptr %this.addr.i, align 4
  store ptr %41, ptr %other.addr.i, align 4
  %this1.i = load ptr, ptr %this.addr.i, align 4
  %42 = load ptr, ptr %other.addr.i, align 4, !nonnull !3, !align !4
  %43 = load target("dx.RawBuffer", i8, 1, 0), ptr %42, align 4
  store target("dx.RawBuffer", i8, 1, 0) %43, ptr %this1.i, align 4
  %44 = load i32, ptr %idx.addr.i.i, align 4
  %mul.i.i = mul i32 %44, 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.tmp1.i.i, ptr align 4 %agg.tmp.i.i, i64 4, i1 false)
  store i32 %mul.i.i, ptr %offset.addr.i.i.i, align 4
  store i32 3, ptr %value.addr.i.i.i, align 4
  %45 = load i32, ptr %offset.addr.i.i.i, align 4
  %46 = load i32, ptr %value.addr.i.i.i, align 4
  store ptr %agg.tmp1.i.i, ptr %this.addr.i.i.i, align 4
  store i32 %45, ptr %Index.addr.i.i.i, align 4
  store i32 %46, ptr %Value.addr.i.i.i, align 4
  %this1.i.i.i = load ptr, ptr %this.addr.i.i.i, align 4
  %47 = load i32, ptr %Value.addr.i.i.i, align 4
  %48 = load target("dx.RawBuffer", i8, 1, 0), ptr %this1.i.i.i, align 4
  %49 = load i32, ptr %Index.addr.i.i.i, align 4
  %50 = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %48, i32 %49)
  store i32 %47, ptr %50, align 4
  %51 = load i32, ptr %value.addr.i.i.i, align 4
  ret void
}

; Function Attrs: nounwind willreturn memory(none)
declare i32 @llvm.dx.thread.id(i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare target("dx.RawBuffer", i8, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(i32, i32, i32, i32, ptr) #3

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0), i32) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #5

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #6

attributes #0 = { alwaysinline convergent mustprogress norecurse nounwind "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent noinline norecurse "hlsl.numthreads"="8,8,1" "hlsl.shader"="compute" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nounwind willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #6 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!dx.valver = !{!0}
!llvm.module.flags = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, i32 8}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{!"clang version 23.0.0git (https://github.com/llvm/llvm-project.git b084fa7a5172e2502324f9df2723e22dad071e3a)"}
!3 = !{}
!4 = !{i64 4}
!5 = !{!6}
!6 = distinct !{!6, !7, !"_ZN4hlsl19RWByteAddressBuffer19__createFromBindingEjjijPKc: %agg.result"}
!7 = distinct !{!7, !"_ZN4hlsl19RWByteAddressBuffer19__createFromBindingEjjijPKc"}
!8 = !{!9}
!9 = distinct !{!9, !10, !"_ZN4hlsl19RWByteAddressBuffer19__createFromBindingEjjijPKc: %agg.result"}
!10 = distinct !{!10, !"_ZN4hlsl19RWByteAddressBuffer19__createFromBindingEjjijPKc"}
!11 = !{!12}
!12 = distinct !{!12, !13, !"_ZN4hlsl19RWByteAddressBuffer19__createFromBindingEjjijPKc: %agg.result"}
!13 = distinct !{!13, !"_ZN4hlsl19RWByteAddressBuffer19__createFromBindingEjjijPKc"}
