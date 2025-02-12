; ModuleID = 'C:\llvm-project\clang\test\CodeGenHLSL\cbuffer_and_namespaces.hlsl'
source_filename = "C:\\llvm-project\\clang\\test\\CodeGenHLSL\\cbuffer_and_namespaces.hlsl"
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxilv1.3-pc-shadermodel6.3-library"

%__cblayout_A = type <{ float }>
%__cblayout_B = type <{ float }>
%__cblayout_C = type <{ float, target("dx.Layout", %Foo, 4, 0) }>
%Foo = type <{ float }>

@A.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_A, 4, 0))
@_ZN2n02n11aE = external addrspace(2) global float, align 4
@B.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_B, 4, 0))
@_ZN2n01aE = external addrspace(2) global float, align 4
@C.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_C, 20, 0, 16))
@_ZN2n02n21aE = external addrspace(2) global float, align 4
@_ZN2n02n21bE = external addrspace(2) global target("dx.Layout", %Foo, 4, 0), align 4

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define noundef nofpclass(nan inf) float @_Z3foov() #0 {
entry:
  %0 = load float, ptr addrspace(2) @_ZN2n02n11aE, align 4
  %1 = load float, ptr addrspace(2) @_ZN2n01aE, align 4
  %add = fadd reassoc nnan ninf nsz arcp afn float %0, %1
  %2 = load float, ptr addrspace(2) @_ZN2n02n21aE, align 4
  %add1 = fadd reassoc nnan ninf nsz arcp afn float %add, %2
  ret float %add1
}

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define void @_Z4mainv() #0 {
entry:
  ret void
}

attributes #0 = { alwaysinline convergent mustprogress norecurse nounwind "approx-func-fp-math"="true" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!hlsl.cbs = !{!0, !1, !2}
!llvm.module.flags = !{!3, !4}
!dx.valver = !{!5}
!llvm.ident = !{!6}

!0 = !{ptr @A.cb, ptr addrspace(2) @_ZN2n02n11aE}
!1 = !{ptr @B.cb, ptr addrspace(2) @_ZN2n01aE}
!2 = !{ptr @C.cb, ptr addrspace(2) @_ZN2n02n21aE, ptr addrspace(2) @_ZN2n02n21bE}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 4, !"dx.disable_optimizations", i32 1}
!5 = !{i32 1, i32 8}
!6 = !{!"clang version 20.0.0git (C:/llvm-project/clang a8cdd4536867465e3d6e2b4ad8c49b27ee94dec8)"}
