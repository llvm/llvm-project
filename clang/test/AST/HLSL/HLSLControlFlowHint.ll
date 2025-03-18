; ModuleID = '/workspace/llvm-project/clang/test/AST/HLSL/HLSLControlFlowHint.hlsl'
source_filename = "/workspace/llvm-project/clang/test/AST/HLSL/HLSLControlFlowHint.hlsl"
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxilv1.3-pc-shadermodel6.3-compute"

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define noundef i32 @_Z6branchi(i32 noundef %X) #0 {
entry:
  %X.addr = alloca i32, align 4
  %resp = alloca i32, align 4
  store i32 %X, ptr %X.addr, align 4
  %0 = load i32, ptr %X.addr, align 4
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else, !hlsl.controlflow.hint !3

if.then:                                          ; preds = %entry
  %1 = load i32, ptr %X.addr, align 4
  %sub = sub nsw i32 0, %1
  store i32 %sub, ptr %resp, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %2 = load i32, ptr %X.addr, align 4
  %mul = mul nsw i32 %2, 2
  store i32 %mul, ptr %resp, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %3 = load i32, ptr %resp, align 4
  ret i32 %3
}

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define noundef i32 @_Z7flatteni(i32 noundef %X) #0 {
entry:
  %X.addr = alloca i32, align 4
  %resp = alloca i32, align 4
  store i32 %X, ptr %X.addr, align 4
  %0 = load i32, ptr %X.addr, align 4
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else, !hlsl.controlflow.hint !4

if.then:                                          ; preds = %entry
  %1 = load i32, ptr %X.addr, align 4
  %sub = sub nsw i32 0, %1
  store i32 %sub, ptr %resp, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %2 = load i32, ptr %X.addr, align 4
  %mul = mul nsw i32 %2, 2
  store i32 %mul, ptr %resp, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %3 = load i32, ptr %resp, align 4
  ret i32 %3
}

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define noundef i32 @_Z7no_attri(i32 noundef %X) #0 {
entry:
  %X.addr = alloca i32, align 4
  %resp = alloca i32, align 4
  store i32 %X, ptr %X.addr, align 4
  %0 = load i32, ptr %X.addr, align 4
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %1 = load i32, ptr %X.addr, align 4
  %sub = sub nsw i32 0, %1
  store i32 %sub, ptr %resp, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %2 = load i32, ptr %X.addr, align 4
  %mul = mul nsw i32 %2, 2
  store i32 %mul, ptr %resp, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %3 = load i32, ptr %resp, align 4
  ret i32 %3
}

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define noundef i32 @_Z14flatten_switchi(i32 noundef %X) #0 {
entry:
  %X.addr = alloca i32, align 4
  %resp = alloca i32, align 4
  store i32 %X, ptr %X.addr, align 4
  %0 = load i32, ptr %X.addr, align 4
  switch i32 %0, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
  ], !hlsl.controlflow.hint !4

sw.bb:                                            ; preds = %entry
  %1 = load i32, ptr %X.addr, align 4
  %sub = sub nsw i32 0, %1
  store i32 %sub, ptr %resp, align 4
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  %2 = load i32, ptr %X.addr, align 4
  %3 = load i32, ptr %X.addr, align 4
  %add = add nsw i32 %2, %3
  store i32 %add, ptr %resp, align 4
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  %4 = load i32, ptr %X.addr, align 4
  %5 = load i32, ptr %X.addr, align 4
  %mul = mul nsw i32 %4, %5
  store i32 %mul, ptr %resp, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb2, %sw.bb1, %sw.bb
  %6 = load i32, ptr %resp, align 4
  ret i32 %6
}

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define noundef i32 @_Z13branch_switchi(i32 noundef %X) #0 {
entry:
  %X.addr = alloca i32, align 4
  %resp = alloca i32, align 4
  store i32 %X, ptr %X.addr, align 4
  %0 = load i32, ptr %X.addr, align 4
  switch i32 %0, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
  ], !hlsl.controlflow.hint !3

sw.bb:                                            ; preds = %entry
  %1 = load i32, ptr %X.addr, align 4
  %sub = sub nsw i32 0, %1
  store i32 %sub, ptr %resp, align 4
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  %2 = load i32, ptr %X.addr, align 4
  %3 = load i32, ptr %X.addr, align 4
  %add = add nsw i32 %2, %3
  store i32 %add, ptr %resp, align 4
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  %4 = load i32, ptr %X.addr, align 4
  %5 = load i32, ptr %X.addr, align 4
  %mul = mul nsw i32 %4, %5
  store i32 %mul, ptr %resp, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb2, %sw.bb1, %sw.bb
  %6 = load i32, ptr %resp, align 4
  ret i32 %6
}

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define noundef i32 @_Z14no_attr_switchi(i32 noundef %X) #0 {
entry:
  %X.addr = alloca i32, align 4
  %resp = alloca i32, align 4
  store i32 %X, ptr %X.addr, align 4
  %0 = load i32, ptr %X.addr, align 4
  switch i32 %0, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
  ]

sw.bb:                                            ; preds = %entry
  %1 = load i32, ptr %X.addr, align 4
  %sub = sub nsw i32 0, %1
  store i32 %sub, ptr %resp, align 4
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  %2 = load i32, ptr %X.addr, align 4
  %3 = load i32, ptr %X.addr, align 4
  %add = add nsw i32 %2, %3
  store i32 %add, ptr %resp, align 4
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  %4 = load i32, ptr %X.addr, align 4
  %5 = load i32, ptr %X.addr, align 4
  %mul = mul nsw i32 %4, %5
  store i32 %mul, ptr %resp, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb2, %sw.bb1, %sw.bb
  %6 = load i32, ptr %resp, align 4
  ret i32 %6
}

attributes #0 = { alwaysinline convergent mustprogress norecurse nounwind "approx-func-fp-math"="true" "hlsl.export" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0}
!dx.valver = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 8}
!2 = !{!"clang version 21.0.0git (https://github.com/joaosaffran/llvm-project.git fe0db909f0b0d61dbc0d2f7a3313138808c20194)"}
!3 = !{!"hlsl.controlflow.hint", i32 1}
!4 = !{!"hlsl.controlflow.hint", i32 2}
