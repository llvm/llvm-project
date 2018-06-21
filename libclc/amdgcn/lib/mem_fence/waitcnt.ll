declare void @llvm.amdgcn.s.waitcnt(i32) #0

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"

; Export waitcnt intrinsic for clang < 5
define void @__clc_amdgcn_s_waitcnt(i32 %flags) #1 {
entry:
  tail call void @llvm.amdgcn.s.waitcnt(i32 %flags)
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind alwaysinline }
