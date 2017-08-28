declare void @llvm.amdgcn.s.waitcnt(i32) #0

; Export waitcnt intrinsic for clang < 5
define void @__clc_amdgcn_s_waitcnt(i32 %flags) #1 {
entry:
  tail call void @llvm.amdgcn.s.waitcnt(i32 %flags)
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind alwaysinline }
