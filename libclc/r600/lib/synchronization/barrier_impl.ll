declare void @llvm.r600.group.barrier() #0

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

define void @barrier(i32 %flags) #1 {
entry:
  ; We should call mem_fence here, but that is not implemented for r600 yet
  tail call void @llvm.r600.group.barrier()
  ret void
}

attributes #0 = { nounwind convergent }
attributes #1 = { nounwind convergent alwaysinline }
