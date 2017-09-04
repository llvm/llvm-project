declare void @llvm.r600.group.barrier() #0

define void @barrier(i32 %flags) #1 {
entry:
  ; We should call mem_fence here, but that is not implemented for r600 yet
  tail call void @llvm.r600.group.barrier()
  ret void
}

attributes #0 = { nounwind convergent }
attributes #1 = { nounwind convergent alwaysinline }
