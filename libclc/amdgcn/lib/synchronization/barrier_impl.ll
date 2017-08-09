declare i32 @__clc_clk_local_mem_fence() #1
declare i32 @__clc_clk_global_mem_fence() #1
declare void @llvm.amdgcn.s.barrier() #0

define void @barrier(i32 %flags) #2 {
barrier_local_test:
  %CLK_LOCAL_MEM_FENCE = call i32 @__clc_clk_local_mem_fence()
  %0 = and i32 %flags, %CLK_LOCAL_MEM_FENCE
  %1 = icmp ne i32 %0, 0
  br i1 %1, label %barrier_local, label %barrier_global_test

barrier_local:
  call void @llvm.amdgcn.s.barrier()
  br label %barrier_global_test

barrier_global_test:
  %CLK_GLOBAL_MEM_FENCE = call i32 @__clc_clk_global_mem_fence()
  %2 = and i32 %flags, %CLK_GLOBAL_MEM_FENCE
  %3 = icmp ne i32 %2, 0
  br i1 %3, label %barrier_global, label %done

barrier_global:
  call void @llvm.amdgcn.s.barrier()
  br label %done

done:
  ret void
}

attributes #0 = { nounwind convergent }
attributes #1 = { nounwind alwaysinline }
attributes #2 = { nounwind convergent alwaysinline }
