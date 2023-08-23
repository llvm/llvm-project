target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8"
target triple = "amdgcn-amd-amdhsa"

; 1024 work-items means up to 32 work groups
@__scratch_lds = linkonce_odr hidden addrspace(3) global [32 x i64] poison, align 8

define protected noundef align 8 dereferenceable(256) ptr addrspace(3) @__get_scratch_lds() #0 {
  ret ptr addrspace(3) @__scratch_lds
}

attributes #0 = { alwaysinline mustprogress nofree norecurse nosync nounwind speculatable willreturn memory(none) }
