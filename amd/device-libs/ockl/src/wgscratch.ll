target triple = "amdgcn-amd-amdhsa"

; 1024 work-items means up to 32 work groups
@__scratch_lds = linkonce_odr hidden addrspace(3) global [32 x i64] poison, align 8

define protected noundef align 8 dereferenceable(256) ptr addrspace(3) @__get_scratch_lds() #0 {
  ret ptr addrspace(3) @__scratch_lds
}

attributes #0 = { alwaysinline mustprogress nofree norecurse nosync nounwind speculatable willreturn memory(none) }
