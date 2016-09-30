target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "amdgcn--amdhsa"

; 1024 work-items means up to 16 work groups
@__scratch_lds = linkonce_odr hidden addrspace(3) global [16 x i64] undef, align 8

define i64 addrspace(3)* @__get_scratch_lds() #0 {
  ret i64 addrspace(3)* getelementptr inbounds ([16 x i64], [16 x i64] addrspace(3)* @__scratch_lds, i64 0, i64 0)
}

attributes #0 = { alwaysinline norecurse nounwind readnone }
