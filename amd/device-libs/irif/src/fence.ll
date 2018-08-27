target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"
target triple = "amdgcn-amd-amdhsa"

define protected void @__llvm_fence_acq_wi() local_unnamed_addr #0 {
  fence syncscope("singlethread") acquire
  ret void
}

define protected void @__llvm_fence_acq_sg() local_unnamed_addr #0 {
  fence syncscope("wavefront") acquire
  ret void
}

define protected void @__llvm_fence_acq_wg() local_unnamed_addr #0 {
  fence syncscope("workgroup") acquire
  ret void
}

define protected void @__llvm_fence_acq_dev() local_unnamed_addr #0 {
  fence syncscope("agent") acquire
  ret void
}

define protected void @__llvm_fence_acq_sys() local_unnamed_addr #0 {
  fence acquire
  ret void
}

define protected void @__llvm_fence_rel_wi() local_unnamed_addr #0 {
  fence syncscope("singlethread") release
  ret void
}

define protected void @__llvm_fence_rel_sg() local_unnamed_addr #0 {
  fence syncscope("wavefront") release
  ret void
}

define protected void @__llvm_fence_rel_wg() local_unnamed_addr #0 {
  fence syncscope("workgroup") release
  ret void
}

define protected void @__llvm_fence_rel_dev() local_unnamed_addr #0 {
  fence syncscope("agent") release
  ret void
}

define protected void @__llvm_fence_rel_sys() local_unnamed_addr #0 {
  fence release
  ret void
}

define protected void @__llvm_fence_ar_wi() local_unnamed_addr #0 {
  fence syncscope("singlethread") acq_rel
  ret void
}

define protected void @__llvm_fence_ar_sg() local_unnamed_addr #0 {
  fence syncscope("wavefront") acq_rel
  ret void
}

define protected void @__llvm_fence_ar_wg() local_unnamed_addr #0 {
  fence syncscope("workgroup") acq_rel
  ret void
}

define protected void @__llvm_fence_ar_dev() local_unnamed_addr #0 {
  fence syncscope("agent") acq_rel
  ret void
}

define protected void @__llvm_fence_ar_sys() local_unnamed_addr #0 {
  fence acq_rel
  ret void
}

define protected void @__llvm_fence_sc_wi() local_unnamed_addr #0 {
  fence syncscope("singlethread") seq_cst
  ret void
}

define protected void @__llvm_fence_sc_sg() local_unnamed_addr #0 {
  fence syncscope("wavefront") seq_cst
  ret void
}

define protected void @__llvm_fence_sc_wg() local_unnamed_addr #0 {
  fence syncscope("workgroup") seq_cst
  ret void
}

define protected void @__llvm_fence_sc_dev() local_unnamed_addr #0 {
  fence syncscope("agent") seq_cst
  ret void
}

define protected void @__llvm_fence_sc_sys() local_unnamed_addr #0 {
  fence seq_cst
  ret void
}

attributes #0 = { alwaysinline norecurse nounwind }

