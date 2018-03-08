target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-A5"
target triple = "amdgcn-amd-amdhsa"

;;;;; Add ;;;;;
define half @__llvm_add_rte_f16(half, half) local_unnamed_addr #0 {
  %3 = tail call half @llvm.experimental.constrained.fadd.f16(half %0, half %1, metadata !"round.tonearest", metadata !"fpexcept.strict") #1
  ret half %3
}

define half @__llvm_add_rtn_f16(half, half) local_unnamed_addr #0 {
  %3 = tail call half @llvm.experimental.constrained.fadd.f16(half %0, half %1, metadata !"round.downward", metadata !"fpexcept.strict") #1
  ret half %3
}

define half @__llvm_add_rtp_f16(half, half) local_unnamed_addr #0 {
  %3 = tail call half @llvm.experimental.constrained.fadd.f16(half %0, half %1, metadata !"round.upward", metadata !"fpexcept.strict") #1
  ret half %3
}

define half @__llvm_add_rtz_f16(half, half) local_unnamed_addr #0 {
  %3 = tail call half @llvm.experimental.constrained.fadd.f16(half %0, half %1, metadata !"round.towardzero", metadata !"fpexcept.strict") #1
  ret half %3
}

define float @__llvm_add_rte_f32(float, float) local_unnamed_addr #0 {
  %3 = tail call float @llvm.experimental.constrained.fadd.f32(float %0, float %1, metadata !"round.tonearest", metadata !"fpexcept.strict") #1
  ret float %3
}

define float @__llvm_add_rtn_f32(float, float) local_unnamed_addr #0 {
  %3 = tail call float @llvm.experimental.constrained.fadd.f32(float %0, float %1, metadata !"round.downward", metadata !"fpexcept.strict") #1
  ret float %3
}

define float @__llvm_add_rtp_f32(float, float) local_unnamed_addr #0 {
  %3 = tail call float @llvm.experimental.constrained.fadd.f32(float %0, float %1, metadata !"round.upward", metadata !"fpexcept.strict") #1
  ret float %3
}

define float @__llvm_add_rtz_f32(float, float) local_unnamed_addr #0 {
  %3 = tail call float @llvm.experimental.constrained.fadd.f32(float %0, float %1, metadata !"round.towardzero", metadata !"fpexcept.strict") #1
  ret float %3
}

define double @__llvm_add_rte_f64(double, double) local_unnamed_addr #0 {
  %3 = tail call double @llvm.experimental.constrained.fadd.f64(double %0, double %1, metadata !"round.tonearest", metadata !"fpexcept.strict") #1
  ret double %3
}

define double @__llvm_add_rtn_f64(double, double) local_unnamed_addr #0 {
  %3 = tail call double @llvm.experimental.constrained.fadd.f64(double %0, double %1, metadata !"round.downward", metadata !"fpexcept.strict") #1
  ret double %3
}

define double @__llvm_add_rtp_f64(double, double) local_unnamed_addr #0 {
  %3 = tail call double @llvm.experimental.constrained.fadd.f64(double %0, double %1, metadata !"round.upward", metadata !"fpexcept.strict") #1
  ret double %3
}

define double @__llvm_add_rtz_f64(double, double) local_unnamed_addr #0 {
  %3 = tail call double @llvm.experimental.constrained.fadd.f64(double %0, double %1, metadata !"round.towardzero", metadata !"fpexcept.strict") #1
  ret double %3
}

;;;;; Sub ;;;;;
define half @__llvm_sub_rte_f16(half, half) local_unnamed_addr #0 {
  %3 = tail call half @llvm.experimental.constrained.fsub.f16(half %0, half %1, metadata !"round.tonearest", metadata !"fpexcept.strict") #1
  ret half %3
}

define half @__llvm_sub_rtn_f16(half, half) local_unnamed_addr #0 {
  %3 = tail call half @llvm.experimental.constrained.fsub.f16(half %0, half %1, metadata !"round.downward", metadata !"fpexcept.strict") #1
  ret half %3
}

define half @__llvm_sub_rtp_f16(half, half) local_unnamed_addr #0 {
  %3 = tail call half @llvm.experimental.constrained.fsub.f16(half %0, half %1, metadata !"round.upward", metadata !"fpexcept.strict") #1
  ret half %3
}

define half @__llvm_sub_rtz_f16(half, half) local_unnamed_addr #0 {
  %3 = tail call half @llvm.experimental.constrained.fsub.f16(half %0, half %1, metadata !"round.towardzero", metadata !"fpexcept.strict") #1
  ret half %3
}

define float @__llvm_sub_rte_f32(float, float) local_unnamed_addr #0 {
  %3 = tail call float @llvm.experimental.constrained.fsub.f32(float %0, float %1, metadata !"round.tonearest", metadata !"fpexcept.strict") #1
  ret float %3
}

define float @__llvm_sub_rtn_f32(float, float) local_unnamed_addr #0 {
  %3 = tail call float @llvm.experimental.constrained.fsub.f32(float %0, float %1, metadata !"round.downward", metadata !"fpexcept.strict") #1
  ret float %3
}

define float @__llvm_sub_rtp_f32(float, float) local_unnamed_addr #0 {
  %3 = tail call float @llvm.experimental.constrained.fsub.f32(float %0, float %1, metadata !"round.upward", metadata !"fpexcept.strict") #1
  ret float %3
}

define float @__llvm_sub_rtz_f32(float, float) local_unnamed_addr #0 {
  %3 = tail call float @llvm.experimental.constrained.fsub.f32(float %0, float %1, metadata !"round.towardzero", metadata !"fpexcept.strict") #1
  ret float %3
}

define double @__llvm_sub_rte_f64(double, double) local_unnamed_addr #0 {
  %3 = tail call double @llvm.experimental.constrained.fsub.f64(double %0, double %1, metadata !"round.tonearest", metadata !"fpexcept.strict") #1
  ret double %3
}

define double @__llvm_sub_rtn_f64(double, double) local_unnamed_addr #0 {
  %3 = tail call double @llvm.experimental.constrained.fsub.f64(double %0, double %1, metadata !"round.downward", metadata !"fpexcept.strict") #1
  ret double %3
}

define double @__llvm_sub_rtp_f64(double, double) local_unnamed_addr #0 {
  %3 = tail call double @llvm.experimental.constrained.fsub.f64(double %0, double %1, metadata !"round.upward", metadata !"fpexcept.strict") #1
  ret double %3
}

define double @__llvm_sub_rtz_f64(double, double) local_unnamed_addr #0 {
  %3 = tail call double @llvm.experimental.constrained.fsub.f64(double %0, double %1, metadata !"round.towardzero", metadata !"fpexcept.strict") #1
  ret double %3
}

;;;;; Mul ;;;;;
define half @__llvm_mul_rte_f16(half, half) local_unnamed_addr #0 {
  %3 = tail call half @llvm.experimental.constrained.fmul.f16(half %0, half %1, metadata !"round.tonearest", metadata !"fpexcept.strict") #1
  ret half %3
}

define half @__llvm_mul_rtn_f16(half, half) local_unnamed_addr #0 {
  %3 = tail call half @llvm.experimental.constrained.fmul.f16(half %0, half %1, metadata !"round.downward", metadata !"fpexcept.strict") #1
  ret half %3
}

define half @__llvm_mul_rtp_f16(half, half) local_unnamed_addr #0 {
  %3 = tail call half @llvm.experimental.constrained.fmul.f16(half %0, half %1, metadata !"round.upward", metadata !"fpexcept.strict") #1
  ret half %3
}

define half @__llvm_mul_rtz_f16(half, half) local_unnamed_addr #0 {
  %3 = tail call half @llvm.experimental.constrained.fmul.f16(half %0, half %1, metadata !"round.towardzero", metadata !"fpexcept.strict") #1
  ret half %3
}

define float @__llvm_mul_rte_f32(float, float) local_unnamed_addr #0 {
  %3 = tail call float @llvm.experimental.constrained.fmul.f32(float %0, float %1, metadata !"round.tonearest", metadata !"fpexcept.strict") #1
  ret float %3
}

define float @__llvm_mul_rtn_f32(float, float) local_unnamed_addr #0 {
  %3 = tail call float @llvm.experimental.constrained.fmul.f32(float %0, float %1, metadata !"round.downward", metadata !"fpexcept.strict") #1
  ret float %3
}

define float @__llvm_mul_rtp_f32(float, float) local_unnamed_addr #0 {
  %3 = tail call float @llvm.experimental.constrained.fmul.f32(float %0, float %1, metadata !"round.upward", metadata !"fpexcept.strict") #1
  ret float %3
}

define float @__llvm_mul_rtz_f32(float, float) local_unnamed_addr #0 {
  %3 = tail call float @llvm.experimental.constrained.fmul.f32(float %0, float %1, metadata !"round.towardzero", metadata !"fpexcept.strict") #1
  ret float %3
}

define double @__llvm_mul_rte_f64(double, double) local_unnamed_addr #0 {
  %3 = tail call double @llvm.experimental.constrained.fmul.f64(double %0, double %1, metadata !"round.tonearest", metadata !"fpexcept.strict") #1
  ret double %3
}

define double @__llvm_mul_rtn_f64(double, double) local_unnamed_addr #0 {
  %3 = tail call double @llvm.experimental.constrained.fmul.f64(double %0, double %1, metadata !"round.downward", metadata !"fpexcept.strict") #1
  ret double %3
}

define double @__llvm_mul_rtp_f64(double, double) local_unnamed_addr #0 {
  %3 = tail call double @llvm.experimental.constrained.fmul.f64(double %0, double %1, metadata !"round.upward", metadata !"fpexcept.strict") #1
  ret double %3
}

define double @__llvm_mul_rtz_f64(double, double) local_unnamed_addr #0 {
  %3 = tail call double @llvm.experimental.constrained.fmul.f64(double %0, double %1, metadata !"round.towardzero", metadata !"fpexcept.strict") #1
  ret double %3
}

;;;;; Div ;;;;;
define half @__llvm_div_rte_f16(half, half) local_unnamed_addr #0 {
  %3 = tail call half @llvm.experimental.constrained.fdiv.f16(half %0, half %1, metadata !"round.tonearest", metadata !"fpexcept.strict") #1
  ret half %3
}

define half @__llvm_div_rtn_f16(half, half) local_unnamed_addr #0 {
  %3 = tail call half @llvm.experimental.constrained.fdiv.f16(half %0, half %1, metadata !"round.downward", metadata !"fpexcept.strict") #1
  ret half %3
}

define half @__llvm_div_rtp_f16(half, half) local_unnamed_addr #0 {
  %3 = tail call half @llvm.experimental.constrained.fdiv.f16(half %0, half %1, metadata !"round.upward", metadata !"fpexcept.strict") #1
  ret half %3
}

define half @__llvm_div_rtz_f16(half, half) local_unnamed_addr #0 {
  %3 = tail call half @llvm.experimental.constrained.fdiv.f16(half %0, half %1, metadata !"round.towardzero", metadata !"fpexcept.strict") #1
  ret half %3
}

define float @__llvm_div_rte_f32(float, float) local_unnamed_addr #0 {
  %3 = tail call float @llvm.experimental.constrained.fdiv.f32(float %0, float %1, metadata !"round.tonearest", metadata !"fpexcept.strict") #1
  ret float %3
}

define float @__llvm_div_rtn_f32(float, float) local_unnamed_addr #0 {
  %3 = tail call float @llvm.experimental.constrained.fdiv.f32(float %0, float %1, metadata !"round.downward", metadata !"fpexcept.strict") #1
  ret float %3
}

define float @__llvm_div_rtp_f32(float, float) local_unnamed_addr #0 {
  %3 = tail call float @llvm.experimental.constrained.fdiv.f32(float %0, float %1, metadata !"round.upward", metadata !"fpexcept.strict") #1
  ret float %3
}

define float @__llvm_div_rtz_f32(float, float) local_unnamed_addr #0 {
  %3 = tail call float @llvm.experimental.constrained.fdiv.f32(float %0, float %1, metadata !"round.towardzero", metadata !"fpexcept.strict") #1
  ret float %3
}

define double @__llvm_div_rte_f64(double, double) local_unnamed_addr #0 {
  %3 = tail call double @llvm.experimental.constrained.fdiv.f64(double %0, double %1, metadata !"round.tonearest", metadata !"fpexcept.strict") #1
  ret double %3
}

define double @__llvm_div_rtn_f64(double, double) local_unnamed_addr #0 {
  %3 = tail call double @llvm.experimental.constrained.fdiv.f64(double %0, double %1, metadata !"round.downward", metadata !"fpexcept.strict") #1
  ret double %3
}

define double @__llvm_div_rtp_f64(double, double) local_unnamed_addr #0 {
  %3 = tail call double @llvm.experimental.constrained.fdiv.f64(double %0, double %1, metadata !"round.upward", metadata !"fpexcept.strict") #1
  ret double %3
}

define double @__llvm_div_rtz_f64(double, double) local_unnamed_addr #0 {
  %3 = tail call double @llvm.experimental.constrained.fdiv.f64(double %0, double %1, metadata !"round.towardzero", metadata !"fpexcept.strict") #1
  ret double %3
}

;;;;; Sqrt ;;;;;
define half @__llvm_sqrt_rte_f16(half) local_unnamed_addr #0 {
  %2 = tail call half @llvm.experimental.constrained.sqrt.f16(half %0,  metadata !"round.tonearest", metadata !"fpexcept.strict") #1
  ret half %2
}

define half @__llvm_sqrt_rtn_f16(half) local_unnamed_addr #0 {
  %2 = tail call half @llvm.experimental.constrained.sqrt.f16(half %0,  metadata !"round.downward", metadata !"fpexcept.strict") #1
  ret half %2
}

define half @__llvm_sqrt_rtp_f16(half) local_unnamed_addr #0 {
  %2 = tail call half @llvm.experimental.constrained.sqrt.f16(half %0,  metadata !"round.upward", metadata !"fpexcept.strict") #1
  ret half %2
}

define half @__llvm_sqrt_rtz_f16(half) local_unnamed_addr #0 {
  %2 = tail call half @llvm.experimental.constrained.sqrt.f16(half %0,  metadata !"round.towardzero", metadata !"fpexcept.strict") #1
  ret half %2
}

define float @__llvm_sqrt_rte_f32(float) local_unnamed_addr #0 {
  %2 = tail call float @llvm.experimental.constrained.sqrt.f32(float %0,  metadata !"round.tonearest", metadata !"fpexcept.strict") #1
  ret float %2
}

define float @__llvm_sqrt_rtn_f32(float) local_unnamed_addr #0 {
  %2 = tail call float @llvm.experimental.constrained.sqrt.f32(float %0,  metadata !"round.downward", metadata !"fpexcept.strict") #1
  ret float %2
}

define float @__llvm_sqrt_rtp_f32(float) local_unnamed_addr #0 {
  %2 = tail call float @llvm.experimental.constrained.sqrt.f32(float %0,  metadata !"round.upward", metadata !"fpexcept.strict") #1
  ret float %2
}

define float @__llvm_sqrt_rtz_f32(float) local_unnamed_addr #0 {
  %2 = tail call float @llvm.experimental.constrained.sqrt.f32(float %0,  metadata !"round.towardzero", metadata !"fpexcept.strict") #1
  ret float %2
}

define double @__llvm_sqrt_rte_f64(double) local_unnamed_addr #0 {
  %2 = tail call double @llvm.experimental.constrained.sqrt.f64(double %0,  metadata !"round.tonearest", metadata !"fpexcept.strict") #1
  ret double %2
}

define double @__llvm_sqrt_rtn_f64(double) local_unnamed_addr #0 {
  %2 = tail call double @llvm.experimental.constrained.sqrt.f64(double %0,  metadata !"round.downward", metadata !"fpexcept.strict") #1
  ret double %2
}

define double @__llvm_sqrt_rtp_f64(double) local_unnamed_addr #0 {
  %2 = tail call double @llvm.experimental.constrained.sqrt.f64(double %0,  metadata !"round.upward", metadata !"fpexcept.strict") #1
  ret double %2
}

define double @__llvm_sqrt_rtz_f64(double) local_unnamed_addr #0 {
  %2 = tail call double @llvm.experimental.constrained.sqrt.f64(double %0,  metadata !"round.towardzero", metadata !"fpexcept.strict") #1
  ret double %2
}

;;;;; Fma ;;;;;
define half @__llvm_fma_rte_f16(half, half, half) local_unnamed_addr #0 {
  %4 = tail call half @llvm.experimental.constrained.fma.f16(half %0, half %1, half %2, metadata !"round.tonearest", metadata !"fpexcept.strict") #1
  ret half %4
}

define half @__llvm_fma_rtn_f16(half, half, half) local_unnamed_addr #0 {
  %4 = tail call half @llvm.experimental.constrained.fma.f16(half %0, half %1, half %2, metadata !"round.downward", metadata !"fpexcept.strict") #1
  ret half %4
}

define half @__llvm_fma_rtp_f16(half, half, half) local_unnamed_addr #0 {
  %4 = tail call half @llvm.experimental.constrained.fma.f16(half %0, half %1, half %2, metadata !"round.upward", metadata !"fpexcept.strict") #1
  ret half %4
}

define half @__llvm_fma_rtz_f16(half, half, half) local_unnamed_addr #0 {
  %4 = tail call half @llvm.experimental.constrained.fma.f16(half %0, half %1, half %2, metadata !"round.towardzero", metadata !"fpexcept.strict") #1
  ret half %4
}

define float @__llvm_fma_rte_f32(float, float, float) local_unnamed_addr #0 {
  %4 = tail call float @llvm.experimental.constrained.fma.f32(float %0, float %1, float %2, metadata !"round.tonearest", metadata !"fpexcept.strict") #1
  ret float %4
}

define float @__llvm_fma_rtn_f32(float, float, float) local_unnamed_addr #0 {
  %4 = tail call float @llvm.experimental.constrained.fma.f32(float %0, float %1, float %2, metadata !"round.downward", metadata !"fpexcept.strict") #1
  ret float %4
}

define float @__llvm_fma_rtp_f32(float, float, float) local_unnamed_addr #0 {
  %4 = tail call float @llvm.experimental.constrained.fma.f32(float %0, float %1, float %2, metadata !"round.upward", metadata !"fpexcept.strict") #1
  ret float %4
}

define float @__llvm_fma_rtz_f32(float, float, float) local_unnamed_addr #0 {
  %4 = tail call float @llvm.experimental.constrained.fma.f32(float %0, float %1, float %2, metadata !"round.towardzero", metadata !"fpexcept.strict") #1
  ret float %4
}

define double @__llvm_fma_rte_f64(double, double, double) local_unnamed_addr #0 {
  %4 = tail call double @llvm.experimental.constrained.fma.f64(double %0, double %1, double %2, metadata !"round.tonearest", metadata !"fpexcept.strict") #1
  ret double %4
}

define double @__llvm_fma_rtn_f64(double, double, double) local_unnamed_addr #0 {
  %4 = tail call double @llvm.experimental.constrained.fma.f64(double %0, double %1, double %2, metadata !"round.downward", metadata !"fpexcept.strict") #1
  ret double %4
}

define double @__llvm_fma_rtp_f64(double, double, double) local_unnamed_addr #0 {
  %4 = tail call double @llvm.experimental.constrained.fma.f64(double %0, double %1, double %2, metadata !"round.upward", metadata !"fpexcept.strict") #1
  ret double %4
}

define double @__llvm_fma_rtz_f64(double, double, double) local_unnamed_addr #0 {
  %4 = tail call double @llvm.experimental.constrained.fma.f64(double %0, double %1, double %2, metadata !"round.towardzero", metadata !"fpexcept.strict") #1
  ret double %4
}

declare half @llvm.experimental.constrained.fdiv.f16(half, half, metadata, metadata) local_unnamed_addr #1
declare half @llvm.experimental.constrained.fmul.f16(half, half, metadata, metadata) local_unnamed_addr #1
declare half @llvm.experimental.constrained.fadd.f16(half, half, metadata, metadata) local_unnamed_addr #1
declare half @llvm.experimental.constrained.fsub.f16(half, half, metadata, metadata) local_unnamed_addr #1
declare half @llvm.experimental.constrained.sqrt.f16(half, metadata, metadata) local_unnamed_addr #1
declare half @llvm.experimental.constrained.fma.f16(half, half, half, metadata, metadata) local_unnamed_addr #1

declare float @llvm.experimental.constrained.fdiv.f32(float, float, metadata, metadata) local_unnamed_addr #1
declare float @llvm.experimental.constrained.fmul.f32(float, float, metadata, metadata) local_unnamed_addr #1
declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata) local_unnamed_addr #1
declare float @llvm.experimental.constrained.fsub.f32(float, float, metadata, metadata) local_unnamed_addr #1
declare float @llvm.experimental.constrained.sqrt.f32(float, metadata, metadata) local_unnamed_addr #1
declare float @llvm.experimental.constrained.fma.f32(float, float, float, metadata, metadata) local_unnamed_addr #1

declare double @llvm.experimental.constrained.fdiv.f64(double, double, metadata, metadata) local_unnamed_addr #1
declare double @llvm.experimental.constrained.fmul.f64(double, double, metadata, metadata) local_unnamed_addr #1
declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata) local_unnamed_addr #1
declare double @llvm.experimental.constrained.fsub.f64(double, double, metadata, metadata) local_unnamed_addr #1
declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata) local_unnamed_addr #1
declare double @llvm.experimental.constrained.fma.f64(double, double, double, metadata, metadata) local_unnamed_addr #1

attributes #0 = { alwaysinline nounwind readnone }
attributes #1 = { nounwind readnone }

