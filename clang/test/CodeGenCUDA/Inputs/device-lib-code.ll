define linkonce_odr protected float @__ocml_fma_f32(float %0, float %1, float %2) local_unnamed_addr {
  %4 = tail call float @llvm.fma.f32(float %0, float %1, float %2)
  ret float %4
}
declare float @llvm.fma.f32(float, float, float)
