_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE exp10(__CLC_GENTYPE val) {
  // exp10(x) = exp2(x * log2(10))
#if __CLC_FPSIZE == 32
  return exp2(val * M_LOG210_F);
#elif __CLC_FPSIZE == 64
  return exp2(val * M_LOG210);
#else
#error unknown _CLC_FPSIZE
#endif
}
