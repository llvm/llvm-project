_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE exp(__CLC_GENTYPE val) {
  // exp(x) = exp2(x * log2(e))
#if __CLC_FPSIZE == 32
  return exp2(val * M_LOG2E_F);
#elif __CLC_FPSIZE == 64
  return exp2(val * M_LOG2E);
#else
#error unknown _CLC_FPSIZE
#endif
}
