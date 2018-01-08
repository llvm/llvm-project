_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE log10(__CLC_GENTYPE val) {
  // log10(x) = log2(x) / log2(10)
  // 1 / log2(10) = 0.30102999566 = log10(2)
  // SP representation is 0.30103 (0x1.344136p-2)
  // DP representation is 0.301029995659999993762312442414(0x1.34413509E61D8p-2)
#if __CLC_FPSIZE == 32
  return log2(val) * 0x1.344136p-2f;
#elif __CLC_FPSIZE == 64
  return log2(val) * 0x1.34413509E61D8p-2;
#else
#error unknown _CLC_FPSIZE
#endif
}
