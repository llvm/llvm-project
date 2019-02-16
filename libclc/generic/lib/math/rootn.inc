// TODO: Enable half precision when the sw routine is implemented
#if __CLC_FPSIZE > 16
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE rootn(__CLC_GENTYPE x, __CLC_INTN y) {
  return __clc_rootn(x, y);
}
#endif
