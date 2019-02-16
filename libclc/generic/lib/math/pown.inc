// TODO: Enable half precision when the sw routine is implemented
#if __CLC_FPSIZE > 16
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE pown(__CLC_GENTYPE x, __CLC_INTN y) {
  return __clc_pown(x, y);
}
#endif
