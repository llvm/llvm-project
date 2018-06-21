// TODO: Enable half precision when the sw routine is implemented
#if __CLC_FPSIZE > 16
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE remquo(__CLC_GENTYPE x, __CLC_GENTYPE y, __CLC_ADDRESS_SPACE __CLC_INTN *q) {
  __CLC_INTN local_q;
  __CLC_GENTYPE ret = __clc_remquo(x, y, &local_q);
  *q = local_q;
  return ret;
}
#endif
