#define __CLC_DECLARE_SINCOS(ADDRSPACE, TYPE) \
  _CLC_OVERLOAD _CLC_DEF TYPE sincos (TYPE x, ADDRSPACE TYPE * cosval) { \
    *cosval = cos(x); \
    return sin(x); \
  }

__CLC_DECLARE_SINCOS(global, __CLC_GENTYPE)
__CLC_DECLARE_SINCOS(local, __CLC_GENTYPE)
__CLC_DECLARE_SINCOS(private, __CLC_GENTYPE)

#undef __CLC_DECLARE_SINCOS
