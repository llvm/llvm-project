#ifdef __CLC_SCALAR
#define __CLC_VECSIZE
#endif

#if __CLC_FPSIZE == 64
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE nan(__CLC_XCONCAT(ulong, __CLC_VECSIZE) code)
{
	return __CLC_AS_GENTYPE(code | 0x7ff0000000000000ul);
}
#else
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE nan(__CLC_XCONCAT(uint, __CLC_VECSIZE) code)
{
	return __CLC_AS_GENTYPE(code | 0x7fc00000);
}
#endif


#ifdef __CLC_SCALAR
#undef __CLC_VECSIZE
#endif
