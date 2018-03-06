#ifdef __CLC_SCALAR
#define __CLC_VECSIZE
#endif

#if __CLC_FPSIZE == 64
#define __CLC_S_GENTYPE __CLC_XCONCAT(long, __CLC_VECSIZE)
#define __CLC_U_GENTYPE __CLC_XCONCAT(ulong, __CLC_VECSIZE)
#elif __CLC_FPSIZE == 32
#define __CLC_S_GENTYPE __CLC_XCONCAT(int, __CLC_VECSIZE)
#define __CLC_U_GENTYPE __CLC_XCONCAT(uint, __CLC_VECSIZE)
#elif __CLC_FPSIZE == 16
#define __CLC_S_GENTYPE __CLC_XCONCAT(char, __CLC_VECSIZE)
#define __CLC_U_GENTYPE __CLC_XCONCAT(uchar, __CLC_VECSIZE)
#endif
#ifdef __CLC_FPSIZE
#define __CLC_GENSIZE   __CLC_FPSIZE
#endif

#define __CLC_AS_S_GENTYPE __CLC_XCONCAT(as_, __CLC_S_GENTYPE)
#define __CLC_AS_GENTYPE __CLC_XCONCAT(as_, __CLC_GENTYPE)

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE select(__CLC_GENTYPE x, __CLC_GENTYPE y, __CLC_S_GENTYPE z)
{
#ifdef __CLC_SCALAR
	return z ? y : x;
#else
	__CLC_S_GENTYPE bitmask = z >> (__CLC_GENSIZE - 1);
	return __CLC_AS_GENTYPE(bitselect(__CLC_AS_S_GENTYPE(x), __CLC_AS_S_GENTYPE(y), bitmask));
#endif
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE select(__CLC_GENTYPE x, __CLC_GENTYPE y, __CLC_U_GENTYPE z)
{
	return select(x, y, __CLC_AS_S_GENTYPE(z));
}

#undef __CLC_AS_S_GENTYPE
#undef __CLC_AS_GENTYPE

#ifdef __CLC_FPSIZE
#undef __CLC_S_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_GENSIZE
#endif
#ifdef __CLC_SCALAR
#undef __CLC_VECSIZE
#endif
