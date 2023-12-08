_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE fma(__CLC_GENTYPE a, __CLC_GENTYPE b, __CLC_GENTYPE c) {
#if __CLC_FPSIZE == 32 && HAVE_HW_FMA32() == 0
	return __clc_sw_fma(a, b, c);
#else
	return __clc_fma(a, b, c);
#endif
}
