_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE popcount(__CLC_GENTYPE x) {
/* LLVM-4+ implements i16 ops for VI+ ASICs. However, ctpop implementation
 * is missing until r326535. Therefore we have to convert sub i32 types to uint
 * as a workaround. */
#if __clang_major__ < 7 && __clang_major__ > 3 && __CLC_GENSIZE < 32
	/* Prevent sign extension on uint conversion */
	const __CLC_U_GENTYPE y = __CLC_XCONCAT(as_, __CLC_U_GENTYPE)(x);
	/* Convert to uintX */
	const __CLC_XCONCAT(uint, __CLC_VECSIZE) z = __CLC_XCONCAT(convert_uint, __CLC_VECSIZE)(y);
	/* Call popcount on uintX type */
	const __CLC_XCONCAT(uint, __CLC_VECSIZE) res = __clc_native_popcount(z);
	/* Convert the result back to gentype. */
	return __CLC_XCONCAT(convert_, __CLC_GENTYPE)(res);
#else
	return __clc_native_popcount(x);
#endif
}
