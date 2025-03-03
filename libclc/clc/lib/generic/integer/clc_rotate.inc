// The rotate(A, B) builtin left-shifts corresponding to the usual OpenCL shift
// modulo rules. These rules state that A is left-shifted by the log2(N) least
// significant bits in B when viewed as an unsigned integer value. Thus we don't
// have to worry about signed shift amounts, and can perform the computation in
// unsigned types.
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_rotate(__CLC_GENTYPE x,
                                                  __CLC_GENTYPE n) {
  __CLC_U_GENTYPE x_as_u = __CLC_AS_U_GENTYPE(x);
  __CLC_U_GENTYPE mask = (__CLC_U_GENTYPE)(__CLC_GENSIZE - 1);

  __CLC_U_GENTYPE lshift_amt = __CLC_AS_U_GENTYPE(n) & mask;

  __CLC_U_GENTYPE rshift_amt =
      (((__CLC_U_GENTYPE)__CLC_GENSIZE - lshift_amt) & mask);

  __CLC_U_GENTYPE result = (x_as_u << lshift_amt) | (x_as_u >> rshift_amt);

  return __CLC_AS_GENTYPE(result);
}
