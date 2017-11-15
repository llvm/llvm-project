_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE maxmag(__CLC_GENTYPE x, __CLC_GENTYPE y) {
  const __CLC_GENTYPE res = select(y, x, isgreater(fabs(x), fabs(y)));
  return select(res, fmax(x, y), isnan(x) | isnan(y) | isequal(fabs(x), fabs(y)));
}
