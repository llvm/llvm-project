_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE asin(__CLC_GENTYPE x) {
  return atan2(x, sqrt( (__CLC_GENTYPE)1.0 -(x*x) ));
}