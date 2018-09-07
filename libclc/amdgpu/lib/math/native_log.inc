_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE native_log(__CLC_GENTYPE val) {
  return native_log2(val) * (1.0f / M_LOG2E_F);
}
