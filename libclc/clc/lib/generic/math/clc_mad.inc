_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_mad(__CLC_GENTYPE a, __CLC_GENTYPE b,
                                               __CLC_GENTYPE c) {
#pragma OPENCL FP_CONTRACT ON
  return a * b + c;
}
