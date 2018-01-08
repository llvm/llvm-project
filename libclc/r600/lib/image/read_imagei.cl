#include <clc/clc.h>

_CLC_DECL float4 __clc_read_imagef_tex(image2d_t, sampler_t, float2);

int4 __clc_reinterpret_v4f_to_v4i(float4 v) {
  union {
    int4 v4i;
    float4 v4f;
  } res = { .v4f = v};
  return res.v4i;
}

_CLC_OVERLOAD _CLC_DEF int4 read_imagei(image2d_t image, sampler_t sampler,
                                        int2 coord) {
  float2 coord_float = (float2)(coord.x, coord.y);
  return __clc_reinterpret_v4f_to_v4i(
    __clc_read_imagef_tex(image, sampler, coord_float));
}
_CLC_OVERLOAD _CLC_DEF int4 read_imagei(image2d_t image, sampler_t sampler,
                                        float2 coord) {
  return __clc_reinterpret_v4f_to_v4i(
    __clc_read_imagef_tex(image, sampler, coord));
}
