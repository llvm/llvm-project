#include <clc/clc.h>

_CLC_DECL void __clc_write_imageui_2d(image2d_t image, int2 coord, uint4 color);

_CLC_OVERLOAD _CLC_DEF void
write_imageui(image2d_t image, int2 coord, uint4 color)
{
  __clc_write_imageui_2d(image, coord, color);
}
