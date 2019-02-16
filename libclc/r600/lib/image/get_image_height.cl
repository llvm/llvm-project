#include <clc/clc.h>

_CLC_DECL int __clc_get_image_height_2d(image2d_t);
_CLC_DECL int __clc_get_image_height_3d(image3d_t);

_CLC_OVERLOAD _CLC_DEF int
get_image_height(image2d_t image) {
  return __clc_get_image_height_2d(image);
}
_CLC_OVERLOAD _CLC_DEF int
get_image_height(image3d_t image) {
  return __clc_get_image_height_3d(image);
}
