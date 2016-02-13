#include <clc/clc.h>

_CLC_DECL int __clc_get_image_channel_data_type_2d(image2d_t);
_CLC_DECL int __clc_get_image_channel_data_type_3d(image3d_t);

_CLC_OVERLOAD _CLC_DEF int
get_image_channel_data_type(image2d_t image) {
  return __clc_get_image_channel_data_type_2d(image);
}
_CLC_OVERLOAD _CLC_DEF int
get_image_channel_data_type(image3d_t image) {
  return __clc_get_image_channel_data_type_3d(image);
}
