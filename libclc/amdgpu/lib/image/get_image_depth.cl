#include <clc/clc.h>

_CLC_DECL int __clc_get_image_depth_3d(image3d_t);

_CLC_OVERLOAD _CLC_DEF int
get_image_depth(image3d_t image) {
	return __clc_get_image_depth_3d(image);
}
