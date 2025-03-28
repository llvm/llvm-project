//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>

_CLC_DECL int __clc_get_image_depth_3d(image3d_t);

_CLC_OVERLOAD _CLC_DEF int
get_image_depth(image3d_t image) {
	return __clc_get_image_depth_3d(image);
}
