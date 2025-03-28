//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>

_CLC_OVERLOAD _CLC_DEF int2 get_image_dim (image2d_t image) {
  return (int2)(get_image_width(image), get_image_height(image));
}
_CLC_OVERLOAD _CLC_DEF int4 get_image_dim (image3d_t image) {
  return (int4)(get_image_width(image), get_image_height(image),
                get_image_depth(image), 0);
}
