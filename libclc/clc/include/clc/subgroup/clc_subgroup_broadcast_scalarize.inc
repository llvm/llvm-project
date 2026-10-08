//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_VECSIZE_OR_1 >= 2
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__clc_sub_group_broadcast(__CLC_GENTYPE x, uint sub_group_local_id) {
  union {
    __CLC_GENTYPE vec;
    __CLC_SCALAR_GENTYPE arr[__CLC_VECSIZE_OR_1];
  } u_x, u_result;
  u_x.vec = x;
  for (int i = 0; i < __CLC_VECSIZE_OR_1; ++i)
    u_result.arr[i] = __clc_sub_group_broadcast(u_x.arr[i], sub_group_local_id);
  return u_result.vec;
}
#endif
