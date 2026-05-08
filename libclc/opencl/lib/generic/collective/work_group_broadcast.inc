//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__CLC_SCALAR) && (defined(__CLC_FPSIZE) || __CLC_GENSIZE >= 32)

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE work_group_broadcast(__CLC_GENTYPE a,
                                                          size_t local_id_x) {
  return __clc_work_group_broadcast(a, local_id_x);
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE work_group_broadcast(__CLC_GENTYPE a,
                                                          size_t local_id_x,
                                                          size_t local_id_y) {
  return __clc_work_group_broadcast(a, local_id_x, local_id_y);
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE work_group_broadcast(__CLC_GENTYPE a,
                                                          size_t local_id_x,
                                                          size_t local_id_y,
                                                          size_t local_id_z) {
  return __clc_work_group_broadcast(a, local_id_x, local_id_y, local_id_z);
}

#endif
