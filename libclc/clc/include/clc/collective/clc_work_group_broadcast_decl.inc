//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE
__clc_work_group_broadcast(__CLC_GENTYPE a, size_t local_id_x);

_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE __clc_work_group_broadcast(
    __CLC_GENTYPE a, size_t local_id_x, size_t local_id_y);

_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE __clc_work_group_broadcast(
    __CLC_GENTYPE a, size_t local_id_x, size_t local_id_y, size_t local_id_z);
