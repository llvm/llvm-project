//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF event_t async_work_group_copy(
    local __CLC_GENTYPE *dst,
    const global __CLC_GENTYPE *src,
    size_t num_gentypes,
    event_t event) {

  return async_work_group_strided_copy(dst, src, num_gentypes, 1, event);
}

_CLC_OVERLOAD _CLC_DEF event_t async_work_group_copy(
    global __CLC_GENTYPE *dst,
    const local __CLC_GENTYPE *src,
    size_t num_gentypes,
    event_t event) {

  return async_work_group_strided_copy(dst, src, num_gentypes, 1, event);
}
