//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DECL event_t async_work_group_strided_copy(
  __CLC_DST_ADDR_SPACE __CLC_GENTYPE *dst,
  const __CLC_SRC_ADDR_SPACE __CLC_GENTYPE *src,
  size_t num_gentypes,
  size_t stride,
  event_t event);
