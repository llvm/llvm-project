//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/collective/clc_work_group_any_all.h"

_CLC_OVERLOAD _CLC_DEF int work_group_all(int predicate) {
  return __clc_work_group_all(predicate);
}

_CLC_OVERLOAD _CLC_DEF int work_group_any(int predicate) {
  return __clc_work_group_any(predicate);
}
