//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_COLLECTIVE_CLC_WORK_GROUP_ANY_ALL_H__
#define __CLC_COLLECTIVE_CLC_WORK_GROUP_ANY_ALL_H__

#include "clc/internal/clc.h"

_CLC_OVERLOAD _CLC_DECL int __clc_work_group_any(int predicate);
_CLC_OVERLOAD _CLC_DECL int __clc_work_group_all(int predicate);

#endif // __CLC_COLLECTIVE_CLC_WORK_GROUP_ANY_ALL_H__
