//===-- Shared cbrtf function -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_SHARED_MATH_CBRTF_H
#define LIBC_SHARED_MATH_CBRTF_H

#include "shared/libc_common.h"
#include "src/__support/math/cbrtf.h"

namespace LIBC_NAMESPACE_DECL {
namespace shared {

using math::cbrtf;

} // namespace shared
} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_SHARED_MATH_CBRTF_H
