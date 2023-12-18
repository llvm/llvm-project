//===-- Implementation of math utils --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "math_utils.h"

namespace LIBC_NAMESPACE {

constexpr float XFlowValues<float>::OVERFLOW_VALUE = 0x1p97f;
constexpr float XFlowValues<float>::UNDERFLOW_VALUE = 0x1p-95f;
constexpr float XFlowValues<float>::MAY_UNDERFLOW_VALUE = 0x1.4p-75f;

constexpr double XFlowValues<double>::OVERFLOW_VALUE = 0x1p769;
constexpr double XFlowValues<double>::UNDERFLOW_VALUE = 0x1p-767;
constexpr double XFlowValues<double>::MAY_UNDERFLOW_VALUE = 0x1.8p-538;

} // namespace LIBC_NAMESPACE
