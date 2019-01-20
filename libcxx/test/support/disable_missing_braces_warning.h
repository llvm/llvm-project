//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SUPPORT_DISABLE_MISSING_BRACES_WARNING_H
#define SUPPORT_DISABLE_MISSING_BRACES_WARNING_H

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wmissing-braces"
#elif defined(__clang__)
#pragma clang diagnostic ignored "-Wmissing-braces"
#endif

#endif // SUPPORT_DISABLE_MISSING_BRACES_WARNING_H
