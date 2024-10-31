//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_DIR_SUPPORT_APPLE_H
#define _LIBCPP___LOCALE_DIR_SUPPORT_APPLE_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#include <__locale_dir/support/management/bsd_like.h> // must come first since it defines __locale_t

#include <__locale_dir/support/characters/bsd_like.h>
#include <__locale_dir/support/other/bsd_like.h>
#include <__locale_dir/support/strtonum/bsd_like.h>

#endif // _LIBCPP___LOCALE_DIR_SUPPORT_APPLE_H
