//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_SUPPORT_NEWLIB_XLOCALE_H
#define _LIBCPP_SUPPORT_NEWLIB_XLOCALE_H

#if defined(_NEWLIB_VERSION)

#include <cstdlib>
#include <clocale>
#include <cwctype>
#include <ctype.h>
#include <support/xlocale/__nop_locale_mgmt.h>

#ifdef __cplusplus
extern "C" {
#endif

// Share implementation with Android's Bionic
#include <support/xlocale/xlocale.h>

#ifdef __cplusplus
} // extern "C"
#endif

#endif // _NEWLIB_VERSION

#endif
