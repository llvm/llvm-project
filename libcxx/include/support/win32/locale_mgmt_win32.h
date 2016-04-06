// -*- C++ -*-
//===----------------- support/win32/locale_mgmt_win32.h ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_SUPPORT_WIN32_LOCALE_MGMT_WIN32_H
#define _LIBCPP_SUPPORT_WIN32_LOCALE_MGMT_WIN32_H

#include <xlocinfo.h> // _locale_t
#define locale_t _locale_t
#define LC_COLLATE_MASK _M_COLLATE
#define LC_CTYPE_MASK _M_CTYPE
#define LC_MONETARY_MASK _M_MONETARY
#define LC_NUMERIC_MASK _M_NUMERIC
#define LC_TIME_MASK _M_TIME
#define LC_MESSAGES_MASK _M_MESSAGES
#define LC_ALL_MASK (  LC_COLLATE_MASK \
                     | LC_CTYPE_MASK \
                     | LC_MESSAGES_MASK \
                     | LC_MONETARY_MASK \
                     | LC_NUMERIC_MASK \
                     | LC_TIME_MASK )
#define freelocale _free_locale
// FIXME: base currently unused. Needs manual work to construct the new locale
locale_t newlocale( int mask, const char * locale, locale_t base );
locale_t uselocale( locale_t newloc );

#endif // _LIBCPP_SUPPORT_WIN32_LOCALE_MGMT_WIN32_H
