// -*- C++ -*-
//===--------------------- support/win32/locale_win32.h -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_SUPPORT_WIN32_LOCALE_WIN32_H
#define _LIBCPP_SUPPORT_WIN32_LOCALE_WIN32_H

// ctype mask table defined in msvcrt.dll
extern "C" unsigned short  __declspec(dllimport) _ctype[];

#include "support/win32/support.h"
#include <memory>
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
lconv *localeconv_l( locale_t loc );
size_t mbrlen_l( const char *__restrict__ s, size_t n,
                 mbstate_t *__restrict__ ps, locale_t loc);
size_t mbsrtowcs_l( wchar_t *__restrict__ dst, const char **__restrict__ src,
                    size_t len, mbstate_t *__restrict__ ps, locale_t loc );
size_t wcrtomb_l( char *__restrict__ s, wchar_t wc, mbstate_t *__restrict__ ps,
                  locale_t loc);
size_t mbrtowc_l( wchar_t *__restrict__ pwc, const char *__restrict__ s,
                  size_t n, mbstate_t *__restrict__ ps, locale_t loc);
size_t mbsnrtowcs_l( wchar_t *__restrict__ dst, const char **__restrict__ src,
                     size_t nms, size_t len, mbstate_t *__restrict__ ps, locale_t loc);
size_t wcsnrtombs_l( char *__restrict__ dst, const wchar_t **__restrict__ src,
                     size_t nwc, size_t len, mbstate_t *__restrict__ ps, locale_t loc);
wint_t btowc_l( int c, locale_t loc );
int wctob_l( wint_t c, locale_t loc );
typedef _VSTD::remove_pointer<locale_t>::type __locale_struct;
typedef _VSTD::unique_ptr<__locale_struct, decltype(&uselocale)> __locale_raii;
_LIBCPP_ALWAYS_INLINE inline
decltype(MB_CUR_MAX) MB_CUR_MAX_L( locale_t __l )
{
  __locale_raii __current( uselocale(__l), uselocale );
  return MB_CUR_MAX;
}

// the *_l functions are prefixed on Windows, only available for msvcr80+, VS2005+
#include <stdio.h>
#define mbtowc_l _mbtowc_l
#define strtoll_l _strtoi64_l
#define strtoull_l _strtoui64_l
// FIXME: current msvcrt does not know about long double
#define strtold_l _strtod_l
#define islower_l _islower_l
#define isupper_l _isupper_l
#define isdigit_l _isdigit_l
#define isxdigit_l _isxdigit_l
#define strcoll_l _strcoll_l
#define strxfrm_l _strxfrm_l
#define wcscoll_l _wcscoll_l
#define wcsxfrm_l _wcsxfrm_l
#define toupper_l _toupper_l
#define tolower_l _tolower_l
#define iswspace_l _iswspace_l
#define iswprint_l _iswprint_l
#define iswcntrl_l _iswcntrl_l
#define iswupper_l _iswupper_l
#define iswlower_l _iswlower_l
#define iswalpha_l _iswalpha_l
#define iswdigit_l _iswdigit_l
#define iswpunct_l _iswpunct_l
#define iswxdigit_l _iswxdigit_l
#define towupper_l _towupper_l
#define towlower_l _towlower_l
#define strftime_l _strftime_l
#define sscanf_l( __s, __l, __f, ...) _sscanf_l( __s, __f, __l, __VA_ARGS__ )
#define vsscanf_l( __s, __l, __f, ...) _sscanf_l( __s, __f, __l, __VA_ARGS__ )
#define sprintf_l( __s, __l, __f, ... ) _sprintf_l( __s, __f, __l, __VA_ARGS__ )
#define snprintf_l( __s, __n, __l, __f, ... ) _snprintf_l( __s, __n, __f, __l, __VA_ARGS__ )
#define vsprintf_l( __s, __l, __f, ... ) _vsprintf_l( __s, __f, __l, __VA_ARGS__ )
#define vsnprintf_l( __s, __n, __l, __f, ... ) _vsnprintf_l( __s, __n, __f, __l, __VA_ARGS__ )
int asprintf_l( char **ret, locale_t loc, const char *format, ... );
int vasprintf_l( char **ret, locale_t loc, const char *format, va_list ap );


// not-so-pressing FIXME: use locale to determine blank characters
inline int isblank_l( int c, locale_t /*loc*/ )
{
    return ( c == ' ' || c == '\t' );
}
inline int iswblank_l( wint_t c, locale_t /*loc*/ )
{
    return ( c == L' ' || c == L'\t' );
}

#endif // _LIBCPP_SUPPORT_WIN32_LOCALE_WIN32_H
