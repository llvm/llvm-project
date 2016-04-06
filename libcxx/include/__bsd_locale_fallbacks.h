// -*- C++ -*-
//===---------------------- __bsd_locale_fallbacks.h ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// The BSDs have lots of *_l functions.  This file provides reimplementations
// of those functions for non-BSD platforms.
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_BSD_LOCALE_FALLBACKS_DEFAULTS_H
#define _LIBCPP_BSD_LOCALE_FALLBACKS_DEFAULTS_H

#include <stdlib.h>
#include <memory>

_LIBCPP_BEGIN_NAMESPACE_STD

typedef _VSTD::remove_pointer<locale_t>::type __use_locale_struct;
typedef _VSTD::unique_ptr<__use_locale_struct, decltype(&uselocale)> __locale_raii;

inline _LIBCPP_ALWAYS_INLINE
decltype(MB_CUR_MAX) __libcpp_mb_cur_max_l(locale_t __l)
{
    __locale_raii __current( uselocale(__l), uselocale );
    return MB_CUR_MAX;
}

inline _LIBCPP_ALWAYS_INLINE
wint_t __libcpp_btowc_l(int __c, locale_t __l)
{
    __locale_raii __current( uselocale(__l), uselocale );
    return btowc(__c);
}

inline _LIBCPP_ALWAYS_INLINE
int __libcpp_wctob_l(wint_t __c, locale_t __l)
{
    __locale_raii __current( uselocale(__l), uselocale );
    return wctob(__c);
}

inline _LIBCPP_ALWAYS_INLINE
size_t __libcpp_wcsnrtombs_l(char *__dest, const wchar_t **__src, size_t __nwc,
                         size_t __len, mbstate_t *__ps, locale_t __l)
{
    __locale_raii __current( uselocale(__l), uselocale );
    return wcsnrtombs(__dest, __src, __nwc, __len, __ps);
}

inline _LIBCPP_ALWAYS_INLINE
size_t __libcpp_wcrtomb_l(char *__s, wchar_t __wc, mbstate_t *__ps, locale_t __l)
{
    __locale_raii __current( uselocale(__l), uselocale );
    return wcrtomb(__s, __wc, __ps);
}

inline _LIBCPP_ALWAYS_INLINE
size_t __libcpp_mbsnrtowcs_l(wchar_t * __dest, const char **__src, size_t __nms,
                      size_t __len, mbstate_t *__ps, locale_t __l)
{
    __locale_raii __current( uselocale(__l), uselocale );
    return mbsnrtowcs(__dest, __src, __nms, __len, __ps);
}

inline _LIBCPP_ALWAYS_INLINE
size_t __libcpp_mbrtowc_l(wchar_t *__pwc, const char *__s, size_t __n,
                   mbstate_t *__ps, locale_t __l)
{
    __locale_raii __current( uselocale(__l), uselocale );
    return mbrtowc(__pwc, __s, __n, __ps);
}

inline _LIBCPP_ALWAYS_INLINE
int __libcpp_mbtowc_l(wchar_t *__pwc, const char *__pmb, size_t __max, locale_t __l)
{
    __locale_raii __current( uselocale(__l), uselocale );
    return mbtowc(__pwc, __pmb, __max);
}

inline _LIBCPP_ALWAYS_INLINE
size_t __libcpp_mbrlen_l(const char *__s, size_t __n, mbstate_t *__ps, locale_t __l)
{
    __locale_raii __current( uselocale(__l), uselocale );
    return mbrlen(__s, __n, __ps);
}

inline _LIBCPP_ALWAYS_INLINE
lconv *__libcpp_localeconv_l(locale_t __l)
{
    __locale_raii __current( uselocale(__l), uselocale );
    return localeconv();
}

inline _LIBCPP_ALWAYS_INLINE
size_t __libcpp_mbsrtowcs_l(wchar_t *__dest, const char **__src, size_t __len,
                     mbstate_t *__ps, locale_t __l)
{
    __locale_raii __current( uselocale(__l), uselocale );
    return mbsrtowcs(__dest, __src, __len, __ps);
}

inline
int __libcpp_snprintf_l(char *__s, size_t __n, locale_t __l, const char *__format, ...) {
    va_list __va;
    va_start(__va, __format);
    __locale_raii __current( uselocale(__l), uselocale );
    int __res = vsnprintf(__s, __n, __format, __va);
    va_end(__va);
    return __res;
}

inline
int __libcpp_asprintf_l(char **__s, locale_t __l, const char *__format, ...) {
    va_list __va;
    va_start(__va, __format);
    __locale_raii __current( uselocale(__l), uselocale );
    int __res = vasprintf(__s, __format, __va);
    va_end(__va);
    return __res;
}

inline
int __libcpp_sscanf_l(const char *__s, locale_t __l, const char *__format, ...) {
    va_list __va;
    va_start(__va, __format);
    __locale_raii __current( uselocale(__l), uselocale );
    int __res = vsscanf(__s, __format, __va);
    va_end(__va);
    return __res;
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_BSD_LOCALE_FALLBACKS_DEFAULTS_H
