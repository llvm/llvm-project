// -*- C++ -*-
//===-------------------- support/win32/locale_win32.cpp ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <locale>
#include <cstdarg> // va_start, va_end
#include <memory>
#include <type_traits>

typedef _VSTD::remove_pointer<locale_t>::type __locale_struct;
typedef _VSTD::unique_ptr<__locale_struct, decltype(&uselocale)> __locale_raii;

// FIXME: base currently unused. Needs manual work to construct the new locale
locale_t newlocale( int mask, const char * locale, locale_t /*base*/ )
{
    return _create_locale( mask, locale );
}
locale_t uselocale( locale_t newloc )
{
    locale_t old_locale = _get_current_locale();
    if ( newloc == NULL )
        return old_locale;
    // uselocale sets the thread's locale by definition, so unconditionally use thread-local locale
    _configthreadlocale( _ENABLE_PER_THREAD_LOCALE );
    // uselocale sets all categories
    // disable setting locale on Windows temporarily because the structure is opaque (PR31516)
    //setlocale( LC_ALL, newloc->locinfo->lc_category[LC_ALL].locale );
    // uselocale returns the old locale_t
    return old_locale;
}
lconv *localeconv_l( locale_t loc )
{
    __locale_raii __current( uselocale(loc), uselocale );
    return localeconv();
}
size_t mbrlen_l( const char *__restrict s, size_t n,
                 mbstate_t *__restrict ps, locale_t loc )
{
    __locale_raii __current( uselocale(loc), uselocale );
    return mbrlen( s, n, ps );
}
size_t mbsrtowcs_l( wchar_t *__restrict dst, const char **__restrict src,
                    size_t len, mbstate_t *__restrict ps, locale_t loc )
{
    __locale_raii __current( uselocale(loc), uselocale );
    return mbsrtowcs( dst, src, len, ps );
}
size_t wcrtomb_l( char *__restrict s, wchar_t wc, mbstate_t *__restrict ps,
                  locale_t loc )
{
    __locale_raii __current( uselocale(loc), uselocale );
    return wcrtomb( s, wc, ps );
}
size_t mbrtowc_l( wchar_t *__restrict pwc, const char *__restrict s,
                  size_t n, mbstate_t *__restrict ps, locale_t loc )
{
    __locale_raii __current( uselocale(loc), uselocale );
    return mbrtowc( pwc, s, n, ps );
}
size_t mbsnrtowcs_l( wchar_t *__restrict dst, const char **__restrict src,
                     size_t nms, size_t len, mbstate_t *__restrict ps, locale_t loc )
{
    __locale_raii __current( uselocale(loc), uselocale );
    return mbsnrtowcs( dst, src, nms, len, ps );
}
size_t wcsnrtombs_l( char *__restrict dst, const wchar_t **__restrict src,
                     size_t nwc, size_t len, mbstate_t *__restrict ps, locale_t loc )
{
    __locale_raii __current( uselocale(loc), uselocale );
    return wcsnrtombs( dst, src, nwc, len, ps );
}
wint_t btowc_l( int c, locale_t loc )
{
    __locale_raii __current( uselocale(loc), uselocale );
    return btowc( c );
}
int wctob_l( wint_t c, locale_t loc )
{
    __locale_raii __current( uselocale(loc), uselocale );
    return wctob( c );
}

int snprintf_l(char *ret, size_t n, locale_t loc, const char *format, ...)
{
    __locale_raii __current( uselocale(loc), uselocale );
    va_list ap;
    va_start( ap, format );
    int result = vsnprintf( ret, n, format, ap );
    va_end(ap);
    return result;
}

int asprintf_l( char **ret, locale_t loc, const char *format, ... )
{
    va_list ap;
    va_start( ap, format );
    int result = vasprintf_l( ret, loc, format, ap );
    va_end(ap);
    return result;
}
int vasprintf_l( char **ret, locale_t loc, const char *format, va_list ap )
{
    __locale_raii __current( uselocale(loc), uselocale );
    return vasprintf( ret, format, ap );
}
