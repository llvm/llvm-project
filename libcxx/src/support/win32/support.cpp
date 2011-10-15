// -*- C++ -*-
//===----------------------- support/win32/support.h ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <support/win32/support.h>
#include <stdarg.h> // va_start, va_end
#include <stddef.h> // size_t
#include <stdlib.h> // malloc
#include <stdio.h>  // vsprintf, vsnprintf
#include <string.h> // strcpy, wcsncpy

int asprintf(char **sptr, const char *__restrict__ fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int result = vasprintf(sptr, fmt, ap);
    va_end(ap);
	return result;
}
int vasprintf( char **sptr, const char *__restrict__ fmt, va_list ap )
{
    *sptr = NULL;
    int count = vsnprintf( *sptr, 0, fmt, ap );
    if( (count >= 0) && ((*sptr = (char*)malloc(count+1)) != NULL) )
    {
        vsprintf( *sptr, fmt, ap );
        sptr[count] = '\0';
    }

    return count;
}

// FIXME: use wcrtomb and avoid copy
// use mbsrtowcs which is available, first copy first nwc elements of src
size_t mbsnrtowcs( wchar_t *__restrict__ dst, const char **__restrict__ src,
                   size_t nmc, size_t len, mbstate_t *__restrict__ ps )
{
    char* local_src = new char[nmc+1];
    char* nmcsrc = local_src;
    strncpy( nmcsrc, *src, nmc );
    nmcsrc[nmc] = '\0';
    const size_t result = mbsrtowcs( dst, const_cast<const char **>(&nmcsrc), len, ps );
    // propagate error
    if( nmcsrc == NULL )
        *src = NULL;
    delete[] local_src;
    return result;
}
// FIXME: use wcrtomb and avoid copy
// use wcsrtombs which is available, first copy first nwc elements of src
size_t wcsnrtombs( char *__restrict__ dst, const wchar_t **__restrict__ src,
                   size_t nwc, size_t len, mbstate_t *__restrict__ ps )
{
    wchar_t* local_src = new wchar_t[nwc];
    wchar_t* nwcsrc = local_src;
    wcsncpy(nwcsrc, *src, nwc);
    nwcsrc[nwc] = '\0';
    const size_t result = wcsrtombs( dst, const_cast<const wchar_t **>(&nwcsrc), len, ps );
    // propogate error
    if( nwcsrc == NULL )
        *src = NULL;
    delete[] nwcsrc;
    return result;
}
