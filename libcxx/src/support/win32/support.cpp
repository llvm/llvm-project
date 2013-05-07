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

int asprintf(char **sptr, const char *__restrict fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int result = vasprintf(sptr, fmt, ap);
    va_end(ap);
    return result;
}

// Like sprintf, but when return value >= 0 it returns a pointer to a malloc'd string in *sptr.
// If return >= 0, use free to delete *sptr.
int vasprintf( char **sptr, const char *__restrict fmt, va_list ap )
{
    *sptr = NULL;
    int count = vsnprintf( NULL, 0, fmt, ap ); // Query the buffer size required.
    if( count >= 0 ) {
        char* p = static_cast<char*>(malloc(count+1)); // Allocate memory for it and the terminator.
        if ( p == NULL )
            return -1;
        if ( vsnprintf( p, count+1, fmt, ap ) == count ) // We should have used exactly what was required.
            *sptr = p;
        else { // Otherwise something is wrong, likely a bug in vsnprintf. If so free the memory and report the error.
            free(p);
            return -1;
        }
    }

    return count;
}

// Returns >= 0: the number of wide characters found in the multi byte sequence src (of src_size_bytes),
// that fit in the buffer dst (of max_dest_chars elements size). The count returned excludes the null terminator.
// When dst is NULL, no characters are copied and no "out" parameters are updated.
// Returns (size_t) -1: an incomplete sequence encountered.
// Leaves *src pointing the next character to convert or NULL if a null character was converted from *src.
size_t mbsnrtowcs( wchar_t *__restrict dst, const char **__restrict src,
                   size_t src_size_bytes, size_t max_dest_chars, mbstate_t *__restrict ps )
{
    const size_t terminated_sequence = static_cast<size_t>(0);
    //const size_t invalid_sequence = static_cast<size_t>(-1);
    const size_t incomplete_sequence = static_cast< size_t>(-2);

    size_t dest_converted = 0;
    size_t source_converted = 0;
    size_t source_remaining = src_size_bytes;
    size_t result = 0;
    bool have_result = false;

    while ( source_remaining ) {
        if ( dst && dest_converted >= max_dest_chars )
            break;
        // Converts one multi byte character.
        // if result > 0, it's the size in bytes of that character.
        // othewise if result is zero it indicates the null character has been found.
        // otherwise it's an error and errno may be set.
        size_t char_size = mbrtowc( dst ? dst + dest_converted : NULL, *src + source_converted, source_remaining, ps );
        // Don't do anything to change errno from here on.
        if ( char_size > 0 ) {
            source_remaining -= char_size;
            source_converted += char_size;
            ++dest_converted;
            continue;
        }
        result = char_size;
        have_result = true;
        break;
    }
    if ( dst ) {
        if ( have_result && result == terminated_sequence )
            *src = NULL;
        else
            *src += source_converted;
    }
    if ( have_result && result != terminated_sequence && result != incomplete_sequence )
        return static_cast<size_t>(-1);

    return dest_converted;
}

// Converts max_source_chars from the wide character buffer pointer to by *src,
// into the multi byte character sequence buffer stored at dst which must be dst_size_bytes bytes in size.
// Returns >= 0: the number of bytes in the sequence sequence converted frome *src, excluding the null terminator.
// Returns size_t(-1) if an error occurs, also sets errno.
// If dst is NULL dst_size_bytes is ignored and no bytes are copied to dst and no "out" parameters are updated.
size_t wcsnrtombs( char *__restrict dst, const wchar_t **__restrict src,
                   size_t max_source_chars, size_t dst_size_bytes, mbstate_t *__restrict ps )
{
    //const size_t invalid_sequence = static_cast<size_t>(-1);

    size_t source_converted = 0;
    size_t dest_converted = 0;
    size_t dest_remaining = dst_size_bytes;
    size_t char_size = 0;
    const errno_t no_error = ( errno_t) 0;
    errno_t result = ( errno_t ) 0;
    bool have_result = false;
    bool terminator_found = false;

    while ( source_converted != max_source_chars ) {
        if ( ! dest_remaining )
            break;
        wchar_t c = (*src)[source_converted];
        if ( dst )
            result = wcrtomb_s( &char_size, dst + dest_converted, dest_remaining, c, ps);
        else
            result = wcrtomb_s( &char_size, NULL, 0, c, ps); 
        // If result is zero there is no error and char_size contains the size of the multi-byte-sequence converted.
        // Otherwise result indicates an errno type error.
        if ( result == no_error ) {
            if ( c == L'\0' ) {
                terminator_found = true;
                break;
            }
            ++source_converted;
            if ( dst )
                dest_remaining -= char_size;
            dest_converted += char_size;
            continue;
        }
        have_result = true;
        break;
    }
    if ( dst ) {
        if ( terminator_found )
            *src = NULL;
        else
            *src = *src + source_converted;
    }
    if ( have_result && result != no_error ) {
        errno = result;
        return static_cast<size_t>(-1);
    }

    return dest_converted;
}
