// -*- C++ -*-
//===----------------------- support/win32/support.h ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_SUPPORT_WIN32_SUPPORT_H
#define _LIBCPP_SUPPORT_WIN32_SUPPORT_H

/*
   Functions and constants used in libc++ that are missing from the Windows C library.
  */

#include <wchar.h>  // mbstate_t
#include <stdio.h> // _snwprintf
#define swprintf _snwprintf
#define vswprintf _vsnwprintf

int vasprintf( char **sptr, const char *__restrict__ fmt , va_list ap );
int asprintf(char **sptr, const char *__restrict__ fmt, ...);

size_t mbsnrtowcs( wchar_t *__restrict__ dst, const char **__restrict__ src,
                   size_t nmc, size_t len, mbstate_t *__restrict__ ps );
size_t wcsnrtombs( char *__restrict__ dst, const wchar_t **__restrict__ src,
                   size_t nwc, size_t len, mbstate_t *__restrict__ ps );

#endif // _LIBCPP_SUPPORT_WIN32_SUPPORT_H