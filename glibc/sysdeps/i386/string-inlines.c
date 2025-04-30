/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <string/string-inlines.c>

#if SHLIB_COMPAT (libc, GLIBC_2_1_1, GLIBC_2_26)
/* Additional compatibility shims for the former
   sysdeps/x86/bits/string.h.  */
void *
__old_memcpy_c (void *d, const void *s, size_t n)
{
  return memcpy (d, s, n);
}
strong_alias (__old_memcpy_c, __old_memcpy_g);
strong_alias (__old_memcpy_c, __old_memcpy_by4);
strong_alias (__old_memcpy_c, __old_memcpy_by2);
compat_symbol (libc, __old_memcpy_c, __memcpy_c, GLIBC_2_1_1);
compat_symbol (libc, __old_memcpy_g, __memcpy_g, GLIBC_2_1_1);
compat_symbol (libc, __old_memcpy_by4, __memcpy_by4, GLIBC_2_1_1);
compat_symbol (libc, __old_memcpy_by2, __memcpy_by2, GLIBC_2_1_1);

void *
__old_memset_cc (void *s, unsigned long int pattern, size_t n)
{
  return memset (s, pattern & 0xff, n);
}
strong_alias (__old_memset_cc, __old_memset_cg);
strong_alias (__old_memset_cc, __old_memset_ccn_by2);
strong_alias (__old_memset_cc, __old_memset_ccn_by4);
compat_symbol (libc, __old_memset_cc, __memset_cc, GLIBC_2_1_1);
compat_symbol (libc, __old_memset_cg, __memset_cg, GLIBC_2_1_1);
compat_symbol (libc, __old_memset_ccn_by4, __memset_ccn_by4, GLIBC_2_1_1);
compat_symbol (libc, __old_memset_ccn_by2, __memset_ccn_by2, GLIBC_2_1_1);

void *
__old_memset_gg (void *s, char c, size_t n)
{
  return memset (s, c, n);
}
strong_alias (__old_memset_gg, __old_memset_gcn_by4);
strong_alias (__old_memset_gg, __old_memset_gcn_by2);
compat_symbol (libc, __old_memset_gg, __memset_gg, GLIBC_2_1_1);
compat_symbol (libc, __old_memset_gcn_by4, __memset_gcn_by4, GLIBC_2_1_1);
compat_symbol (libc, __old_memset_gcn_by2, __memset_gcn_by2, GLIBC_2_1_1);

size_t
__old_strlen_g (const char *str)
{
  return strlen (str);
}
compat_symbol (libc, __old_strlen_g, __strlen_g, GLIBC_2_1_1);

char *
__old_strcpy_g (char *dest, const char *src)
{
  return strcpy (dest, src);
}
compat_symbol (libc, __old_strcpy_g, __strcpy_g, GLIBC_2_1_1);

void *
__old_mempcpy_byn (void *dest, const void *src, size_t len)
{
  return __mempcpy (dest, src, len);
}
strong_alias (__old_mempcpy_byn, __old_mempcpy_by4);
strong_alias (__old_mempcpy_byn, __old_mempcpy_by2);
compat_symbol (libc, __old_mempcpy_byn, __mempcpy_byn, GLIBC_2_1_1);
compat_symbol (libc, __old_mempcpy_by4, __mempcpy_by4, GLIBC_2_1_1);
compat_symbol (libc, __old_mempcpy_by2, __mempcpy_by2, GLIBC_2_1_1);

char *
__old_stpcpy_g (char *dest, const char *src)
{
  return __stpcpy (dest, src);
}
compat_symbol (libc, __old_stpcpy_g, __stpcpy_g, GLIBC_2_1_1);

char *
__old_strncpy_byn (char *dest, const char *src, size_t srclen, size_t n)
{
  return strncpy (dest, src, n);
}
strong_alias (__old_strncpy_byn, __old_strncpy_by4);
strong_alias (__old_strncpy_byn, __old_strncpy_by2);
compat_symbol (libc, __old_strncpy_byn, __strncpy_byn, GLIBC_2_1_1);
compat_symbol (libc, __old_strncpy_by4, __strncpy_by4, GLIBC_2_1_1);
compat_symbol (libc, __old_strncpy_by2, __strncpy_by2, GLIBC_2_1_1);

char *
__old_strncpy_gg (char *dest, const char *src, size_t n)
{
  return strncpy (dest, src, n);
}
compat_symbol (libc, __old_strncpy_gg, __strncpy_gg, GLIBC_2_1_1);

/* __strcat_c took a third argument, which we ignore.  */
char *
__old_strcat_g (char *dest, const char *src)
{
  return strcat (dest, src);
}
strong_alias (__old_strcat_g, __old_strcat_c);
compat_symbol (libc, __old_strcat_g, __strcat_g, GLIBC_2_1_1);
compat_symbol (libc, __old_strcat_c, __strcat_c, GLIBC_2_1_1);

char *
__old_strncat_g (char *dest, const char *src, size_t n)
{
  return __strncat (dest, src, n);
}
compat_symbol (libc, __old_strncat_g, __strncat_g, GLIBC_2_1_1);

int
__old_strcmp_gg (const char *s1, const char *s2)
{
  return strcmp (s1, s2);
}
compat_symbol (libc, __old_strcmp_gg, __strcmp_gg, GLIBC_2_1_1);

int
__old_strncmp_g (const char *s1, const char *s2, size_t n)
{
  return strncmp (s1, s2, n);
}
compat_symbol (libc, __old_strncmp_g, __strncmp_g, GLIBC_2_1_1);

char *
__old_strchr_g (const char *s, int c)
{
  return strchr (s, c);
}
strong_alias (__old_strchr_g, __old_strchr_c);
compat_symbol (libc, __old_strchr_g, __strchr_g, GLIBC_2_1_1);
compat_symbol (libc, __old_strchr_c, __strchr_c, GLIBC_2_1_1);

char *
__old_strchrnul_g (const char *s, int c)
{
  return __strchrnul (s, c);
}
strong_alias (__old_strchrnul_g, __old_strchrnul_c);
compat_symbol (libc, __old_strchrnul_g, __strchrnul_g, GLIBC_2_1_1);
compat_symbol (libc, __old_strchrnul_c, __strchrnul_c, GLIBC_2_1_1);

char *
__old_strrchr_g (const char *s, int c)
{
  return strrchr (s, c);
}
strong_alias (__old_strrchr_g, __old_strrchr_c);
compat_symbol (libc, __old_strrchr_g, __strrchr_g, GLIBC_2_1_1);
compat_symbol (libc, __old_strrchr_c, __strrchr_c, GLIBC_2_1_1);

/* __strcspn_cg took a third argument, which we ignore.  */
size_t
__old_strcspn_g (const char *s, const char *reject)
{
  return strcspn (s, reject);
}
strong_alias (__old_strcspn_g, __old_strcspn_cg);
compat_symbol (libc, __old_strcspn_g, __strcspn_g, GLIBC_2_1_1);
compat_symbol (libc, __old_strcspn_cg, __strcspn_cg, GLIBC_2_1_1);

/* __strspn_cg took a third argument, which we ignore.  */
size_t
__old_strspn_g (const char *s, const char *accept)
{
  return strspn (s, accept);
}
strong_alias (__old_strspn_g, __old_strspn_cg);
compat_symbol (libc, __old_strspn_g, __strspn_g, GLIBC_2_1_1);
compat_symbol (libc, __old_strspn_cg, __strspn_cg, GLIBC_2_1_1);

/* __strpbrk_cg took a third argument, which we ignore.  */
const char *
__old_strpbrk_g (const char *s, const char *accept)
{
  return strpbrk (s, accept);
}
strong_alias (__old_strpbrk_g, __old_strpbrk_cg);
compat_symbol (libc, __old_strpbrk_g, __strpbrk_g, GLIBC_2_1_1);
compat_symbol (libc, __old_strpbrk_cg, __strpbrk_cg, GLIBC_2_1_1);

/* __strstr_cg took a third argument, which we ignore.  */
const char *
__old_strstr_g (const char *s, const char *accept)
{
  return strstr (s, accept);
}
strong_alias (__old_strstr_g, __old_strstr_cg);
compat_symbol (libc, __old_strstr_g, __strstr_g, GLIBC_2_1_1);
compat_symbol (libc, __old_strstr_cg, __strstr_cg, GLIBC_2_1_1);

#endif
