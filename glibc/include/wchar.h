#ifndef _WCHAR_H

/* Workaround PR90731 with GCC 9 when using ldbl redirects in C++.  */
# include <bits/floatn.h>
# if defined __cplusplus && __LDOUBLE_REDIRECTS_TO_FLOAT128_ABI == 1
#  if __GNUC_PREREQ (9, 0) && !__GNUC_PREREQ (9, 3)
#   pragma GCC system_header
#  endif
# endif

# include <wcsmbs/wchar.h>
# ifndef _ISOMAC

#include <bits/floatn.h>

extern __typeof (wcscasecmp_l) __wcscasecmp_l;
extern __typeof (wcsncasecmp_l) __wcsncasecmp_l;
extern __typeof (wcscoll_l) __wcscoll_l;
extern __typeof (wcsxfrm_l) __wcsxfrm_l;
extern __typeof (wcstol_l) __wcstol_l;
extern __typeof (wcstoul_l) __wcstoul_l;
extern __typeof (wcstoll_l) __wcstoll_l;
extern __typeof (wcstoull_l) __wcstoull_l;
extern __typeof (wcstod_l) __wcstod_l;
extern __typeof (wcstof_l) __wcstof_l;
extern __typeof (wcstold_l) __wcstold_l;
extern __typeof (wcsftime_l) __wcsftime_l;
libc_hidden_proto (__wcstol_l)
libc_hidden_proto (__wcstoul_l)
libc_hidden_proto (__wcstoll_l)
libc_hidden_proto (__wcstoull_l)
libc_hidden_proto (__wcstod_l)
libc_hidden_proto (__wcstof_l)
libc_hidden_proto (__wcstold_l)
libc_hidden_proto (__wcsftime_l)


extern double __wcstod_internal (const wchar_t *__restrict __nptr,
				 wchar_t **__restrict __endptr, int __group)
     __THROW;
extern float __wcstof_internal (const wchar_t *__restrict __nptr,
				wchar_t **__restrict __endptr, int __group)
     __THROW;
extern long double __wcstold_internal (const wchar_t *__restrict __nptr,
				       wchar_t **__restrict __endptr,
				       int __group) __THROW;
extern long int __wcstol_internal (const wchar_t *__restrict __nptr,
				   wchar_t **__restrict __endptr,
				   int __base, int __group) __THROW;
extern unsigned long int __wcstoul_internal (const wchar_t *__restrict __npt,
					     wchar_t **__restrict __endptr,
					     int __base, int __group) __THROW;
__extension__
extern long long int __wcstoll_internal (const wchar_t *__restrict __nptr,
					 wchar_t **__restrict __endptr,
					 int __base, int __group) __THROW;
__extension__
extern unsigned long long int __wcstoull_internal (const wchar_t *
						   __restrict __nptr,
						   wchar_t **
						   __restrict __endptr,
						   int __base,
						   int __group) __THROW;
extern unsigned long long int ____wcstoull_l_internal (const wchar_t *,
						       wchar_t **, int, int,
						       locale_t);
libc_hidden_proto (__wcstof_internal)
libc_hidden_proto (__wcstod_internal)
libc_hidden_proto (__wcstold_internal)
libc_hidden_proto (__wcstol_internal)
libc_hidden_proto (__wcstoll_internal)
libc_hidden_proto (__wcstoul_internal)
libc_hidden_proto (__wcstoull_internal)
libc_hidden_proto (wcstof)
libc_hidden_proto (wcstod)
libc_hidden_ldbl_proto (wcstold)
libc_hidden_proto (wcstol)
libc_hidden_proto (wcstoll)
libc_hidden_proto (wcstoul)
libc_hidden_proto (wcstoull)

extern float ____wcstof_l_internal (const wchar_t *, wchar_t **, int,
				    locale_t) attribute_hidden;
extern double ____wcstod_l_internal (const wchar_t *, wchar_t **, int,
				     locale_t) attribute_hidden;
extern long double ____wcstold_l_internal (const wchar_t *, wchar_t **,
					   int, locale_t) attribute_hidden;
extern long int ____wcstol_l_internal (const wchar_t *, wchar_t **, int,
				       int, locale_t) attribute_hidden;
extern unsigned long int ____wcstoul_l_internal (const wchar_t *,
						 wchar_t **,
						 int, int, locale_t)
     attribute_hidden;
extern long long int ____wcstoll_l_internal (const wchar_t *, wchar_t **,
					     int, int, locale_t)
     attribute_hidden;
extern unsigned long long int ____wcstoull_l_internal (const wchar_t *,
						       wchar_t **, int, int,
						       locale_t)
     attribute_hidden;

#if __HAVE_DISTINCT_FLOAT128
extern __typeof (wcstof128_l) __wcstof128_l;
libc_hidden_proto (__wcstof128_l)
extern _Float128 __wcstof128_internal (const wchar_t *__restrict __nptr,
				       wchar_t **__restrict __endptr,
				       int __group) __THROW;

extern _Float128 ____wcstof128_l_internal (const wchar_t *, wchar_t **, int,
					   locale_t) attribute_hidden;

libc_hidden_proto (__wcstof128_internal)
libc_hidden_proto (wcstof128)
#endif

libc_hidden_proto (__wcscasecmp_l)
libc_hidden_proto (__wcsncasecmp_l)

libc_hidden_proto (__wcscoll_l)
libc_hidden_proto (__wcsxfrm_l)

libc_hidden_proto (fputws_unlocked)
libc_hidden_proto (putwc_unlocked)
libc_hidden_proto (putwc)

libc_hidden_proto (mbrtowc)
libc_hidden_proto (wcrtomb)
extern int __wcscmp (const wchar_t *__s1, const wchar_t *__s2)
     __THROW __attribute_pure__;
libc_hidden_proto (__wcscmp)
libc_hidden_proto (wcsftime)
libc_hidden_proto (wcsspn)
libc_hidden_proto (wcschr)
/* The C++ overloading of wcschr means we have to repeat the type to
   declare __wcschr instead of using typeof, to avoid errors in C++
   tests; in addition, __THROW cannot be used with a function type
   from typeof in C++.  The same applies to __wmemchr and, as regards
   __THROW, to __wcscmp and __wcscoll.  */
extern wchar_t *__wcschr (const wchar_t *__wcs, wchar_t __wc)
     __THROW __attribute_pure__;
libc_hidden_proto (__wcschr)
extern int __wcscoll (const wchar_t *__s1, const wchar_t *__s2) __THROW;
libc_hidden_proto (__wcscoll)
libc_hidden_proto (wcspbrk)

extern __typeof (wmemset) __wmemset;
extern wchar_t *__wmemchr (const wchar_t *__s, wchar_t __c, size_t __n)
     __THROW __attribute_pure__;
libc_hidden_proto (wmemchr)
libc_hidden_proto (__wmemchr)
libc_hidden_proto (wmemset)
libc_hidden_proto (__wmemset)
extern int __wmemcmp (const wchar_t *__s1, const wchar_t *__s2, size_t __n)
     __THROW __attribute_pure__;

/* Now define the internal interfaces.  */
extern int __wcscasecmp (const wchar_t *__s1, const wchar_t *__s2)
     __attribute_pure__;
extern int __wcsncasecmp (const wchar_t *__s1, const wchar_t *__s2,
			  size_t __n)
     __attribute_pure__;
extern size_t __wcslen (const wchar_t *__s) __attribute_pure__;
extern size_t __wcsnlen (const wchar_t *__s, size_t __maxlen)
     __attribute_pure__;
extern wchar_t *__wcscat (wchar_t *dest, const wchar_t *src);
extern wint_t __btowc (int __c) attribute_hidden;
extern int __mbsinit (const __mbstate_t *__ps);
extern size_t __mbrtowc (wchar_t *__restrict __pwc,
			 const char *__restrict __s, size_t __n,
			 __mbstate_t *__restrict __p);
libc_hidden_proto (__mbrtowc)
libc_hidden_proto (__mbrlen)
extern size_t __wcrtomb (char *__restrict __s, wchar_t __wc,
			 __mbstate_t *__restrict __ps) attribute_hidden;
extern size_t __mbsrtowcs (wchar_t *__restrict __dst,
			   const char **__restrict __src,
			   size_t __len, __mbstate_t *__restrict __ps)
     attribute_hidden;
extern size_t __wcsrtombs (char *__restrict __dst,
			   const wchar_t **__restrict __src,
			   size_t __len, __mbstate_t *__restrict __ps)
     attribute_hidden;
extern size_t __mbsnrtowcs (wchar_t *__restrict __dst,
			    const char **__restrict __src, size_t __nmc,
			    size_t __len, __mbstate_t *__restrict __ps)
     attribute_hidden;
extern size_t __wcsnrtombs (char *__restrict __dst,
			    const wchar_t **__restrict __src,
			    size_t __nwc, size_t __len,
			    __mbstate_t *__restrict __ps)
     attribute_hidden;
extern wchar_t *__wcscpy (wchar_t *__restrict __dest,
			  const wchar_t *__restrict __src)
			  attribute_hidden __nonnull ((1, 2));
libc_hidden_proto (__wcscpy)
extern wchar_t *__wcsncpy (wchar_t *__restrict __dest,
			   const wchar_t *__restrict __src, size_t __n);
extern wchar_t *__wcpcpy (wchar_t *__dest, const wchar_t *__src);
extern wchar_t *__wcpncpy (wchar_t *__dest, const wchar_t *__src,
			   size_t __n);
extern wchar_t *__wmemcpy (wchar_t *__s1, const wchar_t *s2,
			   size_t __n) attribute_hidden;
extern wchar_t *__wmempcpy (wchar_t *__restrict __s1,
			    const wchar_t *__restrict __s2,
			    size_t __n) attribute_hidden;
extern wchar_t *__wmemmove (wchar_t *__s1, const wchar_t *__s2,
			    size_t __n) attribute_hidden;
extern wchar_t *__wcschrnul (const wchar_t *__s, wchar_t __wc)
     __attribute_pure__;

extern wchar_t *__wmemset_chk (wchar_t *__s, wchar_t __c, size_t __n,
			       size_t __ns) __THROW;

extern int __vfwscanf (__FILE *__restrict __s,
		       const wchar_t *__restrict __format,
		       __gnuc_va_list __arg)
     attribute_hidden
     /* __attribute__ ((__format__ (__wscanf__, 2, 0)) */;
extern int __fwprintf (__FILE *__restrict __s,
		       const wchar_t *__restrict __format, ...)
     attribute_hidden
     /* __attribute__ ((__format__ (__wprintf__, 2, 3))) */;
extern int __vfwprintf_chk (FILE *__restrict __s, int __flag,
			    const wchar_t *__restrict __format,
			    __gnuc_va_list __arg)
     /* __attribute__ ((__format__ (__wprintf__, 3, 0))) */;
extern int __vswprintf_chk (wchar_t *__restrict __s, size_t __n,
			    int __flag, size_t __s_len,
			    const wchar_t *__restrict __format,
			    __gnuc_va_list __arg)
     /* __attribute__ ((__format__ (__wprintf__, 5, 0))) */;

extern int __isoc99_fwscanf (__FILE *__restrict __stream,
			     const wchar_t *__restrict __format, ...);
extern int __isoc99_wscanf (const wchar_t *__restrict __format, ...);
extern int __isoc99_swscanf (const wchar_t *__restrict __s,
			     const wchar_t *__restrict __format, ...)
     __THROW;
extern int __isoc99_vfwscanf (__FILE *__restrict __s,
			      const wchar_t *__restrict __format,
			      __gnuc_va_list __arg);
extern int __isoc99_vwscanf (const wchar_t *__restrict __format,
			     __gnuc_va_list __arg);
extern int __isoc99_vswscanf (const wchar_t *__restrict __s,
			      const wchar_t *__restrict __format,
			      __gnuc_va_list __arg) __THROW;
libc_hidden_proto (__isoc99_vswscanf)
libc_hidden_proto (__isoc99_vfwscanf)

/* Internal functions.  */
extern size_t __mbsrtowcs_l (wchar_t *dst, const char **src, size_t len,
			     mbstate_t *ps, locale_t l) attribute_hidden;

/* Special version.  We know that all uses of mbsinit inside the libc
   have a non-NULL parameter.  And certainly we can access the
   internals of the data structure directly.  */
#  define mbsinit(state) ((state)->__count == 0)
#  define __mbsinit(state) ((state)->__count == 0)

# endif
#endif
