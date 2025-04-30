sinclude(./aclocal.m4)dnl Autoconf lossage
GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.
# Local configure fragment for sysdeps/unix/sysv/linux/powerpc/.

AC_CACHE_CHECK(whether $CC $CFLAGS -mlong-double-128 uses IBM extended format,
	       libc_cv_mlong_double_128ibm, [dnl
save_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS -mlong-double-128"
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <float.h>]], [[
#if LDBL_MANT_DIG != 106
# error "compiler doesn't implement IBM extended format of long double"
#endif
long double foobar (long double x) { return x; }]])],
	       libc_cv_mlong_double_128ibm=yes,
	       libc_cv_mlong_double_128ibm=no)
CFLAGS="$save_CFLAGS"])

if test "$libc_cv_mlong_double_128ibm" = no; then
  AC_CACHE_CHECK(whether $CC $CFLAGS supports -mabi=ibmlongdouble,
		 libc_cv_mabi_ibmlongdouble, [dnl
  save_CFLAGS="$CFLAGS"
  CFLAGS="$CFLAGS -mlong-double-128 -mabi=ibmlongdouble"
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <float.h>]], [[
#if LDBL_MANT_DIG != 106
# error "compiler doesn't implement IBM extended format of long double"
#endif
long double foobar (long double x) { return x; }]])],
		 libc_cv_mabi_ibmlongdouble=yes,
		 libc_cv_mabi_ibmlongdouble=no)
  CFLAGS="$save_CFLAGS"])

  if test "$libc_cv_mabi_ibmlongdouble" = yes; then
    CFLAGS="$CFLAGS -mabi=ibmlongdouble"
  else
    AC_MSG_ERROR([this configuration requires -mlong-double-128 IBM extended format support])
  fi
fi

ldd_rewrite_script=sysdeps/unix/sysv/linux/powerpc/ldd-rewrite.sed
