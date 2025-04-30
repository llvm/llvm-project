sinclude(./aclocal.m4)dnl Autoconf lossage
GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.
# Local configure fragment for sysdeps/ieee754/ldbl-opt/.

AC_CACHE_CHECK(whether $CC $CFLAGS supports -mlong-double-128,
	       libc_cv_mlong_double_128, [dnl
save_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS -mlong-double-128"
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]], [[
#ifndef __LONG_DOUBLE_128__
# error "compiler did not predefine __LONG_DOUBLE_128__ as expected"
#endif
long double foobar (long double x) { return x; }]])],
	       [libc_cv_mlong_double_128=yes],
	       [libc_cv_mlong_double_128=no])
CFLAGS="$save_CFLAGS"])
if test "$libc_cv_mlong_double_128" = no; then
  AC_MSG_ERROR([this configuration requires -mlong-double-128 support])
fi
