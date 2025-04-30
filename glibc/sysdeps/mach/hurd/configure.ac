GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.

dnl We need this setting because of the need for PLT calls in ld.so.
dnl See Roland's comment in
dnl https://sourceware.org/bugzilla/show_bug.cgi?id=15605
AC_DEFINE([NO_RTLD_HIDDEN])

if test -n "$sysheaders"; then
  OLD_CPPFLAGS=$CPPFLAGS
  CPPFLAGS="$CPPFLAGS $SYSINCLUDES"
fi

AC_CACHE_CHECK(Hurd header version, libc_cv_hurd_version, [dnl
AC_PREPROC_IFELSE([AC_LANG_PROGRAM(dnl
[[#include <hurd/version.h>]], [[
#define NEED_VERSION 20020609
#if HURD_INTERFACE_VERSION < NEED_VERSION
# error Hurd version too old: HURD_INTERFACE_VERSION < NEED_VERSION
#endif]])],
	       [libc_cv_hurd_version=ok],
	       [libc_cv_hurd_version=bad])])
if test "x$libc_cv_hurd_version" != xok; then
  AC_MSG_ERROR(Hurd headers not installed or too old)
fi

if test -n "$sysheaders"; then
  CPPFLAGS=$OLD_CPPFLAGS
fi

# Hurd has libpthread as a separate library.
pthread_in_libc=no
