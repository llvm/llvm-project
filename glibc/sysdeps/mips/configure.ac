GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.
# Local configure fragment for sysdeps/mips.

dnl No MIPS GCC supports accessing static and hidden symbols in an
dnl position independent way.
dnl AC_DEFINE(PI_STATIC_AND_HIDDEN)

AC_CACHE_CHECK([whether the compiler is using the 2008 NaN encoding],
  libc_cv_mips_nan2008, [AC_EGREP_CPP(yes, [dnl
#ifdef __mips_nan2008
yes
#endif], libc_cv_mips_nan2008=yes, libc_cv_mips_nan2008=no)])
if test x$libc_cv_mips_nan2008 = xyes; then
  AC_DEFINE(HAVE_MIPS_NAN2008)
fi
