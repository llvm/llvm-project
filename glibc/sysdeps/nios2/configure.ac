GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.
# Local configure fragment for sysdeps/nios2.

# Nios II big endian is not yet supported.
AC_CACHE_CHECK([for big endian],
  [libc_cv_nios2_be],
  [AC_EGREP_CPP(yes,[#ifdef __nios2_big_endian__
                      yes
                     #endif
  ], libc_cv_nios2_be=yes, libc_cv_nios2_be=no)])
if test $libc_cv_nios2_be = yes; then
  AC_MSG_ERROR([Big endian not supported for Nios II])
fi
