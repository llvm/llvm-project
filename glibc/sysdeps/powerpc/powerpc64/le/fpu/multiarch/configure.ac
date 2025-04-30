GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.

OLD_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS -mcpu=power10"
AC_CACHE_CHECK([if the compiler supports -mcpu=power10],
	       libc_cv_mcpu_power10, [
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
]])],
	       [libc_cv_mcpu_power10=yes],
	       [libc_cv_mcpu_power10=no])])
LIBC_CONFIG_VAR([mcpu-power10], [$libc_cv_mcpu_power10])
AS_IF([[test "$libc_cv_mcpu_power10" == "yes"]],[
	AC_DEFINE(USE_PPC64_MCPU_POWER10)])
CFLAGS="$OLD_CFLAGS"
