GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.
# Local configure fragment for sysdeps/powerpc/powerpc64le.

dnl Require at least POWER8 on powerpc64le
OLD_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS $libc_cv_cc_submachine"
AC_CACHE_CHECK([if the target machine is at least POWER8],
	       libc_cv_target_power8_ok, [
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
#ifndef _ARCH_PWR8
#error invalid target architecture
#endif
]])],
	       [libc_cv_target_power8_ok=yes],
	       [libc_cv_target_power8_ok=no])])
AS_IF([test "$libc_cv_target_power8_ok" != "yes"],
      [critic_missing="$critic_missing POWER8 or newer is required on powerpc64le."])
CFLAGS="$OLD_CFLAGS"

dnl Require support for -mno-gnu-attribute
OLD_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS -mno-gnu-attribute"
AC_CACHE_CHECK([if the compiler supports -mno-gnu-attribute],
	       libc_cv_no_gnu_attr_ok, [
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
]])],
	       [libc_cv_no_gnu_attr_ok=yes],
	       [libc_cv_no_gnu_attr_ok=no])])
AS_IF([test "$libc_cv_no_gnu_attr_ok" != "yes"],
      [critic_missing="$critic_missing A compiler with -mno-gnu-attribute is required on powerpc64le."])
CFLAGS="$OLD_CFLAGS"

dnl Some old compiler versions give error for the function signbit() for
dnl 128-bit floating point types.  This is fixed by PR83862 and backported
dnl to gcc6 and gcc7.  This test is to check if we are using the compiler
dnl that has this bug (fixed in GCC 8.0, backported to 7.4).
AC_CACHE_CHECK([if $CC compiles signbit with 128-bit floating point type],
	       libc_cv_compiler_powerpc64le_ice, [dnl
save_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS -mabi=ieeelongdouble -Wno-psabi"
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
int sbr (long double a) { return __builtin_signbit (a); }
int sbm (long double *a) { return __builtin_signbit (*a); }
int sbo (long double *a) { return __builtin_signbit (a[4]); }
int sbi (long double *a, unsigned long n) { return __builtin_signbit (a[n]); }
void sbs (int *p, long double a) { *p = __builtin_signbit (a); }
]])],
		  [libc_cv_compiler_powerpc64le_ice=yes],
		  [libc_cv_compiler_powerpc64le_ice=no])
CFLAGS="$save_CFLAGS"])
AS_IF([test "$libc_cv_compiler_powerpc64le_ice" != "yes"],
      [critic_missing="$critic_missing __builtin_signbit is broken.  GCC 7.4 or newer is required to resolve (PR83862)."])

dnl Some old compiler versions give out error messages when combining
dnl -mabi=% and -mlong-double-128.  i.e GCC 7.5.0 (PR94200)
AC_CACHE_CHECK([if $CC compiles with -mabi=ieeelongdouble and -mlong-double-128],
	       libc_cv_compiler_powerpc64le_ldbl128_mabi, [dnl
save_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS -mabi=ieeelongdouble -mlong-double-128"
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
long double x;
]])],
		  [libc_cv_compiler_powerpc64le_ldbl128_mabi=yes],
		  [libc_cv_compiler_powerpc64le_ldbl128_mabi=no])
CFLAGS="$save_CFLAGS"])
AS_IF([test "$libc_cv_compiler_powerpc64le_ldbl128_mabi" == "no"],
      [critic_missing="$critic_missing The compiler must support -mabi=ieeelongdouble and -mlong-double-128 simultaneously."])

dnl objcopy (binutils) 2.26 or newer required to support the --update-section
dnl feature for fixing up .gnu.attribute section with IEEE ldbl.
AC_CHECK_PROG_VER(OBJCOPY, $OBJCOPY, --version,
		  [GNU objcopy.* \([0-9]*\.[0-9.]*\)],
		  [2.1[0-9][0-9]*|2.2[6-9]*|2.[3-9][0-9]*|[3-9].*|[1-9][0-9]*],
		  AS=: critic_missing="$critic_missing objcopy >= 2.26 is required on powerpc64le")

test -n "$critic_missing" && AC_MSG_ERROR([*** $critic_missing])
