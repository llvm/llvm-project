/* ABI version for long double switch.
   This is used by the Versions and math_ldbl_opt.h files in
   sysdeps/ieee754/ldbl-opt/.  It gives the ABI version where
   long double == double was replaced with proper long double
   for libm *l functions and libc functions using long double.  */

#define NLDBL_VERSION			GLIBC_2.4
#define LONG_DOUBLE_COMPAT_VERSION	GLIBC_2_4
