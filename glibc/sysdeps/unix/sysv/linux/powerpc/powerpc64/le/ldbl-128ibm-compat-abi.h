/* ABI version for long double switch to IEEE 128-bit floating point..
   This is used by the Versions and math_ldbl_opt.h files in
   sysdeps/ieee754/ldbl-128ibm-compat/.  It gives the ABI version where
   long double == ibm128 was replaced with long double == _Float128
   for libm *l functions and libc functions using long double.  */

#define LDBL_IBM128_VERSION		GLIBC_2.32
#define LDBL_IBM128_COMPAT_VERSION	GLIBC_2_32
