// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-none-none -target-cpu 603e < /dev/null | FileCheck -match-full-lines -check-prefix PPC603E %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=powerpc-none-none -target-cpu 603e < /dev/null | FileCheck -match-full-lines -check-prefix PPC603E-CXX %s
//
// PPC603E:#define _ARCH_603 1
// PPC603E:#define _ARCH_603E 1
// PPC603E:#define _ARCH_PPC 1
// PPC603E:#define _ARCH_PPCGR 1
// PPC603E:#define _BIG_ENDIAN 1
// PPC603E-NOT:#define _LP64
// PPC603E:#define __BIGGEST_ALIGNMENT__ 16
// PPC603E:#define __BIG_ENDIAN__ 1
// PPC603E:#define __BYTE_ORDER__ __ORDER_BIG_ENDIAN__
// PPC603E:#define __CHAR16_TYPE__ unsigned short
// PPC603E:#define __CHAR32_TYPE__ unsigned int
// PPC603E:#define __CHAR_BIT__ 8
// PPC603E:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// PPC603E:#define __DBL_DIG__ 15
// PPC603E:#define __DBL_EPSILON__ 2.2204460492503131e-16
// PPC603E:#define __DBL_HAS_DENORM__ 1
// PPC603E:#define __DBL_HAS_INFINITY__ 1
// PPC603E:#define __DBL_HAS_QUIET_NAN__ 1
// PPC603E:#define __DBL_MANT_DIG__ 53
// PPC603E:#define __DBL_MAX_10_EXP__ 308
// PPC603E:#define __DBL_MAX_EXP__ 1024
// PPC603E:#define __DBL_MAX__ 1.7976931348623157e+308
// PPC603E:#define __DBL_MIN_10_EXP__ (-307)
// PPC603E:#define __DBL_MIN_EXP__ (-1021)
// PPC603E:#define __DBL_MIN__ 2.2250738585072014e-308
// PPC603E:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// PPC603E:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// PPC603E:#define __FLT_DIG__ 6
// PPC603E:#define __FLT_EPSILON__ 1.19209290e-7F
// PPC603E:#define __FLT_EVAL_METHOD__ 0
// PPC603E:#define __FLT_HAS_DENORM__ 1
// PPC603E:#define __FLT_HAS_INFINITY__ 1
// PPC603E:#define __FLT_HAS_QUIET_NAN__ 1
// PPC603E:#define __FLT_MANT_DIG__ 24
// PPC603E:#define __FLT_MAX_10_EXP__ 38
// PPC603E:#define __FLT_MAX_EXP__ 128
// PPC603E:#define __FLT_MAX__ 3.40282347e+38F
// PPC603E:#define __FLT_MIN_10_EXP__ (-37)
// PPC603E:#define __FLT_MIN_EXP__ (-125)
// PPC603E:#define __FLT_MIN__ 1.17549435e-38F
// PPC603E:#define __FLT_RADIX__ 2
// PPC603E:#define __INT16_C_SUFFIX__
// PPC603E:#define __INT16_FMTd__ "hd"
// PPC603E:#define __INT16_FMTi__ "hi"
// PPC603E:#define __INT16_MAX__ 32767
// PPC603E:#define __INT16_TYPE__ short
// PPC603E:#define __INT32_C_SUFFIX__
// PPC603E:#define __INT32_FMTd__ "d"
// PPC603E:#define __INT32_FMTi__ "i"
// PPC603E:#define __INT32_MAX__ 2147483647
// PPC603E:#define __INT32_TYPE__ int
// PPC603E:#define __INT64_C_SUFFIX__ LL
// PPC603E:#define __INT64_FMTd__ "lld"
// PPC603E:#define __INT64_FMTi__ "lli"
// PPC603E:#define __INT64_MAX__ 9223372036854775807LL
// PPC603E:#define __INT64_TYPE__ long long int
// PPC603E:#define __INT8_C_SUFFIX__
// PPC603E:#define __INT8_FMTd__ "hhd"
// PPC603E:#define __INT8_FMTi__ "hhi"
// PPC603E:#define __INT8_MAX__ 127
// PPC603E:#define __INT8_TYPE__ signed char
// PPC603E:#define __INTMAX_C_SUFFIX__ LL
// PPC603E:#define __INTMAX_FMTd__ "lld"
// PPC603E:#define __INTMAX_FMTi__ "lli"
// PPC603E:#define __INTMAX_MAX__ 9223372036854775807LL
// PPC603E:#define __INTMAX_TYPE__ long long int
// PPC603E:#define __INTMAX_WIDTH__ 64
// PPC603E:#define __INTPTR_FMTd__ "ld"
// PPC603E:#define __INTPTR_FMTi__ "li"
// PPC603E:#define __INTPTR_MAX__ 2147483647L
// PPC603E:#define __INTPTR_TYPE__ long int
// PPC603E:#define __INTPTR_WIDTH__ 32
// PPC603E:#define __INT_FAST16_FMTd__ "hd"
// PPC603E:#define __INT_FAST16_FMTi__ "hi"
// PPC603E:#define __INT_FAST16_MAX__ 32767
// PPC603E:#define __INT_FAST16_TYPE__ short
// PPC603E:#define __INT_FAST32_FMTd__ "d"
// PPC603E:#define __INT_FAST32_FMTi__ "i"
// PPC603E:#define __INT_FAST32_MAX__ 2147483647
// PPC603E:#define __INT_FAST32_TYPE__ int
// PPC603E:#define __INT_FAST64_FMTd__ "lld"
// PPC603E:#define __INT_FAST64_FMTi__ "lli"
// PPC603E:#define __INT_FAST64_MAX__ 9223372036854775807LL
// PPC603E:#define __INT_FAST64_TYPE__ long long int
// PPC603E:#define __INT_FAST8_FMTd__ "hhd"
// PPC603E:#define __INT_FAST8_FMTi__ "hhi"
// PPC603E:#define __INT_FAST8_MAX__ 127
// PPC603E:#define __INT_FAST8_TYPE__ signed char
// PPC603E:#define __INT_LEAST16_FMTd__ "hd"
// PPC603E:#define __INT_LEAST16_FMTi__ "hi"
// PPC603E:#define __INT_LEAST16_MAX__ 32767
// PPC603E:#define __INT_LEAST16_TYPE__ short
// PPC603E:#define __INT_LEAST32_FMTd__ "d"
// PPC603E:#define __INT_LEAST32_FMTi__ "i"
// PPC603E:#define __INT_LEAST32_MAX__ 2147483647
// PPC603E:#define __INT_LEAST32_TYPE__ int
// PPC603E:#define __INT_LEAST64_FMTd__ "lld"
// PPC603E:#define __INT_LEAST64_FMTi__ "lli"
// PPC603E:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// PPC603E:#define __INT_LEAST64_TYPE__ long long int
// PPC603E:#define __INT_LEAST8_FMTd__ "hhd"
// PPC603E:#define __INT_LEAST8_FMTi__ "hhi"
// PPC603E:#define __INT_LEAST8_MAX__ 127
// PPC603E:#define __INT_LEAST8_TYPE__ signed char
// PPC603E:#define __INT_MAX__ 2147483647
// PPC603E:#define __LDBL_DENORM_MIN__ 4.94065645841246544176568792868221e-324L
// PPC603E:#define __LDBL_DIG__ 31
// PPC603E:#define __LDBL_EPSILON__ 4.94065645841246544176568792868221e-324L
// PPC603E:#define __LDBL_HAS_DENORM__ 1
// PPC603E:#define __LDBL_HAS_INFINITY__ 1
// PPC603E:#define __LDBL_HAS_QUIET_NAN__ 1
// PPC603E:#define __LDBL_MANT_DIG__ 106
// PPC603E:#define __LDBL_MAX_10_EXP__ 308
// PPC603E:#define __LDBL_MAX_EXP__ 1024
// PPC603E:#define __LDBL_MAX__ 1.79769313486231580793728971405301e+308L
// PPC603E:#define __LDBL_MIN_10_EXP__ (-291)
// PPC603E:#define __LDBL_MIN_EXP__ (-968)
// PPC603E:#define __LDBL_MIN__ 2.00416836000897277799610805135016e-292L
// PPC603E:#define __LONGDOUBLE128 1
// PPC603E:#define __LONG_DOUBLE_128__ 1
// PPC603E:#define __LONG_LONG_MAX__ 9223372036854775807LL
// PPC603E:#define __LONG_MAX__ 2147483647L
// PPC603E-NOT:#define __LP64__
// PPC603E:#define __NATURAL_ALIGNMENT__ 1
// PPC603E:#define __POINTER_WIDTH__ 32
// PPC603E:#define __POWERPC__ 1
// PPC603E:#define __PPC__ 1
// PPC603E:#define __PTRDIFF_TYPE__ long int
// PPC603E:#define __PTRDIFF_WIDTH__ 32
// PPC603E:#define __REGISTER_PREFIX__
// PPC603E:#define __SCHAR_MAX__ 127
// PPC603E:#define __SHRT_MAX__ 32767
// PPC603E:#define __SIG_ATOMIC_MAX__ 2147483647
// PPC603E:#define __SIG_ATOMIC_WIDTH__ 32
// PPC603E:#define __SIZEOF_DOUBLE__ 8
// PPC603E:#define __SIZEOF_FLOAT__ 4
// PPC603E:#define __SIZEOF_INT__ 4
// PPC603E:#define __SIZEOF_LONG_DOUBLE__ 16
// PPC603E:#define __SIZEOF_LONG_LONG__ 8
// PPC603E:#define __SIZEOF_LONG__ 4
// PPC603E:#define __SIZEOF_POINTER__ 4
// PPC603E:#define __SIZEOF_PTRDIFF_T__ 4
// PPC603E:#define __SIZEOF_SHORT__ 2
// PPC603E:#define __SIZEOF_SIZE_T__ 4
// PPC603E:#define __SIZEOF_WCHAR_T__ 4
// PPC603E:#define __SIZEOF_WINT_T__ 4
// PPC603E:#define __SIZE_MAX__ 4294967295UL
// PPC603E:#define __SIZE_TYPE__ long unsigned int
// PPC603E:#define __SIZE_WIDTH__ 32
// PPC603E-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 16UL
// PPC603E:#define __UINT16_C_SUFFIX__
// PPC603E:#define __UINT16_MAX__ 65535
// PPC603E:#define __UINT16_TYPE__ unsigned short
// PPC603E:#define __UINT32_C_SUFFIX__ U
// PPC603E:#define __UINT32_MAX__ 4294967295U
// PPC603E:#define __UINT32_TYPE__ unsigned int
// PPC603E:#define __UINT64_C_SUFFIX__ ULL
// PPC603E:#define __UINT64_MAX__ 18446744073709551615ULL
// PPC603E:#define __UINT64_TYPE__ long long unsigned int
// PPC603E:#define __UINT8_C_SUFFIX__
// PPC603E:#define __UINT8_MAX__ 255
// PPC603E:#define __UINT8_TYPE__ unsigned char
// PPC603E:#define __UINTMAX_C_SUFFIX__ ULL
// PPC603E:#define __UINTMAX_MAX__ 18446744073709551615ULL
// PPC603E:#define __UINTMAX_TYPE__ long long unsigned int
// PPC603E:#define __UINTMAX_WIDTH__ 64
// PPC603E:#define __UINTPTR_MAX__ 4294967295UL
// PPC603E:#define __UINTPTR_TYPE__ long unsigned int
// PPC603E:#define __UINTPTR_WIDTH__ 32
// PPC603E:#define __UINT_FAST16_MAX__ 65535
// PPC603E:#define __UINT_FAST16_TYPE__ unsigned short
// PPC603E:#define __UINT_FAST32_MAX__ 4294967295U
// PPC603E:#define __UINT_FAST32_TYPE__ unsigned int
// PPC603E:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// PPC603E:#define __UINT_FAST64_TYPE__ long long unsigned int
// PPC603E:#define __UINT_FAST8_MAX__ 255
// PPC603E:#define __UINT_FAST8_TYPE__ unsigned char
// PPC603E:#define __UINT_LEAST16_MAX__ 65535
// PPC603E:#define __UINT_LEAST16_TYPE__ unsigned short
// PPC603E:#define __UINT_LEAST32_MAX__ 4294967295U
// PPC603E:#define __UINT_LEAST32_TYPE__ unsigned int
// PPC603E:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// PPC603E:#define __UINT_LEAST64_TYPE__ long long unsigned int
// PPC603E:#define __UINT_LEAST8_MAX__ 255
// PPC603E:#define __UINT_LEAST8_TYPE__ unsigned char
// PPC603E:#define __USER_LABEL_PREFIX__
// PPC603E:#define __WCHAR_MAX__ 2147483647
// PPC603E:#define __WCHAR_TYPE__ int
// PPC603E:#define __WCHAR_WIDTH__ 32
// PPC603E:#define __WINT_TYPE__ int
// PPC603E:#define __WINT_WIDTH__ 32
// PPC603E:#define __powerpc__ 1
// PPC603E:#define __ppc__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-none-none -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPC %s
//
// PPC:#define _ARCH_PPC 1
// PPC:#define _BIG_ENDIAN 1
// PPC-NOT:#define _LP64
// PPC:#define __BIGGEST_ALIGNMENT__ 16
// PPC:#define __BIG_ENDIAN__ 1
// PPC:#define __BYTE_ORDER__ __ORDER_BIG_ENDIAN__
// PPC:#define __CHAR16_TYPE__ unsigned short
// PPC:#define __CHAR32_TYPE__ unsigned int
// PPC:#define __CHAR_BIT__ 8
// PPC:#define __CHAR_UNSIGNED__ 1
// PPC:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// PPC:#define __DBL_DIG__ 15
// PPC:#define __DBL_EPSILON__ 2.2204460492503131e-16
// PPC:#define __DBL_HAS_DENORM__ 1
// PPC:#define __DBL_HAS_INFINITY__ 1
// PPC:#define __DBL_HAS_QUIET_NAN__ 1
// PPC:#define __DBL_MANT_DIG__ 53
// PPC:#define __DBL_MAX_10_EXP__ 308
// PPC:#define __DBL_MAX_EXP__ 1024
// PPC:#define __DBL_MAX__ 1.7976931348623157e+308
// PPC:#define __DBL_MIN_10_EXP__ (-307)
// PPC:#define __DBL_MIN_EXP__ (-1021)
// PPC:#define __DBL_MIN__ 2.2250738585072014e-308
// PPC:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// PPC:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// PPC:#define __FLT_DIG__ 6
// PPC:#define __FLT_EPSILON__ 1.19209290e-7F
// PPC:#define __FLT_EVAL_METHOD__ 0
// PPC:#define __FLT_HAS_DENORM__ 1
// PPC:#define __FLT_HAS_INFINITY__ 1
// PPC:#define __FLT_HAS_QUIET_NAN__ 1
// PPC:#define __FLT_MANT_DIG__ 24
// PPC:#define __FLT_MAX_10_EXP__ 38
// PPC:#define __FLT_MAX_EXP__ 128
// PPC:#define __FLT_MAX__ 3.40282347e+38F
// PPC:#define __FLT_MIN_10_EXP__ (-37)
// PPC:#define __FLT_MIN_EXP__ (-125)
// PPC:#define __FLT_MIN__ 1.17549435e-38F
// PPC:#define __FLT_RADIX__ 2
// PPC:#define __HAVE_BSWAP__ 1
// PPC:#define __INT16_C_SUFFIX__
// PPC:#define __INT16_FMTd__ "hd"
// PPC:#define __INT16_FMTi__ "hi"
// PPC:#define __INT16_MAX__ 32767
// PPC:#define __INT16_TYPE__ short
// PPC:#define __INT32_C_SUFFIX__
// PPC:#define __INT32_FMTd__ "d"
// PPC:#define __INT32_FMTi__ "i"
// PPC:#define __INT32_MAX__ 2147483647
// PPC:#define __INT32_TYPE__ int
// PPC:#define __INT64_C_SUFFIX__ LL
// PPC:#define __INT64_FMTd__ "lld"
// PPC:#define __INT64_FMTi__ "lli"
// PPC:#define __INT64_MAX__ 9223372036854775807LL
// PPC:#define __INT64_TYPE__ long long int
// PPC:#define __INT8_C_SUFFIX__
// PPC:#define __INT8_FMTd__ "hhd"
// PPC:#define __INT8_FMTi__ "hhi"
// PPC:#define __INT8_MAX__ 127
// PPC:#define __INT8_TYPE__ signed char
// PPC:#define __INTMAX_C_SUFFIX__ LL
// PPC:#define __INTMAX_FMTd__ "lld"
// PPC:#define __INTMAX_FMTi__ "lli"
// PPC:#define __INTMAX_MAX__ 9223372036854775807LL
// PPC:#define __INTMAX_TYPE__ long long int
// PPC:#define __INTMAX_WIDTH__ 64
// PPC:#define __INTPTR_FMTd__ "ld"
// PPC:#define __INTPTR_FMTi__ "li"
// PPC:#define __INTPTR_MAX__ 2147483647L
// PPC:#define __INTPTR_TYPE__ long int
// PPC:#define __INTPTR_WIDTH__ 32
// PPC:#define __INT_FAST16_FMTd__ "hd"
// PPC:#define __INT_FAST16_FMTi__ "hi"
// PPC:#define __INT_FAST16_MAX__ 32767
// PPC:#define __INT_FAST16_TYPE__ short
// PPC:#define __INT_FAST32_FMTd__ "d"
// PPC:#define __INT_FAST32_FMTi__ "i"
// PPC:#define __INT_FAST32_MAX__ 2147483647
// PPC:#define __INT_FAST32_TYPE__ int
// PPC:#define __INT_FAST64_FMTd__ "lld"
// PPC:#define __INT_FAST64_FMTi__ "lli"
// PPC:#define __INT_FAST64_MAX__ 9223372036854775807LL
// PPC:#define __INT_FAST64_TYPE__ long long int
// PPC:#define __INT_FAST8_FMTd__ "hhd"
// PPC:#define __INT_FAST8_FMTi__ "hhi"
// PPC:#define __INT_FAST8_MAX__ 127
// PPC:#define __INT_FAST8_TYPE__ signed char
// PPC:#define __INT_LEAST16_FMTd__ "hd"
// PPC:#define __INT_LEAST16_FMTi__ "hi"
// PPC:#define __INT_LEAST16_MAX__ 32767
// PPC:#define __INT_LEAST16_TYPE__ short
// PPC:#define __INT_LEAST32_FMTd__ "d"
// PPC:#define __INT_LEAST32_FMTi__ "i"
// PPC:#define __INT_LEAST32_MAX__ 2147483647
// PPC:#define __INT_LEAST32_TYPE__ int
// PPC:#define __INT_LEAST64_FMTd__ "lld"
// PPC:#define __INT_LEAST64_FMTi__ "lli"
// PPC:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// PPC:#define __INT_LEAST64_TYPE__ long long int
// PPC:#define __INT_LEAST8_FMTd__ "hhd"
// PPC:#define __INT_LEAST8_FMTi__ "hhi"
// PPC:#define __INT_LEAST8_MAX__ 127
// PPC:#define __INT_LEAST8_TYPE__ signed char
// PPC:#define __INT_MAX__ 2147483647
// PPC:#define __LDBL_DENORM_MIN__ 4.94065645841246544176568792868221e-324L
// PPC:#define __LDBL_DIG__ 31
// PPC:#define __LDBL_EPSILON__ 4.94065645841246544176568792868221e-324L
// PPC:#define __LDBL_HAS_DENORM__ 1
// PPC:#define __LDBL_HAS_INFINITY__ 1
// PPC:#define __LDBL_HAS_QUIET_NAN__ 1
// PPC:#define __LDBL_MANT_DIG__ 106
// PPC:#define __LDBL_MAX_10_EXP__ 308
// PPC:#define __LDBL_MAX_EXP__ 1024
// PPC:#define __LDBL_MAX__ 1.79769313486231580793728971405301e+308L
// PPC:#define __LDBL_MIN_10_EXP__ (-291)
// PPC:#define __LDBL_MIN_EXP__ (-968)
// PPC:#define __LDBL_MIN__ 2.00416836000897277799610805135016e-292L
// PPC:#define __LONGDOUBLE128 1
// PPC:#define __LONG_DOUBLE_128__ 1
// PPC:#define __LONG_LONG_MAX__ 9223372036854775807LL
// PPC:#define __LONG_MAX__ 2147483647L
// PPC-NOT:#define __LP64__
// PPC:#define __NATURAL_ALIGNMENT__ 1
// PPC:#define __POINTER_WIDTH__ 32
// PPC:#define __POWERPC__ 1
// PPC:#define __PPC__ 1
// PPC:#define __PTRDIFF_TYPE__ long int
// PPC:#define __PTRDIFF_WIDTH__ 32
// PPC:#define __REGISTER_PREFIX__
// PPC:#define __SCHAR_MAX__ 127
// PPC:#define __SHRT_MAX__ 32767
// PPC:#define __SIG_ATOMIC_MAX__ 2147483647
// PPC:#define __SIG_ATOMIC_WIDTH__ 32
// PPC:#define __SIZEOF_DOUBLE__ 8
// PPC:#define __SIZEOF_FLOAT__ 4
// PPC:#define __SIZEOF_INT__ 4
// PPC:#define __SIZEOF_LONG_DOUBLE__ 16
// PPC:#define __SIZEOF_LONG_LONG__ 8
// PPC:#define __SIZEOF_LONG__ 4
// PPC:#define __SIZEOF_POINTER__ 4
// PPC:#define __SIZEOF_PTRDIFF_T__ 4
// PPC:#define __SIZEOF_SHORT__ 2
// PPC:#define __SIZEOF_SIZE_T__ 4
// PPC:#define __SIZEOF_WCHAR_T__ 4
// PPC:#define __SIZEOF_WINT_T__ 4
// PPC:#define __SIZE_MAX__ 4294967295UL
// PPC:#define __SIZE_TYPE__ long unsigned int
// PPC:#define __SIZE_WIDTH__ 32
// PPC:#define __UINT16_C_SUFFIX__
// PPC:#define __UINT16_MAX__ 65535
// PPC:#define __UINT16_TYPE__ unsigned short
// PPC:#define __UINT32_C_SUFFIX__ U
// PPC:#define __UINT32_MAX__ 4294967295U
// PPC:#define __UINT32_TYPE__ unsigned int
// PPC:#define __UINT64_C_SUFFIX__ ULL
// PPC:#define __UINT64_MAX__ 18446744073709551615ULL
// PPC:#define __UINT64_TYPE__ long long unsigned int
// PPC:#define __UINT8_C_SUFFIX__
// PPC:#define __UINT8_MAX__ 255
// PPC:#define __UINT8_TYPE__ unsigned char
// PPC:#define __UINTMAX_C_SUFFIX__ ULL
// PPC:#define __UINTMAX_MAX__ 18446744073709551615ULL
// PPC:#define __UINTMAX_TYPE__ long long unsigned int
// PPC:#define __UINTMAX_WIDTH__ 64
// PPC:#define __UINTPTR_MAX__ 4294967295UL
// PPC:#define __UINTPTR_TYPE__ long unsigned int
// PPC:#define __UINTPTR_WIDTH__ 32
// PPC:#define __UINT_FAST16_MAX__ 65535
// PPC:#define __UINT_FAST16_TYPE__ unsigned short
// PPC:#define __UINT_FAST32_MAX__ 4294967295U
// PPC:#define __UINT_FAST32_TYPE__ unsigned int
// PPC:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// PPC:#define __UINT_FAST64_TYPE__ long long unsigned int
// PPC:#define __UINT_FAST8_MAX__ 255
// PPC:#define __UINT_FAST8_TYPE__ unsigned char
// PPC:#define __UINT_LEAST16_MAX__ 65535
// PPC:#define __UINT_LEAST16_TYPE__ unsigned short
// PPC:#define __UINT_LEAST32_MAX__ 4294967295U
// PPC:#define __UINT_LEAST32_TYPE__ unsigned int
// PPC:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// PPC:#define __UINT_LEAST64_TYPE__ long long unsigned int
// PPC:#define __UINT_LEAST8_MAX__ 255
// PPC:#define __UINT_LEAST8_TYPE__ unsigned char
// PPC:#define __USER_LABEL_PREFIX__
// PPC:#define __WCHAR_MAX__ 2147483647
// PPC:#define __WCHAR_TYPE__ int
// PPC:#define __WCHAR_WIDTH__ 32
// PPC:#define __WINT_TYPE__ int
// PPC:#define __WINT_WIDTH__ 32
// PPC:#define __ppc__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-ibm-aix7.1.0.0 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX %s
//
// PPC-AIX-NOT:#define __64BIT__ 1
// PPC-AIX:#define _AIX 1
// PPC-AIX:#define _ARCH_PPC 1
// PPC-AIX:#define _BIG_ENDIAN 1
// PPC-AIX:#define _IBMR2 1
// PPC-AIX:#define _LONG_LONG 1
// PPC-AIX-NOT:#define _LP64 1
// PPC-AIX:#define _POWER 1
// PPC-AIX:#define __BIGGEST_ALIGNMENT__ 16
// PPC-AIX:#define __BIG_ENDIAN__ 1
// PPC-AIX:#define __BYTE_ORDER__ __ORDER_BIG_ENDIAN__
// PPC-AIX:#define __CHAR16_TYPE__ unsigned short
// PPC-AIX:#define __CHAR32_TYPE__ unsigned int
// PPC-AIX:#define __CHAR_BIT__ 8
// PPC-AIX:#define __CHAR_UNSIGNED__ 1
// PPC-AIX:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// PPC-AIX:#define __DBL_DIG__ 15
// PPC-AIX:#define __DBL_EPSILON__ 2.2204460492503131e-16
// PPC-AIX:#define __DBL_HAS_DENORM__ 1
// PPC-AIX:#define __DBL_HAS_INFINITY__ 1
// PPC-AIX:#define __DBL_HAS_QUIET_NAN__ 1
// PPC-AIX:#define __DBL_MANT_DIG__ 53
// PPC-AIX:#define __DBL_MAX_10_EXP__ 308
// PPC-AIX:#define __DBL_MAX_EXP__ 1024
// PPC-AIX:#define __DBL_MAX__ 1.7976931348623157e+308
// PPC-AIX:#define __DBL_MIN_10_EXP__ (-307)
// PPC-AIX:#define __DBL_MIN_EXP__ (-1021)
// PPC-AIX:#define __DBL_MIN__ 2.2250738585072014e-308
// PPC-AIX:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// PPC-AIX:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// PPC-AIX:#define __FLT_DIG__ 6
// PPC-AIX:#define __FLT_EPSILON__ 1.19209290e-7F
// PPC-AIX:#define __FLT_EVAL_METHOD__ 1
// PPC-AIX:#define __FLT_HAS_DENORM__ 1
// PPC-AIX:#define __FLT_HAS_INFINITY__ 1
// PPC-AIX:#define __FLT_HAS_QUIET_NAN__ 1
// PPC-AIX:#define __FLT_MANT_DIG__ 24
// PPC-AIX:#define __FLT_MAX_10_EXP__ 38
// PPC-AIX:#define __FLT_MAX_EXP__ 128
// PPC-AIX:#define __FLT_MAX__ 3.40282347e+38F
// PPC-AIX:#define __FLT_MIN_10_EXP__ (-37)
// PPC-AIX:#define __FLT_MIN_EXP__ (-125)
// PPC-AIX:#define __FLT_MIN__ 1.17549435e-38F
// PPC-AIX:#define __FLT_RADIX__ 2
// PPC-AIX:#define __INT16_C_SUFFIX__
// PPC-AIX:#define __INT16_FMTd__ "hd"
// PPC-AIX:#define __INT16_FMTi__ "hi"
// PPC-AIX:#define __INT16_MAX__ 32767
// PPC-AIX:#define __INT16_TYPE__ short
// PPC-AIX:#define __INT32_C_SUFFIX__
// PPC-AIX:#define __INT32_FMTd__ "d"
// PPC-AIX:#define __INT32_FMTi__ "i"
// PPC-AIX:#define __INT32_MAX__ 2147483647
// PPC-AIX:#define __INT32_TYPE__ int
// PPC-AIX:#define __INT64_C_SUFFIX__ LL
// PPC-AIX:#define __INT64_FMTd__ "lld"
// PPC-AIX:#define __INT64_FMTi__ "lli"
// PPC-AIX:#define __INT64_MAX__ 9223372036854775807LL
// PPC-AIX:#define __INT64_TYPE__ long long int
// PPC-AIX:#define __INT8_C_SUFFIX__
// PPC-AIX:#define __INT8_FMTd__ "hhd"
// PPC-AIX:#define __INT8_FMTi__ "hhi"
// PPC-AIX:#define __INT8_MAX__ 127
// PPC-AIX:#define __INT8_TYPE__ signed char
// PPC-AIX:#define __INTMAX_C_SUFFIX__ LL
// PPC-AIX:#define __INTMAX_FMTd__ "lld"
// PPC-AIX:#define __INTMAX_FMTi__ "lli"
// PPC-AIX:#define __INTMAX_MAX__ 9223372036854775807LL
// PPC-AIX:#define __INTMAX_TYPE__ long long int
// PPC-AIX:#define __INTMAX_WIDTH__ 64
// PPC-AIX:#define __INTPTR_FMTd__ "ld"
// PPC-AIX:#define __INTPTR_FMTi__ "li"
// PPC-AIX:#define __INTPTR_MAX__ 2147483647L
// PPC-AIX:#define __INTPTR_TYPE__ long int
// PPC-AIX:#define __INTPTR_WIDTH__ 32
// PPC-AIX:#define __INT_FAST16_FMTd__ "hd"
// PPC-AIX:#define __INT_FAST16_FMTi__ "hi"
// PPC-AIX:#define __INT_FAST16_MAX__ 32767
// PPC-AIX:#define __INT_FAST16_TYPE__ short
// PPC-AIX:#define __INT_FAST32_FMTd__ "d"
// PPC-AIX:#define __INT_FAST32_FMTi__ "i"
// PPC-AIX:#define __INT_FAST32_MAX__ 2147483647
// PPC-AIX:#define __INT_FAST32_TYPE__ int
// PPC-AIX:#define __INT_FAST64_FMTd__ "lld"
// PPC-AIX:#define __INT_FAST64_FMTi__ "lli"
// PPC-AIX:#define __INT_FAST64_MAX__ 9223372036854775807LL
// PPC-AIX:#define __INT_FAST64_TYPE__ long long int
// PPC-AIX:#define __INT_FAST8_FMTd__ "hhd"
// PPC-AIX:#define __INT_FAST8_FMTi__ "hhi"
// PPC-AIX:#define __INT_FAST8_MAX__ 127
// PPC-AIX:#define __INT_FAST8_TYPE__ signed char
// PPC-AIX:#define __INT_LEAST16_FMTd__ "hd"
// PPC-AIX:#define __INT_LEAST16_FMTi__ "hi"
// PPC-AIX:#define __INT_LEAST16_MAX__ 32767
// PPC-AIX:#define __INT_LEAST16_TYPE__ short
// PPC-AIX:#define __INT_LEAST32_FMTd__ "d"
// PPC-AIX:#define __INT_LEAST32_FMTi__ "i"
// PPC-AIX:#define __INT_LEAST32_MAX__ 2147483647
// PPC-AIX:#define __INT_LEAST32_TYPE__ int
// PPC-AIX:#define __INT_LEAST64_FMTd__ "lld"
// PPC-AIX:#define __INT_LEAST64_FMTi__ "lli"
// PPC-AIX:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// PPC-AIX:#define __INT_LEAST64_TYPE__ long long int
// PPC-AIX:#define __INT_LEAST8_FMTd__ "hhd"
// PPC-AIX:#define __INT_LEAST8_FMTi__ "hhi"
// PPC-AIX:#define __INT_LEAST8_MAX__ 127
// PPC-AIX:#define __INT_LEAST8_TYPE__ signed char
// PPC-AIX:#define __INT_MAX__ 2147483647
// PPC-AIX:#define __LDBL_DECIMAL_DIG__ 17
// PPC-AIX:#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324L
// PPC-AIX:#define __LDBL_DIG__ 15
// PPC-AIX:#define __LDBL_EPSILON__ 2.2204460492503131e-16L
// PPC-AIX:#define __LDBL_HAS_DENORM__ 1
// PPC-AIX:#define __LDBL_HAS_INFINITY__ 1
// PPC-AIX:#define __LDBL_HAS_QUIET_NAN__ 1
// PPC-AIX:#define __LDBL_MANT_DIG__ 53
// PPC-AIX:#define __LDBL_MAX_10_EXP__ 308
// PPC-AIX:#define __LDBL_MAX_EXP__ 1024
// PPC-AIX:#define __LDBL_MAX__ 1.7976931348623157e+308L
// PPC-AIX:#define __LDBL_MIN_10_EXP__ (-307)
// PPC-AIX:#define __LDBL_MIN_EXP__ (-1021)
// PPC-AIX:#define __LDBL_MIN__ 2.2250738585072014e-308L
// PPC-AIX:#define __LONG_LONG_MAX__ 9223372036854775807LL
// PPC-AIX:#define __LONG_MAX__ 2147483647L
// PPC-AIX-NOT:#define __LP64__ 1
// PPC-AIX-NOT:#define __NATURAL_ALIGNMENT__ 1
// PPC-AIX:#define __POINTER_WIDTH__ 32
// PPC-AIX:#define __POWERPC__ 1
// PPC-AIX:#define __PPC__ 1
// PPC-AIX:#define __PTRDIFF_TYPE__ long int
// PPC-AIX:#define __PTRDIFF_WIDTH__ 32
// PPC-AIX:#define __REGISTER_PREFIX__
// PPC-AIX:#define __SCHAR_MAX__ 127
// PPC-AIX:#define __SHRT_MAX__ 32767
// PPC-AIX:#define __SIG_ATOMIC_MAX__ 2147483647
// PPC-AIX:#define __SIG_ATOMIC_WIDTH__ 32
// PPC-AIX:#define __SIZEOF_DOUBLE__ 8
// PPC-AIX:#define __SIZEOF_FLOAT__ 4
// PPC-AIX:#define __SIZEOF_INT__ 4
// PPC-AIX:#define __SIZEOF_LONG_DOUBLE__ 8
// PPC-AIX:#define __SIZEOF_LONG_LONG__ 8
// PPC-AIX:#define __SIZEOF_LONG__ 4
// PPC-AIX:#define __SIZEOF_POINTER__ 4
// PPC-AIX:#define __SIZEOF_PTRDIFF_T__ 4
// PPC-AIX:#define __SIZEOF_SHORT__ 2
// PPC-AIX:#define __SIZEOF_SIZE_T__ 4
// PPC-AIX:#define __SIZEOF_WCHAR_T__ 2
// PPC-AIX:#define __SIZEOF_WINT_T__ 4
// PPC-AIX:#define __SIZE_MAX__ 4294967295UL
// PPC-AIX:#define __SIZE_TYPE__ long unsigned int
// PPC-AIX:#define __SIZE_WIDTH__ 32
// PPC-AIX:#define __UINT16_C_SUFFIX__
// PPC-AIX:#define __UINT16_MAX__ 65535
// PPC-AIX:#define __UINT16_TYPE__ unsigned short
// PPC-AIX:#define __UINT32_C_SUFFIX__ U
// PPC-AIX:#define __UINT32_MAX__ 4294967295U
// PPC-AIX:#define __UINT32_TYPE__ unsigned int
// PPC-AIX:#define __UINT64_C_SUFFIX__ ULL
// PPC-AIX:#define __UINT64_MAX__ 18446744073709551615ULL
// PPC-AIX:#define __UINT64_TYPE__ long long unsigned int
// PPC-AIX:#define __UINT8_C_SUFFIX__
// PPC-AIX:#define __UINT8_MAX__ 255
// PPC-AIX:#define __UINT8_TYPE__ unsigned char
// PPC-AIX:#define __UINTMAX_C_SUFFIX__ ULL
// PPC-AIX:#define __UINTMAX_MAX__ 18446744073709551615ULL
// PPC-AIX:#define __UINTMAX_TYPE__ long long unsigned int
// PPC-AIX:#define __UINTMAX_WIDTH__ 64
// PPC-AIX:#define __UINTPTR_MAX__ 4294967295UL
// PPC-AIX:#define __UINTPTR_TYPE__ long unsigned int
// PPC-AIX:#define __UINTPTR_WIDTH__ 32
// PPC-AIX:#define __UINT_FAST16_MAX__ 65535
// PPC-AIX:#define __UINT_FAST16_TYPE__ unsigned short
// PPC-AIX:#define __UINT_FAST32_MAX__ 4294967295U
// PPC-AIX:#define __UINT_FAST32_TYPE__ unsigned int
// PPC-AIX:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// PPC-AIX:#define __UINT_FAST64_TYPE__ long long unsigned int
// PPC-AIX:#define __UINT_FAST8_MAX__ 255
// PPC-AIX:#define __UINT_FAST8_TYPE__ unsigned char
// PPC-AIX:#define __UINT_LEAST16_MAX__ 65535
// PPC-AIX:#define __UINT_LEAST16_TYPE__ unsigned short
// PPC-AIX:#define __UINT_LEAST32_MAX__ 4294967295U
// PPC-AIX:#define __UINT_LEAST32_TYPE__ unsigned int
// PPC-AIX:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// PPC-AIX:#define __UINT_LEAST64_TYPE__ long long unsigned int
// PPC-AIX:#define __UINT_LEAST8_MAX__ 255
// PPC-AIX:#define __UINT_LEAST8_TYPE__ unsigned char
// PPC-AIX:#define __USER_LABEL_PREFIX__
// PPC-AIX:#define __WCHAR_MAX__ 65535
// PPC-AIX:#define __WCHAR_TYPE__ unsigned short
// PPC-AIX:#define __WCHAR_WIDTH__ 16
// PPC-AIX:#define __WINT_TYPE__ int
// PPC-AIX:#define __WINT_WIDTH__ 32
// PPC-AIX:#define __powerpc__ 1
// PPC-AIX:#define __ppc__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-ibm-aix7.2.0.0 < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX72 %s
//
// PPC-AIX72:#define _AIX32 1
// PPC-AIX72:#define _AIX41 1
// PPC-AIX72:#define _AIX43 1
// PPC-AIX72:#define _AIX50 1
// PPC-AIX72:#define _AIX51 1
// PPC-AIX72:#define _AIX52 1
// PPC-AIX72:#define _AIX53 1
// PPC-AIX72:#define _AIX61 1
// PPC-AIX72:#define _AIX71 1
// PPC-AIX72:#define _AIX72 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-ibm-aix7.1.0.0 < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX71 %s
//
// PPC-AIX71:#define _AIX32 1
// PPC-AIX71:#define _AIX41 1
// PPC-AIX71:#define _AIX43 1
// PPC-AIX71:#define _AIX50 1
// PPC-AIX71:#define _AIX51 1
// PPC-AIX71:#define _AIX52 1
// PPC-AIX71:#define _AIX53 1
// PPC-AIX71:#define _AIX61 1
// PPC-AIX71:#define _AIX71 1
// PPC-AIX71-NOT:#define _AIX72 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-ibm-aix6.1.0.0 < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX61 %s
//
// PPC-AIX61:#define _AIX32 1
// PPC-AIX61:#define _AIX41 1
// PPC-AIX61:#define _AIX43 1
// PPC-AIX61:#define _AIX50 1
// PPC-AIX61:#define _AIX51 1
// PPC-AIX61:#define _AIX52 1
// PPC-AIX61:#define _AIX53 1
// PPC-AIX61:#define _AIX61 1
// PPC-AIX61-NOT:#define _AIX71 1
// PPC-AIX61-NOT:#define _AIX72 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-ibm-aix5.3.0.0 < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX53 %s
// PPC-AIX53:#define _AIX32 1
// PPC-AIX53:#define _AIX41 1
// PPC-AIX53:#define _AIX43 1
// PPC-AIX53:#define _AIX50 1
// PPC-AIX53:#define _AIX51 1
// PPC-AIX53:#define _AIX52 1
// PPC-AIX53:#define _AIX53 1
// PPC-AIX53-NOT:#define _AIX61 1
// PPC-AIX53-NOT:#define _AIX71 1
// PPC-AIX53-NOT:#define _AIX72 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-ibm-aix5.2.0.0 < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX52 %s
// PPC-AIX52:#define _AIX32 1
// PPC-AIX52:#define _AIX41 1
// PPC-AIX52:#define _AIX43 1
// PPC-AIX52:#define _AIX50 1
// PPC-AIX52:#define _AIX51 1
// PPC-AIX52:#define _AIX52 1
// PPC-AIX52-NOT:#define _AIX53 1
// PPC-AIX52-NOT:#define _AIX61 1
// PPC-AIX52-NOT:#define _AIX71 1
// PPC-AIX52-NOT:#define _AIX72 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-ibm-aix5.1.0.0 < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX51 %s
// PPC-AIX51:#define _AIX32 1
// PPC-AIX51:#define _AIX41 1
// PPC-AIX51:#define _AIX43 1
// PPC-AIX51:#define _AIX50 1
// PPC-AIX51:#define _AIX51 1
// PPC-AIX51-NOT:#define _AIX52 1
// PPC-AIX51-NOT:#define _AIX53 1
// PPC-AIX51-NOT:#define _AIX61 1
// PPC-AIX51-NOT:#define _AIX71 1
// PPC-AIX51-NOT:#define _AIX72 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-ibm-aix5.0.0.0 < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX50 %s
// PPC-AIX50:#define _AIX32 1
// PPC-AIX50:#define _AIX41 1
// PPC-AIX50:#define _AIX43 1
// PPC-AIX50:#define _AIX50 1
// PPC-AIX50-NOT:#define _AIX51 1
// PPC-AIX50-NOT:#define _AIX52 1
// PPC-AIX50-NOT:#define _AIX53 1
// PPC-AIX50-NOT:#define _AIX61 1
// PPC-AIX50-NOT:#define _AIX71 1
// PPC-AIX50-NOT:#define _AIX72 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-ibm-aix4.3.0.0 < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX43 %s
// PPC-AIX43:#define _AIX32 1
// PPC-AIX43:#define _AIX41 1
// PPC-AIX43:#define _AIX43 1
// PPC-AIX43-NOT:#define _AIX50 1
// PPC-AIX43-NOT:#define _AIX51 1
// PPC-AIX43-NOT:#define _AIX52 1
// PPC-AIX43-NOT:#define _AIX53 1
// PPC-AIX43-NOT:#define _AIX61 1
// PPC-AIX43-NOT:#define _AIX71 1
// PPC-AIX43-NOT:#define _AIX72 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-ibm-aix4.1.0.0 < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX41 %s
// PPC-AIX41:#define _AIX32 1
// PPC-AIX41:#define _AIX41 1
// PPC-AIX41-NOT:#define _AIX43 1
// PPC-AIX41-NOT:#define _AIX50 1
// PPC-AIX41-NOT:#define _AIX51 1
// PPC-AIX41-NOT:#define _AIX52 1
// PPC-AIX41-NOT:#define _AIX53 1
// PPC-AIX41-NOT:#define _AIX61 1
// PPC-AIX41-NOT:#define _AIX71 1
// PPC-AIX41-NOT:#define _AIX72 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-ibm-aix3.2.0.0 < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX32 %s
// PPC-AIX32:#define _AIX32 1
// PPC-AIX32-NOT:#define _AIX41 1
// PPC-AIX32-NOT:#define _AIX43 1
// PPC-AIX32-NOT:#define _AIX50 1
// PPC-AIX32-NOT:#define _AIX51 1
// PPC-AIX32-NOT:#define _AIX52 1
// PPC-AIX32-NOT:#define _AIX53 1
// PPC-AIX32-NOT:#define _AIX61 1
// PPC-AIX32-NOT:#define _AIX71 1
// PPC-AIX32-NOT:#define _AIX72 1

// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=powerpc-ibm-aix7.1.0.0 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX-CXX %s
//
// PPC-AIX-CXX:#define _WCHAR_T 1

// RUN: %clang_cc1 -x c++ -fno-wchar -E -dM -ffreestanding -triple=powerpc-ibm-aix7.1.0.0 < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX-NOWCHAR %s
// RUN: %clang_cc1 -x c -E -dM -ffreestanding -triple=powerpc-ibm-aix7.1.0.0 < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX-NOWCHAR %s
//
// PPC-AIX-NOWCHAR-NOT:#define _WCHAR_T 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-ibm-aix7.1.0.0 -fno-signed-char -pthread < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX-THREADSAFE %s
// PPC-AIX-THREADSAFE:#define _THREAD_SAFE 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-ibm-aix7.1.0.0 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX-NOTHREADSAFE %s
// PPC-AIX-NOTHREADSAFE-NOT:#define _THREAD_SAFE 1

// RUN: %clang_cc1 -x c -std=c11 -E -dM -ffreestanding -triple=powerpc-ibm-aix7.1.0.0 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX-STDC %s
// RUN: %clang_cc1 -x c -std=gnu11 -E -dM -ffreestanding -triple=powerpc-ibm-aix7.1.0.0 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX-STDC %s
// RUN: %clang_cc1 -x c -std=c17 -E -dM -ffreestanding -triple=powerpc-ibm-aix7.1.0.0 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX-STDC %s
// PPC-AIX-STDC:#define __STDC_NO_ATOMICS__ 1
// PPC-AIX-STDC:#define __STDC_NO_THREADS__ 1

// RUN: %clang_cc1 -x c -std=c99 -E -dM -ffreestanding -triple=powerpc-ibm-aix7.1.0.0 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPC-AIX-STDC-N %s
// PPC-AIX-STDC-N-NOT:#define __STDC_NO_ATOMICS__ 1
// PPC-AIX-STDC-N-NOT:#define __STDC_NO_THREADS__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-unknown-linux-gnu -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPC-LINUX %s
//
// PPC-LINUX:#define _ARCH_PPC 1
// PPC-LINUX:#define _BIG_ENDIAN 1
// PPC-LINUX-NOT:#define _LP64
// PPC-LINUX:#define __BIGGEST_ALIGNMENT__ 16
// PPC-LINUX:#define __BIG_ENDIAN__ 1
// PPC-LINUX:#define __BYTE_ORDER__ __ORDER_BIG_ENDIAN__
// PPC-LINUX:#define __CHAR16_TYPE__ unsigned short
// PPC-LINUX:#define __CHAR32_TYPE__ unsigned int
// PPC-LINUX:#define __CHAR_BIT__ 8
// PPC-LINUX:#define __CHAR_UNSIGNED__ 1
// PPC-LINUX:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// PPC-LINUX:#define __DBL_DIG__ 15
// PPC-LINUX:#define __DBL_EPSILON__ 2.2204460492503131e-16
// PPC-LINUX:#define __DBL_HAS_DENORM__ 1
// PPC-LINUX:#define __DBL_HAS_INFINITY__ 1
// PPC-LINUX:#define __DBL_HAS_QUIET_NAN__ 1
// PPC-LINUX:#define __DBL_MANT_DIG__ 53
// PPC-LINUX:#define __DBL_MAX_10_EXP__ 308
// PPC-LINUX:#define __DBL_MAX_EXP__ 1024
// PPC-LINUX:#define __DBL_MAX__ 1.7976931348623157e+308
// PPC-LINUX:#define __DBL_MIN_10_EXP__ (-307)
// PPC-LINUX:#define __DBL_MIN_EXP__ (-1021)
// PPC-LINUX:#define __DBL_MIN__ 2.2250738585072014e-308
// PPC-LINUX:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// PPC-LINUX:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// PPC-LINUX:#define __FLT_DIG__ 6
// PPC-LINUX:#define __FLT_EPSILON__ 1.19209290e-7F
// PPC-LINUX:#define __FLT_EVAL_METHOD__ 0
// PPC-LINUX:#define __FLT_HAS_DENORM__ 1
// PPC-LINUX:#define __FLT_HAS_INFINITY__ 1
// PPC-LINUX:#define __FLT_HAS_QUIET_NAN__ 1
// PPC-LINUX:#define __FLT_MANT_DIG__ 24
// PPC-LINUX:#define __FLT_MAX_10_EXP__ 38
// PPC-LINUX:#define __FLT_MAX_EXP__ 128
// PPC-LINUX:#define __FLT_MAX__ 3.40282347e+38F
// PPC-LINUX:#define __FLT_MIN_10_EXP__ (-37)
// PPC-LINUX:#define __FLT_MIN_EXP__ (-125)
// PPC-LINUX:#define __FLT_MIN__ 1.17549435e-38F
// PPC-LINUX:#define __FLT_RADIX__ 2
// PPC-LINUX:#define __HAVE_BSWAP__ 1
// PPC-LINUX:#define __INT16_C_SUFFIX__
// PPC-LINUX:#define __INT16_FMTd__ "hd"
// PPC-LINUX:#define __INT16_FMTi__ "hi"
// PPC-LINUX:#define __INT16_MAX__ 32767
// PPC-LINUX:#define __INT16_TYPE__ short
// PPC-LINUX:#define __INT32_C_SUFFIX__
// PPC-LINUX:#define __INT32_FMTd__ "d"
// PPC-LINUX:#define __INT32_FMTi__ "i"
// PPC-LINUX:#define __INT32_MAX__ 2147483647
// PPC-LINUX:#define __INT32_TYPE__ int
// PPC-LINUX:#define __INT64_C_SUFFIX__ LL
// PPC-LINUX:#define __INT64_FMTd__ "lld"
// PPC-LINUX:#define __INT64_FMTi__ "lli"
// PPC-LINUX:#define __INT64_MAX__ 9223372036854775807LL
// PPC-LINUX:#define __INT64_TYPE__ long long int
// PPC-LINUX:#define __INT8_C_SUFFIX__
// PPC-LINUX:#define __INT8_FMTd__ "hhd"
// PPC-LINUX:#define __INT8_FMTi__ "hhi"
// PPC-LINUX:#define __INT8_MAX__ 127
// PPC-LINUX:#define __INT8_TYPE__ signed char
// PPC-LINUX:#define __INTMAX_C_SUFFIX__ LL
// PPC-LINUX:#define __INTMAX_FMTd__ "lld"
// PPC-LINUX:#define __INTMAX_FMTi__ "lli"
// PPC-LINUX:#define __INTMAX_MAX__ 9223372036854775807LL
// PPC-LINUX:#define __INTMAX_TYPE__ long long int
// PPC-LINUX:#define __INTMAX_WIDTH__ 64
// PPC-LINUX:#define __INTPTR_FMTd__ "d"
// PPC-LINUX:#define __INTPTR_FMTi__ "i"
// PPC-LINUX:#define __INTPTR_MAX__ 2147483647
// PPC-LINUX:#define __INTPTR_TYPE__ int
// PPC-LINUX:#define __INTPTR_WIDTH__ 32
// PPC-LINUX:#define __INT_FAST16_FMTd__ "hd"
// PPC-LINUX:#define __INT_FAST16_FMTi__ "hi"
// PPC-LINUX:#define __INT_FAST16_MAX__ 32767
// PPC-LINUX:#define __INT_FAST16_TYPE__ short
// PPC-LINUX:#define __INT_FAST32_FMTd__ "d"
// PPC-LINUX:#define __INT_FAST32_FMTi__ "i"
// PPC-LINUX:#define __INT_FAST32_MAX__ 2147483647
// PPC-LINUX:#define __INT_FAST32_TYPE__ int
// PPC-LINUX:#define __INT_FAST64_FMTd__ "lld"
// PPC-LINUX:#define __INT_FAST64_FMTi__ "lli"
// PPC-LINUX:#define __INT_FAST64_MAX__ 9223372036854775807LL
// PPC-LINUX:#define __INT_FAST64_TYPE__ long long int
// PPC-LINUX:#define __INT_FAST8_FMTd__ "hhd"
// PPC-LINUX:#define __INT_FAST8_FMTi__ "hhi"
// PPC-LINUX:#define __INT_FAST8_MAX__ 127
// PPC-LINUX:#define __INT_FAST8_TYPE__ signed char
// PPC-LINUX:#define __INT_LEAST16_FMTd__ "hd"
// PPC-LINUX:#define __INT_LEAST16_FMTi__ "hi"
// PPC-LINUX:#define __INT_LEAST16_MAX__ 32767
// PPC-LINUX:#define __INT_LEAST16_TYPE__ short
// PPC-LINUX:#define __INT_LEAST32_FMTd__ "d"
// PPC-LINUX:#define __INT_LEAST32_FMTi__ "i"
// PPC-LINUX:#define __INT_LEAST32_MAX__ 2147483647
// PPC-LINUX:#define __INT_LEAST32_TYPE__ int
// PPC-LINUX:#define __INT_LEAST64_FMTd__ "lld"
// PPC-LINUX:#define __INT_LEAST64_FMTi__ "lli"
// PPC-LINUX:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// PPC-LINUX:#define __INT_LEAST64_TYPE__ long long int
// PPC-LINUX:#define __INT_LEAST8_FMTd__ "hhd"
// PPC-LINUX:#define __INT_LEAST8_FMTi__ "hhi"
// PPC-LINUX:#define __INT_LEAST8_MAX__ 127
// PPC-LINUX:#define __INT_LEAST8_TYPE__ signed char
// PPC-LINUX:#define __INT_MAX__ 2147483647
// PPC-LINUX:#define __LDBL_DENORM_MIN__ 4.94065645841246544176568792868221e-324L
// PPC-LINUX:#define __LDBL_DIG__ 31
// PPC-LINUX:#define __LDBL_EPSILON__ 4.94065645841246544176568792868221e-324L
// PPC-LINUX:#define __LDBL_HAS_DENORM__ 1
// PPC-LINUX:#define __LDBL_HAS_INFINITY__ 1
// PPC-LINUX:#define __LDBL_HAS_QUIET_NAN__ 1
// PPC-LINUX:#define __LDBL_MANT_DIG__ 106
// PPC-LINUX:#define __LDBL_MAX_10_EXP__ 308
// PPC-LINUX:#define __LDBL_MAX_EXP__ 1024
// PPC-LINUX:#define __LDBL_MAX__ 1.79769313486231580793728971405301e+308L
// PPC-LINUX:#define __LDBL_MIN_10_EXP__ (-291)
// PPC-LINUX:#define __LDBL_MIN_EXP__ (-968)
// PPC-LINUX:#define __LDBL_MIN__ 2.00416836000897277799610805135016e-292L
// PPC-LINUX:#define __LONGDOUBLE128 1
// PPC-LINUX:#define __LONG_DOUBLE_128__ 1
// PPC-LINUX:#define __LONG_LONG_MAX__ 9223372036854775807LL
// PPC-LINUX:#define __LONG_MAX__ 2147483647L
// PPC-LINUX-NOT:#define __LP64__
// PPC-LINUX:#define __NATURAL_ALIGNMENT__ 1
// PPC-LINUX:#define __POINTER_WIDTH__ 32
// PPC-LINUX:#define __POWERPC__ 1
// PPC-LINUX:#define __PPC__ 1
// PPC-LINUX:#define __PTRDIFF_TYPE__ int
// PPC-LINUX:#define __PTRDIFF_WIDTH__ 32
// PPC-LINUX:#define __REGISTER_PREFIX__
// PPC-LINUX:#define __SCHAR_MAX__ 127
// PPC-LINUX:#define __SHRT_MAX__ 32767
// PPC-LINUX:#define __SIG_ATOMIC_MAX__ 2147483647
// PPC-LINUX:#define __SIG_ATOMIC_WIDTH__ 32
// PPC-LINUX:#define __SIZEOF_DOUBLE__ 8
// PPC-LINUX:#define __SIZEOF_FLOAT__ 4
// PPC-LINUX:#define __SIZEOF_INT__ 4
// PPC-LINUX:#define __SIZEOF_LONG_DOUBLE__ 16
// PPC-LINUX:#define __SIZEOF_LONG_LONG__ 8
// PPC-LINUX:#define __SIZEOF_LONG__ 4
// PPC-LINUX:#define __SIZEOF_POINTER__ 4
// PPC-LINUX:#define __SIZEOF_PTRDIFF_T__ 4
// PPC-LINUX:#define __SIZEOF_SHORT__ 2
// PPC-LINUX:#define __SIZEOF_SIZE_T__ 4
// PPC-LINUX:#define __SIZEOF_WCHAR_T__ 4
// PPC-LINUX:#define __SIZEOF_WINT_T__ 4
// PPC-LINUX:#define __SIZE_MAX__ 4294967295U
// PPC-LINUX:#define __SIZE_TYPE__ unsigned int
// PPC-LINUX:#define __SIZE_WIDTH__ 32
// PPC-LINUX:#define __UINT16_C_SUFFIX__
// PPC-LINUX:#define __UINT16_MAX__ 65535
// PPC-LINUX:#define __UINT16_TYPE__ unsigned short
// PPC-LINUX:#define __UINT32_C_SUFFIX__ U
// PPC-LINUX:#define __UINT32_MAX__ 4294967295U
// PPC-LINUX:#define __UINT32_TYPE__ unsigned int
// PPC-LINUX:#define __UINT64_C_SUFFIX__ ULL
// PPC-LINUX:#define __UINT64_MAX__ 18446744073709551615ULL
// PPC-LINUX:#define __UINT64_TYPE__ long long unsigned int
// PPC-LINUX:#define __UINT8_C_SUFFIX__
// PPC-LINUX:#define __UINT8_MAX__ 255
// PPC-LINUX:#define __UINT8_TYPE__ unsigned char
// PPC-LINUX:#define __UINTMAX_C_SUFFIX__ ULL
// PPC-LINUX:#define __UINTMAX_MAX__ 18446744073709551615ULL
// PPC-LINUX:#define __UINTMAX_TYPE__ long long unsigned int
// PPC-LINUX:#define __UINTMAX_WIDTH__ 64
// PPC-LINUX:#define __UINTPTR_MAX__ 4294967295U
// PPC-LINUX:#define __UINTPTR_TYPE__ unsigned int
// PPC-LINUX:#define __UINTPTR_WIDTH__ 32
// PPC-LINUX:#define __UINT_FAST16_MAX__ 65535
// PPC-LINUX:#define __UINT_FAST16_TYPE__ unsigned short
// PPC-LINUX:#define __UINT_FAST32_MAX__ 4294967295U
// PPC-LINUX:#define __UINT_FAST32_TYPE__ unsigned int
// PPC-LINUX:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// PPC-LINUX:#define __UINT_FAST64_TYPE__ long long unsigned int
// PPC-LINUX:#define __UINT_FAST8_MAX__ 255
// PPC-LINUX:#define __UINT_FAST8_TYPE__ unsigned char
// PPC-LINUX:#define __UINT_LEAST16_MAX__ 65535
// PPC-LINUX:#define __UINT_LEAST16_TYPE__ unsigned short
// PPC-LINUX:#define __UINT_LEAST32_MAX__ 4294967295U
// PPC-LINUX:#define __UINT_LEAST32_TYPE__ unsigned int
// PPC-LINUX:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// PPC-LINUX:#define __UINT_LEAST64_TYPE__ long long unsigned int
// PPC-LINUX:#define __UINT_LEAST8_MAX__ 255
// PPC-LINUX:#define __UINT_LEAST8_TYPE__ unsigned char
// PPC-LINUX:#define __USER_LABEL_PREFIX__
// PPC-LINUX:#define __WCHAR_MAX__ 2147483647
// PPC-LINUX:#define __WCHAR_TYPE__ int
// PPC-LINUX:#define __WCHAR_WIDTH__ 32
// PPC-LINUX:#define __WINT_TYPE__ unsigned int
// PPC-LINUX:#define __WINT_UNSIGNED__ 1
// PPC-LINUX:#define __WINT_WIDTH__ 32
// PPC-LINUX:#define __powerpc__ 1
// PPC-LINUX:#define __ppc__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-unknown-linux-gnu -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPC32-LINUX %s
//
// PPC32-LINUX-NOT: _CALL_LINUX

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-unknown-linux-gnu -target-feature +spe < /dev/null | FileCheck -match-full-lines -check-prefix PPC32-SPE %s
//
// PPC32-SPE:#define __NO_FPRS__ 1
// PPC32-SPE:#define __SPE__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-unknown-linux-gnu -target-cpu 8548 < /dev/null | FileCheck -match-full-lines -check-prefix PPC8548 %s
//
// PPC8548:#define __NO_FPRS__ 1
// PPC8548:#define __NO_LWSYNC__ 1
// PPC8548:#define __SPE__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-apple-darwin8 < /dev/null | FileCheck -match-full-lines -check-prefix PPC-DARWIN %s
//
// PPC-DARWIN:#define _ARCH_PPC 1
// PPC-DARWIN:#define _BIG_ENDIAN 1
// PPC-DARWIN:#define __BIGGEST_ALIGNMENT__ 16
// PPC-DARWIN:#define __BIG_ENDIAN__ 1
// PPC-DARWIN:#define __BYTE_ORDER__ __ORDER_BIG_ENDIAN__
// PPC-DARWIN:#define __CHAR16_TYPE__ unsigned short
// PPC-DARWIN:#define __CHAR32_TYPE__ unsigned int
// PPC-DARWIN:#define __CHAR_BIT__ 8
// PPC-DARWIN:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// PPC-DARWIN:#define __DBL_DIG__ 15
// PPC-DARWIN:#define __DBL_EPSILON__ 2.2204460492503131e-16
// PPC-DARWIN:#define __DBL_HAS_DENORM__ 1
// PPC-DARWIN:#define __DBL_HAS_INFINITY__ 1
// PPC-DARWIN:#define __DBL_HAS_QUIET_NAN__ 1
// PPC-DARWIN:#define __DBL_MANT_DIG__ 53
// PPC-DARWIN:#define __DBL_MAX_10_EXP__ 308
// PPC-DARWIN:#define __DBL_MAX_EXP__ 1024
// PPC-DARWIN:#define __DBL_MAX__ 1.7976931348623157e+308
// PPC-DARWIN:#define __DBL_MIN_10_EXP__ (-307)
// PPC-DARWIN:#define __DBL_MIN_EXP__ (-1021)
// PPC-DARWIN:#define __DBL_MIN__ 2.2250738585072014e-308
// PPC-DARWIN:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// PPC-DARWIN:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// PPC-DARWIN:#define __FLT_DIG__ 6
// PPC-DARWIN:#define __FLT_EPSILON__ 1.19209290e-7F
// PPC-DARWIN:#define __FLT_EVAL_METHOD__ 0
// PPC-DARWIN:#define __FLT_HAS_DENORM__ 1
// PPC-DARWIN:#define __FLT_HAS_INFINITY__ 1
// PPC-DARWIN:#define __FLT_HAS_QUIET_NAN__ 1
// PPC-DARWIN:#define __FLT_MANT_DIG__ 24
// PPC-DARWIN:#define __FLT_MAX_10_EXP__ 38
// PPC-DARWIN:#define __FLT_MAX_EXP__ 128
// PPC-DARWIN:#define __FLT_MAX__ 3.40282347e+38F
// PPC-DARWIN:#define __FLT_MIN_10_EXP__ (-37)
// PPC-DARWIN:#define __FLT_MIN_EXP__ (-125)
// PPC-DARWIN:#define __FLT_MIN__ 1.17549435e-38F
// PPC-DARWIN:#define __FLT_RADIX__ 2
// PPC-DARWIN:#define __HAVE_BSWAP__ 1
// PPC-DARWIN:#define __INT16_C_SUFFIX__
// PPC-DARWIN:#define __INT16_FMTd__ "hd"
// PPC-DARWIN:#define __INT16_FMTi__ "hi"
// PPC-DARWIN:#define __INT16_MAX__ 32767
// PPC-DARWIN:#define __INT16_TYPE__ short
// PPC-DARWIN:#define __INT32_C_SUFFIX__
// PPC-DARWIN:#define __INT32_FMTd__ "d"
// PPC-DARWIN:#define __INT32_FMTi__ "i"
// PPC-DARWIN:#define __INT32_MAX__ 2147483647
// PPC-DARWIN:#define __INT32_TYPE__ int
// PPC-DARWIN:#define __INT64_C_SUFFIX__ LL
// PPC-DARWIN:#define __INT64_FMTd__ "lld"
// PPC-DARWIN:#define __INT64_FMTi__ "lli"
// PPC-DARWIN:#define __INT64_MAX__ 9223372036854775807LL
// PPC-DARWIN:#define __INT64_TYPE__ long long int
// PPC-DARWIN:#define __INT8_C_SUFFIX__
// PPC-DARWIN:#define __INT8_FMTd__ "hhd"
// PPC-DARWIN:#define __INT8_FMTi__ "hhi"
// PPC-DARWIN:#define __INT8_MAX__ 127
// PPC-DARWIN:#define __INT8_TYPE__ signed char
// PPC-DARWIN:#define __INTMAX_C_SUFFIX__ LL
// PPC-DARWIN:#define __INTMAX_FMTd__ "lld"
// PPC-DARWIN:#define __INTMAX_FMTi__ "lli"
// PPC-DARWIN:#define __INTMAX_MAX__ 9223372036854775807LL
// PPC-DARWIN:#define __INTMAX_TYPE__ long long int
// PPC-DARWIN:#define __INTMAX_WIDTH__ 64
// PPC-DARWIN:#define __INTPTR_FMTd__ "ld"
// PPC-DARWIN:#define __INTPTR_FMTi__ "li"
// PPC-DARWIN:#define __INTPTR_MAX__ 2147483647L
// PPC-DARWIN:#define __INTPTR_TYPE__ long int
// PPC-DARWIN:#define __INTPTR_WIDTH__ 32
// PPC-DARWIN:#define __INT_FAST16_FMTd__ "hd"
// PPC-DARWIN:#define __INT_FAST16_FMTi__ "hi"
// PPC-DARWIN:#define __INT_FAST16_MAX__ 32767
// PPC-DARWIN:#define __INT_FAST16_TYPE__ short
// PPC-DARWIN:#define __INT_FAST32_FMTd__ "d"
// PPC-DARWIN:#define __INT_FAST32_FMTi__ "i"
// PPC-DARWIN:#define __INT_FAST32_MAX__ 2147483647
// PPC-DARWIN:#define __INT_FAST32_TYPE__ int
// PPC-DARWIN:#define __INT_FAST64_FMTd__ "lld"
// PPC-DARWIN:#define __INT_FAST64_FMTi__ "lli"
// PPC-DARWIN:#define __INT_FAST64_MAX__ 9223372036854775807LL
// PPC-DARWIN:#define __INT_FAST64_TYPE__ long long int
// PPC-DARWIN:#define __INT_FAST8_FMTd__ "hhd"
// PPC-DARWIN:#define __INT_FAST8_FMTi__ "hhi"
// PPC-DARWIN:#define __INT_FAST8_MAX__ 127
// PPC-DARWIN:#define __INT_FAST8_TYPE__ signed char
// PPC-DARWIN:#define __INT_LEAST16_FMTd__ "hd"
// PPC-DARWIN:#define __INT_LEAST16_FMTi__ "hi"
// PPC-DARWIN:#define __INT_LEAST16_MAX__ 32767
// PPC-DARWIN:#define __INT_LEAST16_TYPE__ short
// PPC-DARWIN:#define __INT_LEAST32_FMTd__ "d"
// PPC-DARWIN:#define __INT_LEAST32_FMTi__ "i"
// PPC-DARWIN:#define __INT_LEAST32_MAX__ 2147483647
// PPC-DARWIN:#define __INT_LEAST32_TYPE__ int
// PPC-DARWIN:#define __INT_LEAST64_FMTd__ "lld"
// PPC-DARWIN:#define __INT_LEAST64_FMTi__ "lli"
// PPC-DARWIN:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// PPC-DARWIN:#define __INT_LEAST64_TYPE__ long long int
// PPC-DARWIN:#define __INT_LEAST8_FMTd__ "hhd"
// PPC-DARWIN:#define __INT_LEAST8_FMTi__ "hhi"
// PPC-DARWIN:#define __INT_LEAST8_MAX__ 127
// PPC-DARWIN:#define __INT_LEAST8_TYPE__ signed char
// PPC-DARWIN:#define __INT_MAX__ 2147483647
// PPC-DARWIN:#define __LDBL_DENORM_MIN__ 4.94065645841246544176568792868221e-324L
// PPC-DARWIN:#define __LDBL_DIG__ 31
// PPC-DARWIN:#define __LDBL_EPSILON__ 4.94065645841246544176568792868221e-324L
// PPC-DARWIN:#define __LDBL_HAS_DENORM__ 1
// PPC-DARWIN:#define __LDBL_HAS_INFINITY__ 1
// PPC-DARWIN:#define __LDBL_HAS_QUIET_NAN__ 1
// PPC-DARWIN:#define __LDBL_MANT_DIG__ 106
// PPC-DARWIN:#define __LDBL_MAX_10_EXP__ 308
// PPC-DARWIN:#define __LDBL_MAX_EXP__ 1024
// PPC-DARWIN:#define __LDBL_MAX__ 1.79769313486231580793728971405301e+308L
// PPC-DARWIN:#define __LDBL_MIN_10_EXP__ (-291)
// PPC-DARWIN:#define __LDBL_MIN_EXP__ (-968)
// PPC-DARWIN:#define __LDBL_MIN__ 2.00416836000897277799610805135016e-292L
// PPC-DARWIN:#define __LONGDOUBLE128 1
// PPC-DARWIN:#define __LONG_DOUBLE_128__ 1
// PPC-DARWIN:#define __LONG_LONG_MAX__ 9223372036854775807LL
// PPC-DARWIN:#define __LONG_MAX__ 2147483647L
// PPC-DARWIN:#define __MACH__ 1
// PPC-DARWIN:#define __NATURAL_ALIGNMENT__ 1
// PPC-DARWIN:#define __ORDER_BIG_ENDIAN__ 4321
// PPC-DARWIN:#define __ORDER_LITTLE_ENDIAN__ 1234
// PPC-DARWIN:#define __ORDER_PDP_ENDIAN__ 3412
// PPC-DARWIN:#define __POINTER_WIDTH__ 32
// PPC-DARWIN:#define __POWERPC__ 1
// PPC-DARWIN:#define __PPC__ 1
// PPC-DARWIN:#define __PTRDIFF_TYPE__ int
// PPC-DARWIN:#define __PTRDIFF_WIDTH__ 32
// PPC-DARWIN:#define __REGISTER_PREFIX__
// PPC-DARWIN:#define __SCHAR_MAX__ 127
// PPC-DARWIN:#define __SHRT_MAX__ 32767
// PPC-DARWIN:#define __SIG_ATOMIC_MAX__ 2147483647
// PPC-DARWIN:#define __SIG_ATOMIC_WIDTH__ 32
// PPC-DARWIN:#define __SIZEOF_DOUBLE__ 8
// PPC-DARWIN:#define __SIZEOF_FLOAT__ 4
// PPC-DARWIN:#define __SIZEOF_INT__ 4
// PPC-DARWIN:#define __SIZEOF_LONG_DOUBLE__ 16
// PPC-DARWIN:#define __SIZEOF_LONG_LONG__ 8
// PPC-DARWIN:#define __SIZEOF_LONG__ 4
// PPC-DARWIN:#define __SIZEOF_POINTER__ 4
// PPC-DARWIN:#define __SIZEOF_PTRDIFF_T__ 4
// PPC-DARWIN:#define __SIZEOF_SHORT__ 2
// PPC-DARWIN:#define __SIZEOF_SIZE_T__ 4
// PPC-DARWIN:#define __SIZEOF_WCHAR_T__ 4
// PPC-DARWIN:#define __SIZEOF_WINT_T__ 4
// PPC-DARWIN:#define __SIZE_MAX__ 4294967295UL
// PPC-DARWIN:#define __SIZE_TYPE__ long unsigned int
// PPC-DARWIN:#define __SIZE_WIDTH__ 32
// PPC-DARWIN:#define __STDC_HOSTED__ 0
// PPC-DARWIN:#define __STDC_VERSION__ 201710L
// PPC-DARWIN:#define __STDC__ 1
// PPC-DARWIN:#define __UINT16_C_SUFFIX__
// PPC-DARWIN:#define __UINT16_MAX__ 65535
// PPC-DARWIN:#define __UINT16_TYPE__ unsigned short
// PPC-DARWIN:#define __UINT32_C_SUFFIX__ U
// PPC-DARWIN:#define __UINT32_MAX__ 4294967295U
// PPC-DARWIN:#define __UINT32_TYPE__ unsigned int
// PPC-DARWIN:#define __UINT64_C_SUFFIX__ ULL
// PPC-DARWIN:#define __UINT64_MAX__ 18446744073709551615ULL
// PPC-DARWIN:#define __UINT64_TYPE__ long long unsigned int
// PPC-DARWIN:#define __UINT8_C_SUFFIX__
// PPC-DARWIN:#define __UINT8_MAX__ 255
// PPC-DARWIN:#define __UINT8_TYPE__ unsigned char
// PPC-DARWIN:#define __UINTMAX_C_SUFFIX__ ULL
// PPC-DARWIN:#define __UINTMAX_MAX__ 18446744073709551615ULL
// PPC-DARWIN:#define __UINTMAX_TYPE__ long long unsigned int
// PPC-DARWIN:#define __UINTMAX_WIDTH__ 64
// PPC-DARWIN:#define __UINTPTR_MAX__ 4294967295UL
// PPC-DARWIN:#define __UINTPTR_TYPE__ long unsigned int
// PPC-DARWIN:#define __UINTPTR_WIDTH__ 32
// PPC-DARWIN:#define __UINT_FAST16_MAX__ 65535
// PPC-DARWIN:#define __UINT_FAST16_TYPE__ unsigned short
// PPC-DARWIN:#define __UINT_FAST32_MAX__ 4294967295U
// PPC-DARWIN:#define __UINT_FAST32_TYPE__ unsigned int
// PPC-DARWIN:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// PPC-DARWIN:#define __UINT_FAST64_TYPE__ long long unsigned int
// PPC-DARWIN:#define __UINT_FAST8_MAX__ 255
// PPC-DARWIN:#define __UINT_FAST8_TYPE__ unsigned char
// PPC-DARWIN:#define __UINT_LEAST16_MAX__ 65535
// PPC-DARWIN:#define __UINT_LEAST16_TYPE__ unsigned short
// PPC-DARWIN:#define __UINT_LEAST32_MAX__ 4294967295U
// PPC-DARWIN:#define __UINT_LEAST32_TYPE__ unsigned int
// PPC-DARWIN:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// PPC-DARWIN:#define __UINT_LEAST64_TYPE__ long long unsigned int
// PPC-DARWIN:#define __UINT_LEAST8_MAX__ 255
// PPC-DARWIN:#define __UINT_LEAST8_TYPE__ unsigned char
// PPC-DARWIN:#define __USER_LABEL_PREFIX__ _
// PPC-DARWIN:#define __WCHAR_MAX__ 2147483647
// PPC-DARWIN:#define __WCHAR_TYPE__ int
// PPC-DARWIN:#define __WCHAR_WIDTH__ 32
// PPC-DARWIN:#define __WINT_TYPE__ int
// PPC-DARWIN:#define __WINT_WIDTH__ 32
// PPC-DARWIN:#define __powerpc__ 1
// PPC-DARWIN:#define __ppc__ 1
