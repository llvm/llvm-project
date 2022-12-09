// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu pwr7 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPC64 %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu pwr7 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPC64 -check-prefix PPC64-CXX %s
//
// PPC64:#define _ARCH_PPC 1
// PPC64:#define _ARCH_PPC64 1
// PPC64:#define _ARCH_PPCGR 1
// PPC64:#define _ARCH_PPCSQ 1
// PPC64:#define _ARCH_PWR4 1
// PPC64:#define _ARCH_PWR5 1
// PPC64:#define _ARCH_PWR6 1
// PPC64:#define _ARCH_PWR7 1
// PPC64:#define _BIG_ENDIAN 1
// PPC64:#define _LP64 1
// PPC64:#define __BIGGEST_ALIGNMENT__ 16
// PPC64:#define __BIG_ENDIAN__ 1
// PPC64:#define __BYTE_ORDER__ __ORDER_BIG_ENDIAN__
// PPC64:#define __CHAR16_TYPE__ unsigned short
// PPC64:#define __CHAR32_TYPE__ unsigned int
// PPC64:#define __CHAR_BIT__ 8
// PPC64:#define __CHAR_UNSIGNED__ 1
// PPC64:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// PPC64:#define __DBL_DIG__ 15
// PPC64:#define __DBL_EPSILON__ 2.2204460492503131e-16
// PPC64:#define __DBL_HAS_DENORM__ 1
// PPC64:#define __DBL_HAS_INFINITY__ 1
// PPC64:#define __DBL_HAS_QUIET_NAN__ 1
// PPC64:#define __DBL_MANT_DIG__ 53
// PPC64:#define __DBL_MAX_10_EXP__ 308
// PPC64:#define __DBL_MAX_EXP__ 1024
// PPC64:#define __DBL_MAX__ 1.7976931348623157e+308
// PPC64:#define __DBL_MIN_10_EXP__ (-307)
// PPC64:#define __DBL_MIN_EXP__ (-1021)
// PPC64:#define __DBL_MIN__ 2.2250738585072014e-308
// PPC64:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// PPC64:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// PPC64:#define __FLT_DIG__ 6
// PPC64:#define __FLT_EPSILON__ 1.19209290e-7F
// PPC64:#define __FLT_HAS_DENORM__ 1
// PPC64:#define __FLT_HAS_INFINITY__ 1
// PPC64:#define __FLT_HAS_QUIET_NAN__ 1
// PPC64:#define __FLT_MANT_DIG__ 24
// PPC64:#define __FLT_MAX_10_EXP__ 38
// PPC64:#define __FLT_MAX_EXP__ 128
// PPC64:#define __FLT_MAX__ 3.40282347e+38F
// PPC64:#define __FLT_MIN_10_EXP__ (-37)
// PPC64:#define __FLT_MIN_EXP__ (-125)
// PPC64:#define __FLT_MIN__ 1.17549435e-38F
// PPC64:#define __FLT_RADIX__ 2
// PPC64:#define __HAVE_BSWAP__ 1
// PPC64:#define __INT16_C_SUFFIX__
// PPC64:#define __INT16_FMTd__ "hd"
// PPC64:#define __INT16_FMTi__ "hi"
// PPC64:#define __INT16_MAX__ 32767
// PPC64:#define __INT16_TYPE__ short
// PPC64:#define __INT32_C_SUFFIX__
// PPC64:#define __INT32_FMTd__ "d"
// PPC64:#define __INT32_FMTi__ "i"
// PPC64:#define __INT32_MAX__ 2147483647
// PPC64:#define __INT32_TYPE__ int
// PPC64:#define __INT64_C_SUFFIX__ L
// PPC64:#define __INT64_FMTd__ "ld"
// PPC64:#define __INT64_FMTi__ "li"
// PPC64:#define __INT64_MAX__ 9223372036854775807L
// PPC64:#define __INT64_TYPE__ long int
// PPC64:#define __INT8_C_SUFFIX__
// PPC64:#define __INT8_FMTd__ "hhd"
// PPC64:#define __INT8_FMTi__ "hhi"
// PPC64:#define __INT8_MAX__ 127
// PPC64:#define __INT8_TYPE__ signed char
// PPC64:#define __INTMAX_C_SUFFIX__ L
// PPC64:#define __INTMAX_FMTd__ "ld"
// PPC64:#define __INTMAX_FMTi__ "li"
// PPC64:#define __INTMAX_MAX__ 9223372036854775807L
// PPC64:#define __INTMAX_TYPE__ long int
// PPC64:#define __INTMAX_WIDTH__ 64
// PPC64:#define __INTPTR_FMTd__ "ld"
// PPC64:#define __INTPTR_FMTi__ "li"
// PPC64:#define __INTPTR_MAX__ 9223372036854775807L
// PPC64:#define __INTPTR_TYPE__ long int
// PPC64:#define __INTPTR_WIDTH__ 64
// PPC64:#define __INT_FAST16_FMTd__ "hd"
// PPC64:#define __INT_FAST16_FMTi__ "hi"
// PPC64:#define __INT_FAST16_MAX__ 32767
// PPC64:#define __INT_FAST16_TYPE__ short
// PPC64:#define __INT_FAST32_FMTd__ "d"
// PPC64:#define __INT_FAST32_FMTi__ "i"
// PPC64:#define __INT_FAST32_MAX__ 2147483647
// PPC64:#define __INT_FAST32_TYPE__ int
// PPC64:#define __INT_FAST64_FMTd__ "ld"
// PPC64:#define __INT_FAST64_FMTi__ "li"
// PPC64:#define __INT_FAST64_MAX__ 9223372036854775807L
// PPC64:#define __INT_FAST64_TYPE__ long int
// PPC64:#define __INT_FAST8_FMTd__ "hhd"
// PPC64:#define __INT_FAST8_FMTi__ "hhi"
// PPC64:#define __INT_FAST8_MAX__ 127
// PPC64:#define __INT_FAST8_TYPE__ signed char
// PPC64:#define __INT_LEAST16_FMTd__ "hd"
// PPC64:#define __INT_LEAST16_FMTi__ "hi"
// PPC64:#define __INT_LEAST16_MAX__ 32767
// PPC64:#define __INT_LEAST16_TYPE__ short
// PPC64:#define __INT_LEAST32_FMTd__ "d"
// PPC64:#define __INT_LEAST32_FMTi__ "i"
// PPC64:#define __INT_LEAST32_MAX__ 2147483647
// PPC64:#define __INT_LEAST32_TYPE__ int
// PPC64:#define __INT_LEAST64_FMTd__ "ld"
// PPC64:#define __INT_LEAST64_FMTi__ "li"
// PPC64:#define __INT_LEAST64_MAX__ 9223372036854775807L
// PPC64:#define __INT_LEAST64_TYPE__ long int
// PPC64:#define __INT_LEAST8_FMTd__ "hhd"
// PPC64:#define __INT_LEAST8_FMTi__ "hhi"
// PPC64:#define __INT_LEAST8_MAX__ 127
// PPC64:#define __INT_LEAST8_TYPE__ signed char
// PPC64:#define __INT_MAX__ 2147483647
// PPC64:#define __LDBL_DENORM_MIN__ 4.94065645841246544176568792868221e-324L
// PPC64:#define __LDBL_DIG__ 31
// PPC64:#define __LDBL_EPSILON__ 4.94065645841246544176568792868221e-324L
// PPC64:#define __LDBL_HAS_DENORM__ 1
// PPC64:#define __LDBL_HAS_INFINITY__ 1
// PPC64:#define __LDBL_HAS_QUIET_NAN__ 1
// PPC64:#define __LDBL_MANT_DIG__ 106
// PPC64:#define __LDBL_MAX_10_EXP__ 308
// PPC64:#define __LDBL_MAX_EXP__ 1024
// PPC64:#define __LDBL_MAX__ 1.79769313486231580793728971405301e+308L
// PPC64:#define __LDBL_MIN_10_EXP__ (-291)
// PPC64:#define __LDBL_MIN_EXP__ (-968)
// PPC64:#define __LDBL_MIN__ 2.00416836000897277799610805135016e-292L
// PPC64:#define __LONGDOUBLE128 1
// PPC64:#define __LONG_DOUBLE_128__ 1
// PPC64:#define __LONG_LONG_MAX__ 9223372036854775807LL
// PPC64:#define __LONG_MAX__ 9223372036854775807L
// PPC64:#define __LP64__ 1
// PPC64:#define __NATURAL_ALIGNMENT__ 1
// PPC64:#define __POINTER_WIDTH__ 64
// PPC64:#define __POWERPC__ 1
// PPC64:#define __PPC64__ 1
// PPC64:#define __PPC__ 1
// PPC64:#define __PTRDIFF_TYPE__ long int
// PPC64:#define __PTRDIFF_WIDTH__ 64
// PPC64:#define __REGISTER_PREFIX__
// PPC64:#define __SCHAR_MAX__ 127
// PPC64:#define __SHRT_MAX__ 32767
// PPC64:#define __SIG_ATOMIC_MAX__ 2147483647
// PPC64:#define __SIG_ATOMIC_WIDTH__ 32
// PPC64:#define __SIZEOF_DOUBLE__ 8
// PPC64:#define __SIZEOF_FLOAT__ 4
// PPC64:#define __SIZEOF_INT__ 4
// PPC64:#define __SIZEOF_LONG_DOUBLE__ 16
// PPC64:#define __SIZEOF_LONG_LONG__ 8
// PPC64:#define __SIZEOF_LONG__ 8
// PPC64:#define __SIZEOF_POINTER__ 8
// PPC64:#define __SIZEOF_PTRDIFF_T__ 8
// PPC64:#define __SIZEOF_SHORT__ 2
// PPC64:#define __SIZEOF_SIZE_T__ 8
// PPC64:#define __SIZEOF_WCHAR_T__ 4
// PPC64:#define __SIZEOF_WINT_T__ 4
// PPC64:#define __SIZE_MAX__ 18446744073709551615UL
// PPC64:#define __SIZE_TYPE__ long unsigned int
// PPC64:#define __SIZE_WIDTH__ 64
// PPC64-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 16UL
// PPC64:#define __UINT16_C_SUFFIX__
// PPC64:#define __UINT16_MAX__ 65535
// PPC64:#define __UINT16_TYPE__ unsigned short
// PPC64:#define __UINT32_C_SUFFIX__ U
// PPC64:#define __UINT32_MAX__ 4294967295U
// PPC64:#define __UINT32_TYPE__ unsigned int
// PPC64:#define __UINT64_C_SUFFIX__ UL
// PPC64:#define __UINT64_MAX__ 18446744073709551615UL
// PPC64:#define __UINT64_TYPE__ long unsigned int
// PPC64:#define __UINT8_C_SUFFIX__
// PPC64:#define __UINT8_MAX__ 255
// PPC64:#define __UINT8_TYPE__ unsigned char
// PPC64:#define __UINTMAX_C_SUFFIX__ UL
// PPC64:#define __UINTMAX_MAX__ 18446744073709551615UL
// PPC64:#define __UINTMAX_TYPE__ long unsigned int
// PPC64:#define __UINTMAX_WIDTH__ 64
// PPC64:#define __UINTPTR_MAX__ 18446744073709551615UL
// PPC64:#define __UINTPTR_TYPE__ long unsigned int
// PPC64:#define __UINTPTR_WIDTH__ 64
// PPC64:#define __UINT_FAST16_MAX__ 65535
// PPC64:#define __UINT_FAST16_TYPE__ unsigned short
// PPC64:#define __UINT_FAST32_MAX__ 4294967295U
// PPC64:#define __UINT_FAST32_TYPE__ unsigned int
// PPC64:#define __UINT_FAST64_MAX__ 18446744073709551615UL
// PPC64:#define __UINT_FAST64_TYPE__ long unsigned int
// PPC64:#define __UINT_FAST8_MAX__ 255
// PPC64:#define __UINT_FAST8_TYPE__ unsigned char
// PPC64:#define __UINT_LEAST16_MAX__ 65535
// PPC64:#define __UINT_LEAST16_TYPE__ unsigned short
// PPC64:#define __UINT_LEAST32_MAX__ 4294967295U
// PPC64:#define __UINT_LEAST32_TYPE__ unsigned int
// PPC64:#define __UINT_LEAST64_MAX__ 18446744073709551615UL
// PPC64:#define __UINT_LEAST64_TYPE__ long unsigned int
// PPC64:#define __UINT_LEAST8_MAX__ 255
// PPC64:#define __UINT_LEAST8_TYPE__ unsigned char
// PPC64:#define __USER_LABEL_PREFIX__
// PPC64:#define __WCHAR_MAX__ 2147483647
// PPC64:#define __WCHAR_TYPE__ int
// PPC64:#define __WCHAR_WIDTH__ 32
// PPC64:#define __WINT_TYPE__ int
// PPC64:#define __WINT_WIDTH__ 32
// PPC64:#define __ppc__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64le-none-none -target-cpu pwr7 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPC64LE %s
//
// PPC64LE:#define _ARCH_PPC 1
// PPC64LE:#define _ARCH_PPC64 1
// PPC64LE:#define _ARCH_PPCGR 1
// PPC64LE:#define _ARCH_PPCSQ 1
// PPC64LE:#define _ARCH_PWR4 1
// PPC64LE:#define _ARCH_PWR5 1
// PPC64LE:#define _ARCH_PWR5X 1
// PPC64LE:#define _ARCH_PWR6 1
// PPC64LE-NOT:#define _ARCH_PWR6X 1
// PPC64LE:#define _ARCH_PWR7 1
// PPC64LE:#define _CALL_ELF 2
// PPC64LE:#define _LITTLE_ENDIAN 1
// PPC64LE:#define _LP64 1
// PPC64LE:#define __BIGGEST_ALIGNMENT__ 16
// PPC64LE:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// PPC64LE:#define __CHAR16_TYPE__ unsigned short
// PPC64LE:#define __CHAR32_TYPE__ unsigned int
// PPC64LE:#define __CHAR_BIT__ 8
// PPC64LE:#define __CHAR_UNSIGNED__ 1
// PPC64LE:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// PPC64LE:#define __DBL_DIG__ 15
// PPC64LE:#define __DBL_EPSILON__ 2.2204460492503131e-16
// PPC64LE:#define __DBL_HAS_DENORM__ 1
// PPC64LE:#define __DBL_HAS_INFINITY__ 1
// PPC64LE:#define __DBL_HAS_QUIET_NAN__ 1
// PPC64LE:#define __DBL_MANT_DIG__ 53
// PPC64LE:#define __DBL_MAX_10_EXP__ 308
// PPC64LE:#define __DBL_MAX_EXP__ 1024
// PPC64LE:#define __DBL_MAX__ 1.7976931348623157e+308
// PPC64LE:#define __DBL_MIN_10_EXP__ (-307)
// PPC64LE:#define __DBL_MIN_EXP__ (-1021)
// PPC64LE:#define __DBL_MIN__ 2.2250738585072014e-308
// PPC64LE:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// PPC64LE:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// PPC64LE:#define __FLT_DIG__ 6
// PPC64LE:#define __FLT_EPSILON__ 1.19209290e-7F
// PPC64LE:#define __FLT_HAS_DENORM__ 1
// PPC64LE:#define __FLT_HAS_INFINITY__ 1
// PPC64LE:#define __FLT_HAS_QUIET_NAN__ 1
// PPC64LE:#define __FLT_MANT_DIG__ 24
// PPC64LE:#define __FLT_MAX_10_EXP__ 38
// PPC64LE:#define __FLT_MAX_EXP__ 128
// PPC64LE:#define __FLT_MAX__ 3.40282347e+38F
// PPC64LE:#define __FLT_MIN_10_EXP__ (-37)
// PPC64LE:#define __FLT_MIN_EXP__ (-125)
// PPC64LE:#define __FLT_MIN__ 1.17549435e-38F
// PPC64LE:#define __FLT_RADIX__ 2
// PPC64LE:#define __HAVE_BSWAP__ 1
// PPC64LE:#define __INT16_C_SUFFIX__
// PPC64LE:#define __INT16_FMTd__ "hd"
// PPC64LE:#define __INT16_FMTi__ "hi"
// PPC64LE:#define __INT16_MAX__ 32767
// PPC64LE:#define __INT16_TYPE__ short
// PPC64LE:#define __INT32_C_SUFFIX__
// PPC64LE:#define __INT32_FMTd__ "d"
// PPC64LE:#define __INT32_FMTi__ "i"
// PPC64LE:#define __INT32_MAX__ 2147483647
// PPC64LE:#define __INT32_TYPE__ int
// PPC64LE:#define __INT64_C_SUFFIX__ L
// PPC64LE:#define __INT64_FMTd__ "ld"
// PPC64LE:#define __INT64_FMTi__ "li"
// PPC64LE:#define __INT64_MAX__ 9223372036854775807L
// PPC64LE:#define __INT64_TYPE__ long int
// PPC64LE:#define __INT8_C_SUFFIX__
// PPC64LE:#define __INT8_FMTd__ "hhd"
// PPC64LE:#define __INT8_FMTi__ "hhi"
// PPC64LE:#define __INT8_MAX__ 127
// PPC64LE:#define __INT8_TYPE__ signed char
// PPC64LE:#define __INTMAX_C_SUFFIX__ L
// PPC64LE:#define __INTMAX_FMTd__ "ld"
// PPC64LE:#define __INTMAX_FMTi__ "li"
// PPC64LE:#define __INTMAX_MAX__ 9223372036854775807L
// PPC64LE:#define __INTMAX_TYPE__ long int
// PPC64LE:#define __INTMAX_WIDTH__ 64
// PPC64LE:#define __INTPTR_FMTd__ "ld"
// PPC64LE:#define __INTPTR_FMTi__ "li"
// PPC64LE:#define __INTPTR_MAX__ 9223372036854775807L
// PPC64LE:#define __INTPTR_TYPE__ long int
// PPC64LE:#define __INTPTR_WIDTH__ 64
// PPC64LE:#define __INT_FAST16_FMTd__ "hd"
// PPC64LE:#define __INT_FAST16_FMTi__ "hi"
// PPC64LE:#define __INT_FAST16_MAX__ 32767
// PPC64LE:#define __INT_FAST16_TYPE__ short
// PPC64LE:#define __INT_FAST32_FMTd__ "d"
// PPC64LE:#define __INT_FAST32_FMTi__ "i"
// PPC64LE:#define __INT_FAST32_MAX__ 2147483647
// PPC64LE:#define __INT_FAST32_TYPE__ int
// PPC64LE:#define __INT_FAST64_FMTd__ "ld"
// PPC64LE:#define __INT_FAST64_FMTi__ "li"
// PPC64LE:#define __INT_FAST64_MAX__ 9223372036854775807L
// PPC64LE:#define __INT_FAST64_TYPE__ long int
// PPC64LE:#define __INT_FAST8_FMTd__ "hhd"
// PPC64LE:#define __INT_FAST8_FMTi__ "hhi"
// PPC64LE:#define __INT_FAST8_MAX__ 127
// PPC64LE:#define __INT_FAST8_TYPE__ signed char
// PPC64LE:#define __INT_LEAST16_FMTd__ "hd"
// PPC64LE:#define __INT_LEAST16_FMTi__ "hi"
// PPC64LE:#define __INT_LEAST16_MAX__ 32767
// PPC64LE:#define __INT_LEAST16_TYPE__ short
// PPC64LE:#define __INT_LEAST32_FMTd__ "d"
// PPC64LE:#define __INT_LEAST32_FMTi__ "i"
// PPC64LE:#define __INT_LEAST32_MAX__ 2147483647
// PPC64LE:#define __INT_LEAST32_TYPE__ int
// PPC64LE:#define __INT_LEAST64_FMTd__ "ld"
// PPC64LE:#define __INT_LEAST64_FMTi__ "li"
// PPC64LE:#define __INT_LEAST64_MAX__ 9223372036854775807L
// PPC64LE:#define __INT_LEAST64_TYPE__ long int
// PPC64LE:#define __INT_LEAST8_FMTd__ "hhd"
// PPC64LE:#define __INT_LEAST8_FMTi__ "hhi"
// PPC64LE:#define __INT_LEAST8_MAX__ 127
// PPC64LE:#define __INT_LEAST8_TYPE__ signed char
// PPC64LE:#define __INT_MAX__ 2147483647
// PPC64LE:#define __LDBL_DENORM_MIN__ 4.94065645841246544176568792868221e-324L
// PPC64LE:#define __LDBL_DIG__ 31
// PPC64LE:#define __LDBL_EPSILON__ 4.94065645841246544176568792868221e-324L
// PPC64LE:#define __LDBL_HAS_DENORM__ 1
// PPC64LE:#define __LDBL_HAS_INFINITY__ 1
// PPC64LE:#define __LDBL_HAS_QUIET_NAN__ 1
// PPC64LE:#define __LDBL_MANT_DIG__ 106
// PPC64LE:#define __LDBL_MAX_10_EXP__ 308
// PPC64LE:#define __LDBL_MAX_EXP__ 1024
// PPC64LE:#define __LDBL_MAX__ 1.79769313486231580793728971405301e+308L
// PPC64LE:#define __LDBL_MIN_10_EXP__ (-291)
// PPC64LE:#define __LDBL_MIN_EXP__ (-968)
// PPC64LE:#define __LDBL_MIN__ 2.00416836000897277799610805135016e-292L
// PPC64LE:#define __LITTLE_ENDIAN__ 1
// PPC64LE:#define __LONGDOUBLE128 1
// PPC64LE:#define __LONG_DOUBLE_128__ 1
// PPC64LE:#define __LONG_LONG_MAX__ 9223372036854775807LL
// PPC64LE:#define __LONG_MAX__ 9223372036854775807L
// PPC64LE:#define __LP64__ 1
// PPC64LE:#define __NATURAL_ALIGNMENT__ 1
// PPC64LE:#define __POINTER_WIDTH__ 64
// PPC64LE:#define __POWERPC__ 1
// PPC64LE:#define __PPC64__ 1
// PPC64LE:#define __PPC__ 1
// PPC64LE:#define __PTRDIFF_TYPE__ long int
// PPC64LE:#define __PTRDIFF_WIDTH__ 64
// PPC64LE:#define __REGISTER_PREFIX__
// PPC64LE:#define __SCHAR_MAX__ 127
// PPC64LE:#define __SHRT_MAX__ 32767
// PPC64LE:#define __SIG_ATOMIC_MAX__ 2147483647
// PPC64LE:#define __SIG_ATOMIC_WIDTH__ 32
// PPC64LE:#define __SIZEOF_DOUBLE__ 8
// PPC64LE:#define __SIZEOF_FLOAT__ 4
// PPC64LE:#define __SIZEOF_INT__ 4
// PPC64LE:#define __SIZEOF_LONG_DOUBLE__ 16
// PPC64LE:#define __SIZEOF_LONG_LONG__ 8
// PPC64LE:#define __SIZEOF_LONG__ 8
// PPC64LE:#define __SIZEOF_POINTER__ 8
// PPC64LE:#define __SIZEOF_PTRDIFF_T__ 8
// PPC64LE:#define __SIZEOF_SHORT__ 2
// PPC64LE:#define __SIZEOF_SIZE_T__ 8
// PPC64LE:#define __SIZEOF_WCHAR_T__ 4
// PPC64LE:#define __SIZEOF_WINT_T__ 4
// PPC64LE:#define __SIZE_MAX__ 18446744073709551615UL
// PPC64LE:#define __SIZE_TYPE__ long unsigned int
// PPC64LE:#define __SIZE_WIDTH__ 64
// PPC64LE:#define __STRUCT_PARM_ALIGN__ 16
// PPC64LE:#define __UINT16_C_SUFFIX__
// PPC64LE:#define __UINT16_MAX__ 65535
// PPC64LE:#define __UINT16_TYPE__ unsigned short
// PPC64LE:#define __UINT32_C_SUFFIX__ U
// PPC64LE:#define __UINT32_MAX__ 4294967295U
// PPC64LE:#define __UINT32_TYPE__ unsigned int
// PPC64LE:#define __UINT64_C_SUFFIX__ UL
// PPC64LE:#define __UINT64_MAX__ 18446744073709551615UL
// PPC64LE:#define __UINT64_TYPE__ long unsigned int
// PPC64LE:#define __UINT8_C_SUFFIX__
// PPC64LE:#define __UINT8_MAX__ 255
// PPC64LE:#define __UINT8_TYPE__ unsigned char
// PPC64LE:#define __UINTMAX_C_SUFFIX__ UL
// PPC64LE:#define __UINTMAX_MAX__ 18446744073709551615UL
// PPC64LE:#define __UINTMAX_TYPE__ long unsigned int
// PPC64LE:#define __UINTMAX_WIDTH__ 64
// PPC64LE:#define __UINTPTR_MAX__ 18446744073709551615UL
// PPC64LE:#define __UINTPTR_TYPE__ long unsigned int
// PPC64LE:#define __UINTPTR_WIDTH__ 64
// PPC64LE:#define __UINT_FAST16_MAX__ 65535
// PPC64LE:#define __UINT_FAST16_TYPE__ unsigned short
// PPC64LE:#define __UINT_FAST32_MAX__ 4294967295U
// PPC64LE:#define __UINT_FAST32_TYPE__ unsigned int
// PPC64LE:#define __UINT_FAST64_MAX__ 18446744073709551615UL
// PPC64LE:#define __UINT_FAST64_TYPE__ long unsigned int
// PPC64LE:#define __UINT_FAST8_MAX__ 255
// PPC64LE:#define __UINT_FAST8_TYPE__ unsigned char
// PPC64LE:#define __UINT_LEAST16_MAX__ 65535
// PPC64LE:#define __UINT_LEAST16_TYPE__ unsigned short
// PPC64LE:#define __UINT_LEAST32_MAX__ 4294967295U
// PPC64LE:#define __UINT_LEAST32_TYPE__ unsigned int
// PPC64LE:#define __UINT_LEAST64_MAX__ 18446744073709551615UL
// PPC64LE:#define __UINT_LEAST64_TYPE__ long unsigned int
// PPC64LE:#define __UINT_LEAST8_MAX__ 255
// PPC64LE:#define __UINT_LEAST8_TYPE__ unsigned char
// PPC64LE:#define __USER_LABEL_PREFIX__
// PPC64LE:#define __WCHAR_MAX__ 2147483647
// PPC64LE:#define __WCHAR_TYPE__ int
// PPC64LE:#define __WCHAR_WIDTH__ 32
// PPC64LE:#define __WINT_TYPE__ int
// PPC64LE:#define __WINT_WIDTH__ 32
// PPC64LE:#define __ppc__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu 630 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPC630 %s
//
// PPC630:#define _ARCH_630 1
// PPC630:#define _ARCH_PPC 1
// PPC630:#define _ARCH_PPC64 1
// PPC630:#define _ARCH_PPCGR 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu pwr3 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPWR3 %s
//
// PPCPWR3:#define _ARCH_PPC 1
// PPCPWR3:#define _ARCH_PPC64 1
// PPCPWR3:#define _ARCH_PPCGR 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu power3 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPOWER3 %s
//
// PPCPOWER3:#define _ARCH_PPC 1
// PPCPOWER3:#define _ARCH_PPC64 1
// PPCPOWER3:#define _ARCH_PPCGR 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu pwr4 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPWR4 %s
//
// PPCPWR4:#define _ARCH_PPC 1
// PPCPWR4:#define _ARCH_PPC64 1
// PPCPWR4:#define _ARCH_PPCGR 1
// PPCPWR4:#define _ARCH_PPCSQ 1
// PPCPWR4:#define _ARCH_PWR4 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu power4 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPOWER4 %s
//
// PPCPOWER4:#define _ARCH_PPC 1
// PPCPOWER4:#define _ARCH_PPC64 1
// PPCPOWER4:#define _ARCH_PPCGR 1
// PPCPOWER4:#define _ARCH_PPCSQ 1
// PPCPOWER4:#define _ARCH_PWR4 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu pwr5 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPWR5 %s
//
// PPCPWR5:#define _ARCH_PPC 1
// PPCPWR5:#define _ARCH_PPC64 1
// PPCPWR5:#define _ARCH_PPCGR 1
// PPCPWR5:#define _ARCH_PPCSQ 1
// PPCPWR5:#define _ARCH_PWR4 1
// PPCPWR5:#define _ARCH_PWR5 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu power5 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPOWER5 %s
//
// PPCPOWER5:#define _ARCH_PPC 1
// PPCPOWER5:#define _ARCH_PPC64 1
// PPCPOWER5:#define _ARCH_PPCGR 1
// PPCPOWER5:#define _ARCH_PPCSQ 1
// PPCPOWER5:#define _ARCH_PWR4 1
// PPCPOWER5:#define _ARCH_PWR5 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu pwr5x -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPWR5X %s
//
// PPCPWR5X:#define _ARCH_PPC 1
// PPCPWR5X:#define _ARCH_PPC64 1
// PPCPWR5X:#define _ARCH_PPCGR 1
// PPCPWR5X:#define _ARCH_PPCSQ 1
// PPCPWR5X:#define _ARCH_PWR4 1
// PPCPWR5X:#define _ARCH_PWR5 1
// PPCPWR5X:#define _ARCH_PWR5X 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu power5x -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPOWER5X %s
//
// PPCPOWER5X:#define _ARCH_PPC 1
// PPCPOWER5X:#define _ARCH_PPC64 1
// PPCPOWER5X:#define _ARCH_PPCGR 1
// PPCPOWER5X:#define _ARCH_PPCSQ 1
// PPCPOWER5X:#define _ARCH_PWR4 1
// PPCPOWER5X:#define _ARCH_PWR5 1
// PPCPOWER5X:#define _ARCH_PWR5X 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu pwr6 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPWR6 %s
//
// PPCPWR6:#define _ARCH_PPC 1
// PPCPWR6:#define _ARCH_PPC64 1
// PPCPWR6:#define _ARCH_PPCGR 1
// PPCPWR6:#define _ARCH_PPCSQ 1
// PPCPWR6:#define _ARCH_PWR4 1
// PPCPWR6:#define _ARCH_PWR5 1
// PPCPWR6:#define _ARCH_PWR5X 1
// PPCPWR6:#define _ARCH_PWR6 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu power6 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPOWER6 %s
//
// PPCPOWER6:#define _ARCH_PPC 1
// PPCPOWER6:#define _ARCH_PPC64 1
// PPCPOWER6:#define _ARCH_PPCGR 1
// PPCPOWER6:#define _ARCH_PPCSQ 1
// PPCPOWER6:#define _ARCH_PWR4 1
// PPCPOWER6:#define _ARCH_PWR5 1
// PPCPOWER6:#define _ARCH_PWR5X 1
// PPCPOWER6:#define _ARCH_PWR6 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu pwr6x -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPWR6X %s
//
// PPCPWR6X:#define _ARCH_PPC 1
// PPCPWR6X:#define _ARCH_PPC64 1
// PPCPWR6X:#define _ARCH_PPCGR 1
// PPCPWR6X:#define _ARCH_PPCSQ 1
// PPCPWR6X:#define _ARCH_PWR4 1
// PPCPWR6X:#define _ARCH_PWR5 1
// PPCPWR6X:#define _ARCH_PWR5X 1
// PPCPWR6X:#define _ARCH_PWR6 1
// PPCPWR6X:#define _ARCH_PWR6X 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu power6x -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPOWER6X %s
//
// PPCPOWER6X:#define _ARCH_PPC 1
// PPCPOWER6X:#define _ARCH_PPC64 1
// PPCPOWER6X:#define _ARCH_PPCGR 1
// PPCPOWER6X:#define _ARCH_PPCSQ 1
// PPCPOWER6X:#define _ARCH_PWR4 1
// PPCPOWER6X:#define _ARCH_PWR5 1
// PPCPOWER6X:#define _ARCH_PWR5X 1
// PPCPOWER6X:#define _ARCH_PWR6 1
// PPCPOWER6X:#define _ARCH_PWR6X 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu pwr7 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPWR7 %s
//
// PPCPWR7:#define _ARCH_PPC 1
// PPCPWR7:#define _ARCH_PPC64 1
// PPCPWR7:#define _ARCH_PPCGR 1
// PPCPWR7:#define _ARCH_PPCSQ 1
// PPCPWR7:#define _ARCH_PWR4 1
// PPCPWR7:#define _ARCH_PWR5 1
// PPCPWR7:#define _ARCH_PWR5X 1
// PPCPWR7:#define _ARCH_PWR6 1
// PPCPWR7-NOT:#define _ARCH_PWR6X 1
// PPCPWR7:#define _ARCH_PWR7 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu power7 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPOWER7 %s
//
// PPCPOWER7:#define _ARCH_PPC 1
// PPCPOWER7:#define _ARCH_PPC64 1
// PPCPOWER7:#define _ARCH_PPCGR 1
// PPCPOWER7:#define _ARCH_PPCSQ 1
// PPCPOWER7:#define _ARCH_PWR4 1
// PPCPOWER7:#define _ARCH_PWR5 1
// PPCPOWER7:#define _ARCH_PWR5X 1
// PPCPOWER7:#define _ARCH_PWR6 1
// PPCPOWER7-NOT:#define _ARCH_PWR6X 1
// PPCPOWER7:#define _ARCH_PWR7 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu pwr8 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPWR8 %s
//
// PPCPWR8:#define _ARCH_PPC 1
// PPCPWR8:#define _ARCH_PPC64 1
// PPCPWR8:#define _ARCH_PPCGR 1
// PPCPWR8:#define _ARCH_PPCSQ 1
// PPCPWR8:#define _ARCH_PWR4 1
// PPCPWR8:#define _ARCH_PWR5 1
// PPCPWR8:#define _ARCH_PWR5X 1
// PPCPWR8:#define _ARCH_PWR6 1
// PPCPWR8-NOT:#define _ARCH_PWR6X 1
// PPCPWR8:#define _ARCH_PWR7 1
// PPCPWR8:#define _ARCH_PWR8 1
// PPCPWR8-NOT:#define __ROP_PROTECT__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu power8 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPOWER8 %s
//
// ppc64le also defaults to power8.
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64le-none-none -target-cpu ppc64le -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPOWER8 %s
//
// PPCPOWER8:#define _ARCH_PPC 1
// PPCPOWER8:#define _ARCH_PPC64 1
// PPCPOWER8:#define _ARCH_PPCGR 1
// PPCPOWER8:#define _ARCH_PPCSQ 1
// PPCPOWER8:#define _ARCH_PWR4 1
// PPCPOWER8:#define _ARCH_PWR5 1
// PPCPOWER8:#define _ARCH_PWR5X 1
// PPCPOWER8:#define _ARCH_PWR6 1
// PPCPOWER8-NOT:#define _ARCH_PWR6X 1
// PPCPOWER8:#define _ARCH_PWR7 1
// PPCPOWER8:#define _ARCH_PWR8 1
// PPCPOWER8-NOT:#define __ROP_PROTECT__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu pwr9 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPWR9 %s
//
// PPCPWR9:#define _ARCH_PPC 1
// PPCPWR9:#define _ARCH_PPC64 1
// PPCPWR9:#define _ARCH_PPCGR 1
// PPCPWR9:#define _ARCH_PPCSQ 1
// PPCPWR9:#define _ARCH_PWR4 1
// PPCPWR9:#define _ARCH_PWR5 1
// PPCPWR9:#define _ARCH_PWR5X 1
// PPCPWR9:#define _ARCH_PWR6 1
// PPCPWR9-NOT:#define _ARCH_PWR6X 1
// PPCPWR9:#define _ARCH_PWR7 1
// PPCPWR9:#define _ARCH_PWR9 1
// PPCPWR9-NOT:#define __ROP_PROTECT__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu power9 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPOWER9 %s
//
// PPCPOWER9:#define _ARCH_PPC 1
// PPCPOWER9:#define _ARCH_PPC64 1
// PPCPOWER9:#define _ARCH_PPCGR 1
// PPCPOWER9:#define _ARCH_PPCSQ 1
// PPCPOWER9:#define _ARCH_PWR4 1
// PPCPOWER9:#define _ARCH_PWR5 1
// PPCPOWER9:#define _ARCH_PWR5X 1
// PPCPOWER9:#define _ARCH_PWR6 1
// PPCPOWER9-NOT:#define _ARCH_PWR6X 1
// PPCPOWER9:#define _ARCH_PWR7 1
// PPCPOWER9:#define _ARCH_PWR9 1
// PPCPOWER9-NOT:#define __ROP_PROTECT__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu pwr10 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPOWER10 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu power10 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCPOWER10 %s
//
// PPCPOWER10:#define _ARCH_PPC 1
// PPCPOWER10:#define _ARCH_PPC64 1
// PPCPOWER10:#define _ARCH_PPCGR 1
// PPCPOWER10:#define _ARCH_PPCSQ 1
// PPCPOWER10:#define _ARCH_PWR10 1
// PPCPOWER10:#define _ARCH_PWR4 1
// PPCPOWER10:#define _ARCH_PWR5 1
// PPCPOWER10:#define _ARCH_PWR5X 1
// PPCPOWER10:#define _ARCH_PWR6 1
// PPCPOWER10-NOT:#define _ARCH_PWR6X 1
// PPCPOWER10:#define _ARCH_PWR7 1
// PPCPOWER10:#define _ARCH_PWR8 1
// PPCPOWER10:#define _ARCH_PWR9 1
// PPCPOWER10:#define __MMA__ 1
// PPCPOWER10:#define __PCREL__ 1
// PPCPOWER10-NOT:#define __ROP_PROTECT__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu future -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCFUTURE %s
//
// PPCFUTURE:#define _ARCH_PPC 1
// PPCFUTURE:#define _ARCH_PPC64 1
// PPCFUTURE:#define _ARCH_PPCGR 1
// PPCFUTURE:#define _ARCH_PPCSQ 1
// PPCFUTURE:#define _ARCH_PWR10 1
// PPCFUTURE:#define _ARCH_PWR4 1
// PPCFUTURE:#define _ARCH_PWR5 1
// PPCFUTURE:#define _ARCH_PWR5X 1
// PPCFUTURE:#define _ARCH_PWR6 1
// PPCFUTURE-NOT:#define _ARCH_PWR6X 1
// PPCFUTURE:#define _ARCH_PWR7 1
// PPCFUTURE:#define _ARCH_PWR8 1
// PPCFUTURE:#define _ARCH_PWR9 1
// PPCFUTURE:#define _ARCH_PWR_FUTURE 1
// PPCFUTURE:#define __MMA__ 1
// PPCFUTURE:#define __PCREL__ 1
// PPCFUTURE-NOT:#define __ROP_PROTECT__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-feature +mma -target-cpu power10 -fno-signed-char < /dev/null | FileCheck -check-prefix PPC-MMA %s
// PPC-MMA:#define __MMA__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-feature +rop-protect -target-cpu power10 -fno-signed-char < /dev/null | FileCheck -check-prefix PPC-ROP %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-feature +rop-protect -target-cpu power9 -fno-signed-char < /dev/null | FileCheck -check-prefix PPC-ROP %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-feature +rop-protect -target-cpu power8 -fno-signed-char < /dev/null | FileCheck -check-prefix PPC-ROP %s
// PPC-ROP:#define __ROP_PROTECT__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-feature +float128 -target-cpu power9 -fno-signed-char < /dev/null | FileCheck -check-prefix PPC-FLOAT128 %s
// PPC-FLOAT128:#define __FLOAT128__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-ibm-aix7.1.0.0 -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPC64-AIX %s
//
// PPC64-AIX:#define _AIX 1
// PPC64-AIX:#define _ARCH_PPC 1
// PPC64-AIX:#define _ARCH_PPC64 1
// PPC64-AIX:#define _BIG_ENDIAN 1
// PPC64-AIX:#define _IBMR2 1
// PPC64-AIX-NOT:#define _ILP32 1
// PPC64-AIX:#define _LONG_LONG 1
// PPC64-AIX:#define _LP64 1
// PPC64-AIX:#define _POWER 1
// PPC64-AIX:#define __64BIT__ 1
// PPC64-AIX:#define __BIGGEST_ALIGNMENT__ 16
// PPC64-AIX:#define __BIG_ENDIAN__ 1
// PPC64-AIX:#define __BYTE_ORDER__ __ORDER_BIG_ENDIAN__
// PPC64-AIX:#define __CHAR16_TYPE__ unsigned short
// PPC64-AIX:#define __CHAR32_TYPE__ unsigned int
// PPC64-AIX:#define __CHAR_BIT__ 8
// PPC64-AIX:#define __CHAR_UNSIGNED__ 1
// PPC64-AIX:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// PPC64-AIX:#define __DBL_DIG__ 15
// PPC64-AIX:#define __DBL_EPSILON__ 2.2204460492503131e-16
// PPC64-AIX:#define __DBL_HAS_DENORM__ 1
// PPC64-AIX:#define __DBL_HAS_INFINITY__ 1
// PPC64-AIX:#define __DBL_HAS_QUIET_NAN__ 1
// PPC64-AIX:#define __DBL_MANT_DIG__ 53
// PPC64-AIX:#define __DBL_MAX_10_EXP__ 308
// PPC64-AIX:#define __DBL_MAX_EXP__ 1024
// PPC64-AIX:#define __DBL_MAX__ 1.7976931348623157e+308
// PPC64-AIX:#define __DBL_MIN_10_EXP__ (-307)
// PPC64-AIX:#define __DBL_MIN_EXP__ (-1021)
// PPC64-AIX:#define __DBL_MIN__ 2.2250738585072014e-308
// PPC64-AIX:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// PPC64-AIX:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// PPC64-AIX:#define __FLT_DIG__ 6
// PPC64-AIX:#define __FLT_EPSILON__ 1.19209290e-7F
// PPC64-AIX:#define __FLT_HAS_DENORM__ 1
// PPC64-AIX:#define __FLT_HAS_INFINITY__ 1
// PPC64-AIX:#define __FLT_HAS_QUIET_NAN__ 1
// PPC64-AIX:#define __FLT_MANT_DIG__ 24
// PPC64-AIX:#define __FLT_MAX_10_EXP__ 38
// PPC64-AIX:#define __FLT_MAX_EXP__ 128
// PPC64-AIX:#define __FLT_MAX__ 3.40282347e+38F
// PPC64-AIX:#define __FLT_MIN_10_EXP__ (-37)
// PPC64-AIX:#define __FLT_MIN_EXP__ (-125)
// PPC64-AIX:#define __FLT_MIN__ 1.17549435e-38F
// PPC64-AIX:#define __FLT_RADIX__ 2
// PPC64-AIX-NOT:#define __ILP32__ 1
// PPC64-AIX:#define __INT16_C_SUFFIX__
// PPC64-AIX:#define __INT16_FMTd__ "hd"
// PPC64-AIX:#define __INT16_FMTi__ "hi"
// PPC64-AIX:#define __INT16_MAX__ 32767
// PPC64-AIX:#define __INT16_TYPE__ short
// PPC64-AIX:#define __INT32_C_SUFFIX__
// PPC64-AIX:#define __INT32_FMTd__ "d"
// PPC64-AIX:#define __INT32_FMTi__ "i"
// PPC64-AIX:#define __INT32_MAX__ 2147483647
// PPC64-AIX:#define __INT32_TYPE__ int
// PPC64-AIX:#define __INT64_C_SUFFIX__ L
// PPC64-AIX:#define __INT64_FMTd__ "ld"
// PPC64-AIX:#define __INT64_FMTi__ "li"
// PPC64-AIX:#define __INT64_MAX__ 9223372036854775807L
// PPC64-AIX:#define __INT64_TYPE__ long int
// PPC64-AIX:#define __INT8_C_SUFFIX__
// PPC64-AIX:#define __INT8_FMTd__ "hhd"
// PPC64-AIX:#define __INT8_FMTi__ "hhi"
// PPC64-AIX:#define __INT8_MAX__ 127
// PPC64-AIX:#define __INT8_TYPE__ signed char
// PPC64-AIX:#define __INTMAX_C_SUFFIX__ L
// PPC64-AIX:#define __INTMAX_FMTd__ "ld"
// PPC64-AIX:#define __INTMAX_FMTi__ "li"
// PPC64-AIX:#define __INTMAX_MAX__ 9223372036854775807L
// PPC64-AIX:#define __INTMAX_TYPE__ long int
// PPC64-AIX:#define __INTMAX_WIDTH__ 64
// PPC64-AIX:#define __INTPTR_FMTd__ "ld"
// PPC64-AIX:#define __INTPTR_FMTi__ "li"
// PPC64-AIX:#define __INTPTR_MAX__ 9223372036854775807L
// PPC64-AIX:#define __INTPTR_TYPE__ long int
// PPC64-AIX:#define __INTPTR_WIDTH__ 64
// PPC64-AIX:#define __INT_FAST16_FMTd__ "hd"
// PPC64-AIX:#define __INT_FAST16_FMTi__ "hi"
// PPC64-AIX:#define __INT_FAST16_MAX__ 32767
// PPC64-AIX:#define __INT_FAST16_TYPE__ short
// PPC64-AIX:#define __INT_FAST32_FMTd__ "d"
// PPC64-AIX:#define __INT_FAST32_FMTi__ "i"
// PPC64-AIX:#define __INT_FAST32_MAX__ 2147483647
// PPC64-AIX:#define __INT_FAST32_TYPE__ int
// PPC64-AIX:#define __INT_FAST64_FMTd__ "ld"
// PPC64-AIX:#define __INT_FAST64_FMTi__ "li"
// PPC64-AIX:#define __INT_FAST64_MAX__ 9223372036854775807L
// PPC64-AIX:#define __INT_FAST64_TYPE__ long int
// PPC64-AIX:#define __INT_FAST8_FMTd__ "hhd"
// PPC64-AIX:#define __INT_FAST8_FMTi__ "hhi"
// PPC64-AIX:#define __INT_FAST8_MAX__ 127
// PPC64-AIX:#define __INT_FAST8_TYPE__ signed char
// PPC64-AIX:#define __INT_LEAST16_FMTd__ "hd"
// PPC64-AIX:#define __INT_LEAST16_FMTi__ "hi"
// PPC64-AIX:#define __INT_LEAST16_MAX__ 32767
// PPC64-AIX:#define __INT_LEAST16_TYPE__ short
// PPC64-AIX:#define __INT_LEAST32_FMTd__ "d"
// PPC64-AIX:#define __INT_LEAST32_FMTi__ "i"
// PPC64-AIX:#define __INT_LEAST32_MAX__ 2147483647
// PPC64-AIX:#define __INT_LEAST32_TYPE__ int
// PPC64-AIX:#define __INT_LEAST64_FMTd__ "ld"
// PPC64-AIX:#define __INT_LEAST64_FMTi__ "li"
// PPC64-AIX:#define __INT_LEAST64_MAX__ 9223372036854775807L
// PPC64-AIX:#define __INT_LEAST64_TYPE__ long int
// PPC64-AIX:#define __INT_LEAST8_FMTd__ "hhd"
// PPC64-AIX:#define __INT_LEAST8_FMTi__ "hhi"
// PPC64-AIX:#define __INT_LEAST8_MAX__ 127
// PPC64-AIX:#define __INT_LEAST8_TYPE__ signed char
// PPC64-AIX:#define __INT_MAX__ 2147483647
// PPC64-AIX:#define __LDBL_DECIMAL_DIG__ 17
// PPC64-AIX:#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324L
// PPC64-AIX:#define __LDBL_DIG__ 15
// PPC64-AIX:#define __LDBL_EPSILON__ 2.2204460492503131e-16L
// PPC64-AIX:#define __LDBL_HAS_DENORM__ 1
// PPC64-AIX:#define __LDBL_HAS_INFINITY__ 1
// PPC64-AIX:#define __LDBL_HAS_QUIET_NAN__ 1
// PPC64-AIX:#define __LDBL_MANT_DIG__ 53
// PPC64-AIX:#define __LDBL_MAX_10_EXP__ 308
// PPC64-AIX:#define __LDBL_MAX_EXP__ 1024
// PPC64-AIX:#define __LDBL_MAX__ 1.7976931348623157e+308L
// PPC64-AIX:#define __LDBL_MIN_10_EXP__ (-307)
// PPC64-AIX:#define __LDBL_MIN_EXP__ (-1021)
// PPC64-AIX:#define __LDBL_MIN__ 2.2250738585072014e-308L
// PPC64-AIX:#define __LONG_LONG_MAX__ 9223372036854775807LL
// PPC64-AIX:#define __LONG_MAX__ 9223372036854775807L
// PPC64-AIX:#define __LP64__ 1
// PPC64-AIX-NOT:#define __NATURAL_ALIGNMENT__ 1
// PPC64-AIX:#define __POINTER_WIDTH__ 64
// PPC64-AIX:#define __POWERPC__ 1
// PPC64-AIX:#define __PPC64__ 1
// PPC64-AIX:#define __PPC__ 1
// PPC64-AIX:#define __PTRDIFF_TYPE__ long int
// PPC64-AIX:#define __PTRDIFF_WIDTH__ 64
// PPC64-AIX:#define __REGISTER_PREFIX__
// PPC64-AIX:#define __SCHAR_MAX__ 127
// PPC64-AIX:#define __SHRT_MAX__ 32767
// PPC64-AIX:#define __SIG_ATOMIC_MAX__ 2147483647
// PPC64-AIX:#define __SIG_ATOMIC_WIDTH__ 32
// PPC64-AIX:#define __SIZEOF_DOUBLE__ 8
// PPC64-AIX:#define __SIZEOF_FLOAT__ 4
// PPC64-AIX:#define __SIZEOF_INT128__ 16
// PPC64-AIX:#define __SIZEOF_INT__ 4
// PPC64-AIX:#define __SIZEOF_LONG_DOUBLE__ 8
// PPC64-AIX:#define __SIZEOF_LONG_LONG__ 8
// PPC64-AIX:#define __SIZEOF_LONG__ 8
// PPC64-AIX:#define __SIZEOF_POINTER__ 8
// PPC64-AIX:#define __SIZEOF_PTRDIFF_T__ 8
// PPC64-AIX:#define __SIZEOF_SHORT__ 2
// PPC64-AIX:#define __SIZEOF_SIZE_T__ 8
// PPC64-AIX:#define __SIZEOF_WCHAR_T__ 4
// PPC64-AIX:#define __SIZEOF_WINT_T__ 4
// PPC64-AIX:#define __SIZE_MAX__ 18446744073709551615UL
// PPC64-AIX:#define __SIZE_TYPE__ long unsigned int
// PPC64-AIX:#define __SIZE_WIDTH__ 64
// PPC64-AIX:#define __UINT16_C_SUFFIX__
// PPC64-AIX:#define __UINT16_MAX__ 65535
// PPC64-AIX:#define __UINT16_TYPE__ unsigned short
// PPC64-AIX:#define __UINT32_C_SUFFIX__ U
// PPC64-AIX:#define __UINT32_MAX__ 4294967295U
// PPC64-AIX:#define __UINT32_TYPE__ unsigned int
// PPC64-AIX:#define __UINT64_C_SUFFIX__ UL
// PPC64-AIX:#define __UINT64_MAX__ 18446744073709551615UL
// PPC64-AIX:#define __UINT64_TYPE__ long unsigned int
// PPC64-AIX:#define __UINT8_C_SUFFIX__
// PPC64-AIX:#define __UINT8_MAX__ 255
// PPC64-AIX:#define __UINT8_TYPE__ unsigned char
// PPC64-AIX:#define __UINTMAX_C_SUFFIX__ UL
// PPC64-AIX:#define __UINTMAX_MAX__ 18446744073709551615UL
// PPC64-AIX:#define __UINTMAX_TYPE__ long unsigned int
// PPC64-AIX:#define __UINTMAX_WIDTH__ 64
// PPC64-AIX:#define __UINTPTR_MAX__ 18446744073709551615UL
// PPC64-AIX:#define __UINTPTR_TYPE__ long unsigned int
// PPC64-AIX:#define __UINTPTR_WIDTH__ 64
// PPC64-AIX:#define __UINT_FAST16_MAX__ 65535
// PPC64-AIX:#define __UINT_FAST16_TYPE__ unsigned short
// PPC64-AIX:#define __UINT_FAST32_MAX__ 4294967295U
// PPC64-AIX:#define __UINT_FAST32_TYPE__ unsigned int
// PPC64-AIX:#define __UINT_FAST64_MAX__ 18446744073709551615UL
// PPC64-AIX:#define __UINT_FAST64_TYPE__ long unsigned int
// PPC64-AIX:#define __UINT_FAST8_MAX__ 255
// PPC64-AIX:#define __UINT_FAST8_TYPE__ unsigned char
// PPC64-AIX:#define __UINT_LEAST16_MAX__ 65535
// PPC64-AIX:#define __UINT_LEAST16_TYPE__ unsigned short
// PPC64-AIX:#define __UINT_LEAST32_MAX__ 4294967295U
// PPC64-AIX:#define __UINT_LEAST32_TYPE__ unsigned int
// PPC64-AIX:#define __UINT_LEAST64_MAX__ 18446744073709551615UL
// PPC64-AIX:#define __UINT_LEAST64_TYPE__ long unsigned int
// PPC64-AIX:#define __UINT_LEAST8_MAX__ 255
// PPC64-AIX:#define __UINT_LEAST8_TYPE__ unsigned char
// PPC64-AIX:#define __USER_LABEL_PREFIX__
// PPC64-AIX:#define __WCHAR_MAX__ 4294967295U
// PPC64-AIX:#define __WCHAR_TYPE__ unsigned int
// PPC64-AIX:#define __WCHAR_WIDTH__ 32
// PPC64-AIX:#define __WINT_TYPE__ int
// PPC64-AIX:#define __WINT_WIDTH__ 32
// PPC64-AIX:#define __powerpc64__ 1
// PPC64-AIX:#define __powerpc__ 1
// PPC64-AIX:#define __ppc__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-unknown-linux-gnu -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPC64-LINUX %s
//
// PPC64-LINUX:#define _ARCH_PPC 1
// PPC64-LINUX:#define _ARCH_PPC64 1
// PPC64-LINUX:#define _BIG_ENDIAN 1
// PPC64-LINUX:#define _CALL_LINUX 1
// PPC64-LINUX:#define _LP64 1
// PPC64-LINUX:#define __BIGGEST_ALIGNMENT__ 16
// PPC64-LINUX:#define __BIG_ENDIAN__ 1
// PPC64-LINUX:#define __BYTE_ORDER__ __ORDER_BIG_ENDIAN__
// PPC64-LINUX:#define __CHAR16_TYPE__ unsigned short
// PPC64-LINUX:#define __CHAR32_TYPE__ unsigned int
// PPC64-LINUX:#define __CHAR_BIT__ 8
// PPC64-LINUX:#define __CHAR_UNSIGNED__ 1
// PPC64-LINUX:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// PPC64-LINUX:#define __DBL_DIG__ 15
// PPC64-LINUX:#define __DBL_EPSILON__ 2.2204460492503131e-16
// PPC64-LINUX:#define __DBL_HAS_DENORM__ 1
// PPC64-LINUX:#define __DBL_HAS_INFINITY__ 1
// PPC64-LINUX:#define __DBL_HAS_QUIET_NAN__ 1
// PPC64-LINUX:#define __DBL_MANT_DIG__ 53
// PPC64-LINUX:#define __DBL_MAX_10_EXP__ 308
// PPC64-LINUX:#define __DBL_MAX_EXP__ 1024
// PPC64-LINUX:#define __DBL_MAX__ 1.7976931348623157e+308
// PPC64-LINUX:#define __DBL_MIN_10_EXP__ (-307)
// PPC64-LINUX:#define __DBL_MIN_EXP__ (-1021)
// PPC64-LINUX:#define __DBL_MIN__ 2.2250738585072014e-308
// PPC64-LINUX:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// PPC64-LINUX:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// PPC64-LINUX:#define __FLT_DIG__ 6
// PPC64-LINUX:#define __FLT_EPSILON__ 1.19209290e-7F
// PPC64-LINUX:#define __FLT_HAS_DENORM__ 1
// PPC64-LINUX:#define __FLT_HAS_INFINITY__ 1
// PPC64-LINUX:#define __FLT_HAS_QUIET_NAN__ 1
// PPC64-LINUX:#define __FLT_MANT_DIG__ 24
// PPC64-LINUX:#define __FLT_MAX_10_EXP__ 38
// PPC64-LINUX:#define __FLT_MAX_EXP__ 128
// PPC64-LINUX:#define __FLT_MAX__ 3.40282347e+38F
// PPC64-LINUX:#define __FLT_MIN_10_EXP__ (-37)
// PPC64-LINUX:#define __FLT_MIN_EXP__ (-125)
// PPC64-LINUX:#define __FLT_MIN__ 1.17549435e-38F
// PPC64-LINUX:#define __FLT_RADIX__ 2
// PPC64-LINUX:#define __HAVE_BSWAP__ 1
// PPC64-LINUX:#define __INT16_C_SUFFIX__
// PPC64-LINUX:#define __INT16_FMTd__ "hd"
// PPC64-LINUX:#define __INT16_FMTi__ "hi"
// PPC64-LINUX:#define __INT16_MAX__ 32767
// PPC64-LINUX:#define __INT16_TYPE__ short
// PPC64-LINUX:#define __INT32_C_SUFFIX__
// PPC64-LINUX:#define __INT32_FMTd__ "d"
// PPC64-LINUX:#define __INT32_FMTi__ "i"
// PPC64-LINUX:#define __INT32_MAX__ 2147483647
// PPC64-LINUX:#define __INT32_TYPE__ int
// PPC64-LINUX:#define __INT64_C_SUFFIX__ L
// PPC64-LINUX:#define __INT64_FMTd__ "ld"
// PPC64-LINUX:#define __INT64_FMTi__ "li"
// PPC64-LINUX:#define __INT64_MAX__ 9223372036854775807L
// PPC64-LINUX:#define __INT64_TYPE__ long int
// PPC64-LINUX:#define __INT8_C_SUFFIX__
// PPC64-LINUX:#define __INT8_FMTd__ "hhd"
// PPC64-LINUX:#define __INT8_FMTi__ "hhi"
// PPC64-LINUX:#define __INT8_MAX__ 127
// PPC64-LINUX:#define __INT8_TYPE__ signed char
// PPC64-LINUX:#define __INTMAX_C_SUFFIX__ L
// PPC64-LINUX:#define __INTMAX_FMTd__ "ld"
// PPC64-LINUX:#define __INTMAX_FMTi__ "li"
// PPC64-LINUX:#define __INTMAX_MAX__ 9223372036854775807L
// PPC64-LINUX:#define __INTMAX_TYPE__ long int
// PPC64-LINUX:#define __INTMAX_WIDTH__ 64
// PPC64-LINUX:#define __INTPTR_FMTd__ "ld"
// PPC64-LINUX:#define __INTPTR_FMTi__ "li"
// PPC64-LINUX:#define __INTPTR_MAX__ 9223372036854775807L
// PPC64-LINUX:#define __INTPTR_TYPE__ long int
// PPC64-LINUX:#define __INTPTR_WIDTH__ 64
// PPC64-LINUX:#define __INT_FAST16_FMTd__ "hd"
// PPC64-LINUX:#define __INT_FAST16_FMTi__ "hi"
// PPC64-LINUX:#define __INT_FAST16_MAX__ 32767
// PPC64-LINUX:#define __INT_FAST16_TYPE__ short
// PPC64-LINUX:#define __INT_FAST32_FMTd__ "d"
// PPC64-LINUX:#define __INT_FAST32_FMTi__ "i"
// PPC64-LINUX:#define __INT_FAST32_MAX__ 2147483647
// PPC64-LINUX:#define __INT_FAST32_TYPE__ int
// PPC64-LINUX:#define __INT_FAST64_FMTd__ "ld"
// PPC64-LINUX:#define __INT_FAST64_FMTi__ "li"
// PPC64-LINUX:#define __INT_FAST64_MAX__ 9223372036854775807L
// PPC64-LINUX:#define __INT_FAST64_TYPE__ long int
// PPC64-LINUX:#define __INT_FAST8_FMTd__ "hhd"
// PPC64-LINUX:#define __INT_FAST8_FMTi__ "hhi"
// PPC64-LINUX:#define __INT_FAST8_MAX__ 127
// PPC64-LINUX:#define __INT_FAST8_TYPE__ signed char
// PPC64-LINUX:#define __INT_LEAST16_FMTd__ "hd"
// PPC64-LINUX:#define __INT_LEAST16_FMTi__ "hi"
// PPC64-LINUX:#define __INT_LEAST16_MAX__ 32767
// PPC64-LINUX:#define __INT_LEAST16_TYPE__ short
// PPC64-LINUX:#define __INT_LEAST32_FMTd__ "d"
// PPC64-LINUX:#define __INT_LEAST32_FMTi__ "i"
// PPC64-LINUX:#define __INT_LEAST32_MAX__ 2147483647
// PPC64-LINUX:#define __INT_LEAST32_TYPE__ int
// PPC64-LINUX:#define __INT_LEAST64_FMTd__ "ld"
// PPC64-LINUX:#define __INT_LEAST64_FMTi__ "li"
// PPC64-LINUX:#define __INT_LEAST64_MAX__ 9223372036854775807L
// PPC64-LINUX:#define __INT_LEAST64_TYPE__ long int
// PPC64-LINUX:#define __INT_LEAST8_FMTd__ "hhd"
// PPC64-LINUX:#define __INT_LEAST8_FMTi__ "hhi"
// PPC64-LINUX:#define __INT_LEAST8_MAX__ 127
// PPC64-LINUX:#define __INT_LEAST8_TYPE__ signed char
// PPC64-LINUX:#define __INT_MAX__ 2147483647
// PPC64-LINUX:#define __LDBL_DENORM_MIN__ 4.94065645841246544176568792868221e-324L
// PPC64-LINUX:#define __LDBL_DIG__ 31
// PPC64-LINUX:#define __LDBL_EPSILON__ 4.94065645841246544176568792868221e-324L
// PPC64-LINUX:#define __LDBL_HAS_DENORM__ 1
// PPC64-LINUX:#define __LDBL_HAS_INFINITY__ 1
// PPC64-LINUX:#define __LDBL_HAS_QUIET_NAN__ 1
// PPC64-LINUX:#define __LDBL_MANT_DIG__ 106
// PPC64-LINUX:#define __LDBL_MAX_10_EXP__ 308
// PPC64-LINUX:#define __LDBL_MAX_EXP__ 1024
// PPC64-LINUX:#define __LDBL_MAX__ 1.79769313486231580793728971405301e+308L
// PPC64-LINUX:#define __LDBL_MIN_10_EXP__ (-291)
// PPC64-LINUX:#define __LDBL_MIN_EXP__ (-968)
// PPC64-LINUX:#define __LDBL_MIN__ 2.00416836000897277799610805135016e-292L
// PPC64-LINUX:#define __LONGDOUBLE128 1
// PPC64-LINUX:#define __LONG_DOUBLE_128__ 1
// PPC64-LINUX:#define __LONG_LONG_MAX__ 9223372036854775807LL
// PPC64-LINUX:#define __LONG_MAX__ 9223372036854775807L
// PPC64-LINUX:#define __LP64__ 1
// PPC64-LINUX:#define __NATURAL_ALIGNMENT__ 1
// PPC64-LINUX:#define __POINTER_WIDTH__ 64
// PPC64-LINUX:#define __POWERPC__ 1
// PPC64-LINUX:#define __PPC64__ 1
// PPC64-LINUX:#define __PPC__ 1
// PPC64-LINUX:#define __PTRDIFF_TYPE__ long int
// PPC64-LINUX:#define __PTRDIFF_WIDTH__ 64
// PPC64-LINUX:#define __REGISTER_PREFIX__
// PPC64-LINUX:#define __SCHAR_MAX__ 127
// PPC64-LINUX:#define __SHRT_MAX__ 32767
// PPC64-LINUX:#define __SIG_ATOMIC_MAX__ 2147483647
// PPC64-LINUX:#define __SIG_ATOMIC_WIDTH__ 32
// PPC64-LINUX:#define __SIZEOF_DOUBLE__ 8
// PPC64-LINUX:#define __SIZEOF_FLOAT__ 4
// PPC64-LINUX:#define __SIZEOF_INT__ 4
// PPC64-LINUX:#define __SIZEOF_LONG_DOUBLE__ 16
// PPC64-LINUX:#define __SIZEOF_LONG_LONG__ 8
// PPC64-LINUX:#define __SIZEOF_LONG__ 8
// PPC64-LINUX:#define __SIZEOF_POINTER__ 8
// PPC64-LINUX:#define __SIZEOF_PTRDIFF_T__ 8
// PPC64-LINUX:#define __SIZEOF_SHORT__ 2
// PPC64-LINUX:#define __SIZEOF_SIZE_T__ 8
// PPC64-LINUX:#define __SIZEOF_WCHAR_T__ 4
// PPC64-LINUX:#define __SIZEOF_WINT_T__ 4
// PPC64-LINUX:#define __SIZE_MAX__ 18446744073709551615UL
// PPC64-LINUX:#define __SIZE_TYPE__ long unsigned int
// PPC64-LINUX:#define __SIZE_WIDTH__ 64
// PPC64-LINUX:#define __UINT16_C_SUFFIX__
// PPC64-LINUX:#define __UINT16_MAX__ 65535
// PPC64-LINUX:#define __UINT16_TYPE__ unsigned short
// PPC64-LINUX:#define __UINT32_C_SUFFIX__ U
// PPC64-LINUX:#define __UINT32_MAX__ 4294967295U
// PPC64-LINUX:#define __UINT32_TYPE__ unsigned int
// PPC64-LINUX:#define __UINT64_C_SUFFIX__ UL
// PPC64-LINUX:#define __UINT64_MAX__ 18446744073709551615UL
// PPC64-LINUX:#define __UINT64_TYPE__ long unsigned int
// PPC64-LINUX:#define __UINT8_C_SUFFIX__
// PPC64-LINUX:#define __UINT8_MAX__ 255
// PPC64-LINUX:#define __UINT8_TYPE__ unsigned char
// PPC64-LINUX:#define __UINTMAX_C_SUFFIX__ UL
// PPC64-LINUX:#define __UINTMAX_MAX__ 18446744073709551615UL
// PPC64-LINUX:#define __UINTMAX_TYPE__ long unsigned int
// PPC64-LINUX:#define __UINTMAX_WIDTH__ 64
// PPC64-LINUX:#define __UINTPTR_MAX__ 18446744073709551615UL
// PPC64-LINUX:#define __UINTPTR_TYPE__ long unsigned int
// PPC64-LINUX:#define __UINTPTR_WIDTH__ 64
// PPC64-LINUX:#define __UINT_FAST16_MAX__ 65535
// PPC64-LINUX:#define __UINT_FAST16_TYPE__ unsigned short
// PPC64-LINUX:#define __UINT_FAST32_MAX__ 4294967295U
// PPC64-LINUX:#define __UINT_FAST32_TYPE__ unsigned int
// PPC64-LINUX:#define __UINT_FAST64_MAX__ 18446744073709551615UL
// PPC64-LINUX:#define __UINT_FAST64_TYPE__ long unsigned int
// PPC64-LINUX:#define __UINT_FAST8_MAX__ 255
// PPC64-LINUX:#define __UINT_FAST8_TYPE__ unsigned char
// PPC64-LINUX:#define __UINT_LEAST16_MAX__ 65535
// PPC64-LINUX:#define __UINT_LEAST16_TYPE__ unsigned short
// PPC64-LINUX:#define __UINT_LEAST32_MAX__ 4294967295U
// PPC64-LINUX:#define __UINT_LEAST32_TYPE__ unsigned int
// PPC64-LINUX:#define __UINT_LEAST64_MAX__ 18446744073709551615UL
// PPC64-LINUX:#define __UINT_LEAST64_TYPE__ long unsigned int
// PPC64-LINUX:#define __UINT_LEAST8_MAX__ 255
// PPC64-LINUX:#define __UINT_LEAST8_TYPE__ unsigned char
// PPC64-LINUX:#define __USER_LABEL_PREFIX__
// PPC64-LINUX:#define __WCHAR_MAX__ 2147483647
// PPC64-LINUX:#define __WCHAR_TYPE__ int
// PPC64-LINUX:#define __WCHAR_WIDTH__ 32
// PPC64-LINUX:#define __WINT_TYPE__ unsigned int
// PPC64-LINUX:#define __WINT_UNSIGNED__ 1
// PPC64-LINUX:#define __WINT_WIDTH__ 32
// PPC64-LINUX:#define __powerpc64__ 1
// PPC64-LINUX:#define __powerpc__ 1
// PPC64-LINUX:#define __ppc__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=powerpc64-unknown-linux-gnu < /dev/null | FileCheck -match-full-lines -check-prefix PPC64-ELFv1 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=powerpc64-unknown-linux-gnu -target-abi elfv1 < /dev/null | FileCheck -match-full-lines -check-prefix PPC64-ELFv1 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=powerpc64-unknown-linux-gnu -target-abi elfv2 < /dev/null | FileCheck -match-full-lines -check-prefix PPC64-ELFv2 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=powerpc64le-unknown-linux-gnu < /dev/null | FileCheck -match-full-lines -check-prefix PPC64-ELFv2 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=powerpc64le-unknown-linux-gnu -target-abi elfv1 < /dev/null | FileCheck -match-full-lines -check-prefix PPC64-ELFv1 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=powerpc64le-unknown-linux-gnu -target-abi elfv2 < /dev/null | FileCheck -match-full-lines -check-prefix PPC64-ELFv2 %s

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-unknown-freebsd11 -target-abi elfv1 -xc /dev/null | FileCheck --check-prefix=PPC64-ELFv1 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-unknown-freebsd12 -target-abi elfv1 -xc /dev/null | FileCheck --check-prefix=PPC64-ELFv1 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-unknown-freebsd13 -target-abi elfv2 -xc /dev/null | FileCheck --check-prefix=PPC64-ELFv2 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64le-unknown-freebsd13 -target-abi elfv2 -xc /dev/null | FileCheck --check-prefix=PPC64-ELFv2 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-unknown-openbsd -target-abi elfv2 -xc /dev/null | FileCheck --check-prefix=PPC64-ELFv2 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-linux-musl -target-abi elfv2 -xc /dev/null | FileCheck --check-prefix=PPC64-ELFv2 %s

// PPC64-ELFv1:#define _CALL_ELF 1
// PPC64-ELFv2:#define _CALL_ELF 2
//
// Most of this is encompassed in other places.
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64le-unknown-linux-gnu -target-abi elfv2 < /dev/null | FileCheck -match-full-lines -check-prefix PPC64LE-LINUX %s
//
// PPC64LE-LINUX:#define _CALL_LINUX 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-unknown-freebsd < /dev/null | FileCheck -match-full-lines -check-prefix PPC64-FREEBSD %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64le-unknown-freebsd < /dev/null | FileCheck -match-full-lines -check-prefix PPC64-FREEBSD %s
// PPC64-FREEBSD-NOT: #define __LONG_DOUBLE_128__ 1
