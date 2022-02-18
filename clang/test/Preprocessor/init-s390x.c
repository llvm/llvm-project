// RUN: %clang_cc1 -E -dM -ffreestanding -triple=s390x-none-none -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix S390X %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=s390x-none-none -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix S390X -check-prefix S390X-CXX %s

// S390X:#define __BIGGEST_ALIGNMENT__ 8
// S390X:#define __CHAR16_TYPE__ unsigned short
// S390X:#define __CHAR32_TYPE__ unsigned int
// S390X:#define __CHAR_BIT__ 8
// S390X:#define __CHAR_UNSIGNED__ 1
// S390X:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// S390X:#define __DBL_DIG__ 15
// S390X:#define __DBL_EPSILON__ 2.2204460492503131e-16
// S390X:#define __DBL_HAS_DENORM__ 1
// S390X:#define __DBL_HAS_INFINITY__ 1
// S390X:#define __DBL_HAS_QUIET_NAN__ 1
// S390X:#define __DBL_MANT_DIG__ 53
// S390X:#define __DBL_MAX_10_EXP__ 308
// S390X:#define __DBL_MAX_EXP__ 1024
// S390X:#define __DBL_MAX__ 1.7976931348623157e+308
// S390X:#define __DBL_MIN_10_EXP__ (-307)
// S390X:#define __DBL_MIN_EXP__ (-1021)
// S390X:#define __DBL_MIN__ 2.2250738585072014e-308
// S390X:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// S390X:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// S390X:#define __FLT_DIG__ 6
// S390X:#define __FLT_EPSILON__ 1.19209290e-7F
// S390X:#define __FLT_EVAL_METHOD__ 0
// S390X:#define __FLT_HAS_DENORM__ 1
// S390X:#define __FLT_HAS_INFINITY__ 1
// S390X:#define __FLT_HAS_QUIET_NAN__ 1
// S390X:#define __FLT_MANT_DIG__ 24
// S390X:#define __FLT_MAX_10_EXP__ 38
// S390X:#define __FLT_MAX_EXP__ 128
// S390X:#define __FLT_MAX__ 3.40282347e+38F
// S390X:#define __FLT_MIN_10_EXP__ (-37)
// S390X:#define __FLT_MIN_EXP__ (-125)
// S390X:#define __FLT_MIN__ 1.17549435e-38F
// S390X:#define __FLT_RADIX__ 2
// S390X:#define __INT16_C_SUFFIX__
// S390X:#define __INT16_FMTd__ "hd"
// S390X:#define __INT16_FMTi__ "hi"
// S390X:#define __INT16_MAX__ 32767
// S390X:#define __INT16_TYPE__ short
// S390X:#define __INT32_C_SUFFIX__
// S390X:#define __INT32_FMTd__ "d"
// S390X:#define __INT32_FMTi__ "i"
// S390X:#define __INT32_MAX__ 2147483647
// S390X:#define __INT32_TYPE__ int
// S390X:#define __INT64_C_SUFFIX__ L
// S390X:#define __INT64_FMTd__ "ld"
// S390X:#define __INT64_FMTi__ "li"
// S390X:#define __INT64_MAX__ 9223372036854775807L
// S390X:#define __INT64_TYPE__ long int
// S390X:#define __INT8_C_SUFFIX__
// S390X:#define __INT8_FMTd__ "hhd"
// S390X:#define __INT8_FMTi__ "hhi"
// S390X:#define __INT8_MAX__ 127
// S390X:#define __INT8_TYPE__ signed char
// S390X:#define __INTMAX_C_SUFFIX__ L
// S390X:#define __INTMAX_FMTd__ "ld"
// S390X:#define __INTMAX_FMTi__ "li"
// S390X:#define __INTMAX_MAX__ 9223372036854775807L
// S390X:#define __INTMAX_TYPE__ long int
// S390X:#define __INTMAX_WIDTH__ 64
// S390X:#define __INTPTR_FMTd__ "ld"
// S390X:#define __INTPTR_FMTi__ "li"
// S390X:#define __INTPTR_MAX__ 9223372036854775807L
// S390X:#define __INTPTR_TYPE__ long int
// S390X:#define __INTPTR_WIDTH__ 64
// S390X:#define __INT_FAST16_FMTd__ "hd"
// S390X:#define __INT_FAST16_FMTi__ "hi"
// S390X:#define __INT_FAST16_MAX__ 32767
// S390X:#define __INT_FAST16_TYPE__ short
// S390X:#define __INT_FAST32_FMTd__ "d"
// S390X:#define __INT_FAST32_FMTi__ "i"
// S390X:#define __INT_FAST32_MAX__ 2147483647
// S390X:#define __INT_FAST32_TYPE__ int
// S390X:#define __INT_FAST64_FMTd__ "ld"
// S390X:#define __INT_FAST64_FMTi__ "li"
// S390X:#define __INT_FAST64_MAX__ 9223372036854775807L
// S390X:#define __INT_FAST64_TYPE__ long int
// S390X:#define __INT_FAST8_FMTd__ "hhd"
// S390X:#define __INT_FAST8_FMTi__ "hhi"
// S390X:#define __INT_FAST8_MAX__ 127
// S390X:#define __INT_FAST8_TYPE__ signed char
// S390X:#define __INT_LEAST16_FMTd__ "hd"
// S390X:#define __INT_LEAST16_FMTi__ "hi"
// S390X:#define __INT_LEAST16_MAX__ 32767
// S390X:#define __INT_LEAST16_TYPE__ short
// S390X:#define __INT_LEAST32_FMTd__ "d"
// S390X:#define __INT_LEAST32_FMTi__ "i"
// S390X:#define __INT_LEAST32_MAX__ 2147483647
// S390X:#define __INT_LEAST32_TYPE__ int
// S390X:#define __INT_LEAST64_FMTd__ "ld"
// S390X:#define __INT_LEAST64_FMTi__ "li"
// S390X:#define __INT_LEAST64_MAX__ 9223372036854775807L
// S390X:#define __INT_LEAST64_TYPE__ long int
// S390X:#define __INT_LEAST8_FMTd__ "hhd"
// S390X:#define __INT_LEAST8_FMTi__ "hhi"
// S390X:#define __INT_LEAST8_MAX__ 127
// S390X:#define __INT_LEAST8_TYPE__ signed char
// S390X:#define __INT_MAX__ 2147483647
// S390X:#define __LDBL_DENORM_MIN__ 6.47517511943802511092443895822764655e-4966L
// S390X:#define __LDBL_DIG__ 33
// S390X:#define __LDBL_EPSILON__ 1.92592994438723585305597794258492732e-34L
// S390X:#define __LDBL_HAS_DENORM__ 1
// S390X:#define __LDBL_HAS_INFINITY__ 1
// S390X:#define __LDBL_HAS_QUIET_NAN__ 1
// S390X:#define __LDBL_MANT_DIG__ 113
// S390X:#define __LDBL_MAX_10_EXP__ 4932
// S390X:#define __LDBL_MAX_EXP__ 16384
// S390X:#define __LDBL_MAX__ 1.18973149535723176508575932662800702e+4932L
// S390X:#define __LDBL_MIN_10_EXP__ (-4931)
// S390X:#define __LDBL_MIN_EXP__ (-16381)
// S390X:#define __LDBL_MIN__ 3.36210314311209350626267781732175260e-4932L
// S390X:#define __LONG_LONG_MAX__ 9223372036854775807LL
// S390X:#define __LONG_MAX__ 9223372036854775807L
// S390X:#define __NO_INLINE__ 1
// S390X:#define __POINTER_WIDTH__ 64
// S390X:#define __PTRDIFF_TYPE__ long int
// S390X:#define __PTRDIFF_WIDTH__ 64
// S390X:#define __SCHAR_MAX__ 127
// S390X:#define __SHRT_MAX__ 32767
// S390X:#define __SIG_ATOMIC_MAX__ 2147483647
// S390X:#define __SIG_ATOMIC_WIDTH__ 32
// S390X:#define __SIZEOF_DOUBLE__ 8
// S390X:#define __SIZEOF_FLOAT__ 4
// S390X:#define __SIZEOF_INT__ 4
// S390X:#define __SIZEOF_LONG_DOUBLE__ 16
// S390X:#define __SIZEOF_LONG_LONG__ 8
// S390X:#define __SIZEOF_LONG__ 8
// S390X:#define __SIZEOF_POINTER__ 8
// S390X:#define __SIZEOF_PTRDIFF_T__ 8
// S390X:#define __SIZEOF_SHORT__ 2
// S390X:#define __SIZEOF_SIZE_T__ 8
// S390X:#define __SIZEOF_WCHAR_T__ 4
// S390X:#define __SIZEOF_WINT_T__ 4
// S390X:#define __SIZE_TYPE__ long unsigned int
// S390X:#define __SIZE_WIDTH__ 64
// S390X-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 8UL
// S390X:#define __UINT16_C_SUFFIX__
// S390X:#define __UINT16_MAX__ 65535
// S390X:#define __UINT16_TYPE__ unsigned short
// S390X:#define __UINT32_C_SUFFIX__ U
// S390X:#define __UINT32_MAX__ 4294967295U
// S390X:#define __UINT32_TYPE__ unsigned int
// S390X:#define __UINT64_C_SUFFIX__ UL
// S390X:#define __UINT64_MAX__ 18446744073709551615UL
// S390X:#define __UINT64_TYPE__ long unsigned int
// S390X:#define __UINT8_C_SUFFIX__
// S390X:#define __UINT8_MAX__ 255
// S390X:#define __UINT8_TYPE__ unsigned char
// S390X:#define __UINTMAX_C_SUFFIX__ UL
// S390X:#define __UINTMAX_MAX__ 18446744073709551615UL
// S390X:#define __UINTMAX_TYPE__ long unsigned int
// S390X:#define __UINTMAX_WIDTH__ 64
// S390X:#define __UINTPTR_MAX__ 18446744073709551615UL
// S390X:#define __UINTPTR_TYPE__ long unsigned int
// S390X:#define __UINTPTR_WIDTH__ 64
// S390X:#define __UINT_FAST16_MAX__ 65535
// S390X:#define __UINT_FAST16_TYPE__ unsigned short
// S390X:#define __UINT_FAST32_MAX__ 4294967295U
// S390X:#define __UINT_FAST32_TYPE__ unsigned int
// S390X:#define __UINT_FAST64_MAX__ 18446744073709551615UL
// S390X:#define __UINT_FAST64_TYPE__ long unsigned int
// S390X:#define __UINT_FAST8_MAX__ 255
// S390X:#define __UINT_FAST8_TYPE__ unsigned char
// S390X:#define __UINT_LEAST16_MAX__ 65535
// S390X:#define __UINT_LEAST16_TYPE__ unsigned short
// S390X:#define __UINT_LEAST32_MAX__ 4294967295U
// S390X:#define __UINT_LEAST32_TYPE__ unsigned int
// S390X:#define __UINT_LEAST64_MAX__ 18446744073709551615UL
// S390X:#define __UINT_LEAST64_TYPE__ long unsigned int
// S390X:#define __UINT_LEAST8_MAX__ 255
// S390X:#define __UINT_LEAST8_TYPE__ unsigned char
// S390X:#define __USER_LABEL_PREFIX__
// S390X:#define __WCHAR_MAX__ 2147483647
// S390X:#define __WCHAR_TYPE__ int
// S390X:#define __WCHAR_WIDTH__ 32
// S390X:#define __WINT_TYPE__ int
// S390X:#define __WINT_WIDTH__ 32
// S390X:#define __s390__ 1
// S390X:#define __s390x__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=s390x-none-zos -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix S390X-ZOS %s
// RUN: %clang_cc1 -x c++ -std=gnu++14 -E -dM -ffreestanding -triple=s390x-none-zos -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix S390X-ZOS -check-prefix S390X-ZOS-GNUXX %s

// S390X-ZOS-GNUXX: #define _EXT 1
// S390X-ZOS:       #define _LONG_LONG 1
// S390X-ZOS-GNUXX: #define _MI_BUILTIN 1
// S390X-ZOS:       #define _OPEN_DEFAULT 1
// S390X-ZOS:       #define _UNIX03_WITHDRAWN 1
// S390X-ZOS-GNUXX: #define _XOPEN_SOURCE 600
// S390X-ZOS:       #define __370__ 1
// S390X-ZOS:       #define __64BIT__ 1
// S390X-ZOS:       #define __BFP__ 1
// S390X-ZOS:       #define __BOOL__ 1
// S390X-ZOS-GNUXX: #define __DLL__ 1
// S390X-ZOS:       #define __LONGNAME__ 1
// S390X-ZOS:       #define __MVS__ 1
// S390X-ZOS:       #define __THW_370__ 1
// S390X-ZOS:       #define __THW_BIG_ENDIAN__ 1
// S390X-ZOS:       #define __TOS_390__ 1
// S390X-ZOS:       #define __TOS_MVS__ 1
// S390X-ZOS:       #define __XPLINK__ 1
// S390X-ZOS-GNUXX: #define __wchar_t 1
