//
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=mips-none-none < /dev/null | FileCheck -match-full-lines -check-prefix MIPS32BE -check-prefix MIPS32BE-C %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=mips-none-none < /dev/null | FileCheck -match-full-lines -check-prefix MIPS32BE -check-prefix MIPS32BE-CXX %s
//
// MIPS32BE:#define MIPSEB 1
// MIPS32BE:#define _ABIO32 1
// MIPS32BE-NOT:#define _LP64
// MIPS32BE:#define _MIPSEB 1
// MIPS32BE:#define _MIPS_ARCH "mips32r2"
// MIPS32BE:#define _MIPS_ARCH_MIPS32R2 1
// MIPS32BE:#define _MIPS_FPSET 16
// MIPS32BE:#define _MIPS_SIM _ABIO32
// MIPS32BE:#define _MIPS_SZINT 32
// MIPS32BE:#define _MIPS_SZLONG 32
// MIPS32BE:#define _MIPS_SZPTR 32
// MIPS32BE:#define __BIGGEST_ALIGNMENT__ 8
// MIPS32BE:#define __BIG_ENDIAN__ 1
// MIPS32BE:#define __BYTE_ORDER__ __ORDER_BIG_ENDIAN__
// MIPS32BE:#define __CHAR16_TYPE__ unsigned short
// MIPS32BE:#define __CHAR32_TYPE__ unsigned int
// MIPS32BE:#define __CHAR_BIT__ 8
// MIPS32BE:#define __CONSTANT_CFSTRINGS__ 1
// MIPS32BE:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// MIPS32BE:#define __DBL_DIG__ 15
// MIPS32BE:#define __DBL_EPSILON__ 2.2204460492503131e-16
// MIPS32BE:#define __DBL_HAS_DENORM__ 1
// MIPS32BE:#define __DBL_HAS_INFINITY__ 1
// MIPS32BE:#define __DBL_HAS_QUIET_NAN__ 1
// MIPS32BE:#define __DBL_MANT_DIG__ 53
// MIPS32BE:#define __DBL_MAX_10_EXP__ 308
// MIPS32BE:#define __DBL_MAX_EXP__ 1024
// MIPS32BE:#define __DBL_MAX__ 1.7976931348623157e+308
// MIPS32BE:#define __DBL_MIN_10_EXP__ (-307)
// MIPS32BE:#define __DBL_MIN_EXP__ (-1021)
// MIPS32BE:#define __DBL_MIN__ 2.2250738585072014e-308
// MIPS32BE:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// MIPS32BE:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// MIPS32BE:#define __FLT_DIG__ 6
// MIPS32BE:#define __FLT_EPSILON__ 1.19209290e-7F
// MIPS32BE:#define __FLT_EVAL_METHOD__ 0
// MIPS32BE:#define __FLT_HAS_DENORM__ 1
// MIPS32BE:#define __FLT_HAS_INFINITY__ 1
// MIPS32BE:#define __FLT_HAS_QUIET_NAN__ 1
// MIPS32BE:#define __FLT_MANT_DIG__ 24
// MIPS32BE:#define __FLT_MAX_10_EXP__ 38
// MIPS32BE:#define __FLT_MAX_EXP__ 128
// MIPS32BE:#define __FLT_MAX__ 3.40282347e+38F
// MIPS32BE:#define __FLT_MIN_10_EXP__ (-37)
// MIPS32BE:#define __FLT_MIN_EXP__ (-125)
// MIPS32BE:#define __FLT_MIN__ 1.17549435e-38F
// MIPS32BE:#define __FLT_RADIX__ 2
// MIPS32BE:#define __INT16_C_SUFFIX__
// MIPS32BE:#define __INT16_FMTd__ "hd"
// MIPS32BE:#define __INT16_FMTi__ "hi"
// MIPS32BE:#define __INT16_MAX__ 32767
// MIPS32BE:#define __INT16_TYPE__ short
// MIPS32BE:#define __INT32_C_SUFFIX__
// MIPS32BE:#define __INT32_FMTd__ "d"
// MIPS32BE:#define __INT32_FMTi__ "i"
// MIPS32BE:#define __INT32_MAX__ 2147483647
// MIPS32BE:#define __INT32_TYPE__ int
// MIPS32BE:#define __INT64_C_SUFFIX__ LL
// MIPS32BE:#define __INT64_FMTd__ "lld"
// MIPS32BE:#define __INT64_FMTi__ "lli"
// MIPS32BE:#define __INT64_MAX__ 9223372036854775807LL
// MIPS32BE:#define __INT64_TYPE__ long long int
// MIPS32BE:#define __INT8_C_SUFFIX__
// MIPS32BE:#define __INT8_FMTd__ "hhd"
// MIPS32BE:#define __INT8_FMTi__ "hhi"
// MIPS32BE:#define __INT8_MAX__ 127
// MIPS32BE:#define __INT8_TYPE__ signed char
// MIPS32BE:#define __INTMAX_C_SUFFIX__ LL
// MIPS32BE:#define __INTMAX_FMTd__ "lld"
// MIPS32BE:#define __INTMAX_FMTi__ "lli"
// MIPS32BE:#define __INTMAX_MAX__ 9223372036854775807LL
// MIPS32BE:#define __INTMAX_TYPE__ long long int
// MIPS32BE:#define __INTMAX_WIDTH__ 64
// MIPS32BE:#define __INTPTR_FMTd__ "ld"
// MIPS32BE:#define __INTPTR_FMTi__ "li"
// MIPS32BE:#define __INTPTR_MAX__ 2147483647L
// MIPS32BE:#define __INTPTR_TYPE__ long int
// MIPS32BE:#define __INTPTR_WIDTH__ 32
// MIPS32BE:#define __INT_FAST16_FMTd__ "hd"
// MIPS32BE:#define __INT_FAST16_FMTi__ "hi"
// MIPS32BE:#define __INT_FAST16_MAX__ 32767
// MIPS32BE:#define __INT_FAST16_TYPE__ short
// MIPS32BE:#define __INT_FAST32_FMTd__ "d"
// MIPS32BE:#define __INT_FAST32_FMTi__ "i"
// MIPS32BE:#define __INT_FAST32_MAX__ 2147483647
// MIPS32BE:#define __INT_FAST32_TYPE__ int
// MIPS32BE:#define __INT_FAST64_FMTd__ "lld"
// MIPS32BE:#define __INT_FAST64_FMTi__ "lli"
// MIPS32BE:#define __INT_FAST64_MAX__ 9223372036854775807LL
// MIPS32BE:#define __INT_FAST64_TYPE__ long long int
// MIPS32BE:#define __INT_FAST8_FMTd__ "hhd"
// MIPS32BE:#define __INT_FAST8_FMTi__ "hhi"
// MIPS32BE:#define __INT_FAST8_MAX__ 127
// MIPS32BE:#define __INT_FAST8_TYPE__ signed char
// MIPS32BE:#define __INT_LEAST16_FMTd__ "hd"
// MIPS32BE:#define __INT_LEAST16_FMTi__ "hi"
// MIPS32BE:#define __INT_LEAST16_MAX__ 32767
// MIPS32BE:#define __INT_LEAST16_TYPE__ short
// MIPS32BE:#define __INT_LEAST32_FMTd__ "d"
// MIPS32BE:#define __INT_LEAST32_FMTi__ "i"
// MIPS32BE:#define __INT_LEAST32_MAX__ 2147483647
// MIPS32BE:#define __INT_LEAST32_TYPE__ int
// MIPS32BE:#define __INT_LEAST64_FMTd__ "lld"
// MIPS32BE:#define __INT_LEAST64_FMTi__ "lli"
// MIPS32BE:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// MIPS32BE:#define __INT_LEAST64_TYPE__ long long int
// MIPS32BE:#define __INT_LEAST8_FMTd__ "hhd"
// MIPS32BE:#define __INT_LEAST8_FMTi__ "hhi"
// MIPS32BE:#define __INT_LEAST8_MAX__ 127
// MIPS32BE:#define __INT_LEAST8_TYPE__ signed char
// MIPS32BE:#define __INT_MAX__ 2147483647
// MIPS32BE:#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324L
// MIPS32BE:#define __LDBL_DIG__ 15
// MIPS32BE:#define __LDBL_EPSILON__ 2.2204460492503131e-16L
// MIPS32BE:#define __LDBL_HAS_DENORM__ 1
// MIPS32BE:#define __LDBL_HAS_INFINITY__ 1
// MIPS32BE:#define __LDBL_HAS_QUIET_NAN__ 1
// MIPS32BE:#define __LDBL_MANT_DIG__ 53
// MIPS32BE:#define __LDBL_MAX_10_EXP__ 308
// MIPS32BE:#define __LDBL_MAX_EXP__ 1024
// MIPS32BE:#define __LDBL_MAX__ 1.7976931348623157e+308L
// MIPS32BE:#define __LDBL_MIN_10_EXP__ (-307)
// MIPS32BE:#define __LDBL_MIN_EXP__ (-1021)
// MIPS32BE:#define __LDBL_MIN__ 2.2250738585072014e-308L
// MIPS32BE:#define __LONG_LONG_MAX__ 9223372036854775807LL
// MIPS32BE:#define __LONG_MAX__ 2147483647L
// MIPS32BE-NOT:#define __LP64__
// MIPS32BE:#define __MIPSEB 1
// MIPS32BE:#define __MIPSEB__ 1
// MIPS32BE:#define __POINTER_WIDTH__ 32
// MIPS32BE:#define __PRAGMA_REDEFINE_EXTNAME 1
// MIPS32BE:#define __PTRDIFF_TYPE__ int
// MIPS32BE:#define __PTRDIFF_WIDTH__ 32
// MIPS32BE:#define __REGISTER_PREFIX__
// MIPS32BE:#define __SCHAR_MAX__ 127
// MIPS32BE:#define __SHRT_MAX__ 32767
// MIPS32BE:#define __SIG_ATOMIC_MAX__ 2147483647
// MIPS32BE:#define __SIG_ATOMIC_WIDTH__ 32
// MIPS32BE:#define __SIZEOF_DOUBLE__ 8
// MIPS32BE:#define __SIZEOF_FLOAT__ 4
// MIPS32BE:#define __SIZEOF_INT__ 4
// MIPS32BE:#define __SIZEOF_LONG_DOUBLE__ 8
// MIPS32BE:#define __SIZEOF_LONG_LONG__ 8
// MIPS32BE:#define __SIZEOF_LONG__ 4
// MIPS32BE:#define __SIZEOF_POINTER__ 4
// MIPS32BE:#define __SIZEOF_PTRDIFF_T__ 4
// MIPS32BE:#define __SIZEOF_SHORT__ 2
// MIPS32BE:#define __SIZEOF_SIZE_T__ 4
// MIPS32BE:#define __SIZEOF_WCHAR_T__ 4
// MIPS32BE:#define __SIZEOF_WINT_T__ 4
// MIPS32BE:#define __SIZE_MAX__ 4294967295U
// MIPS32BE:#define __SIZE_TYPE__ unsigned int
// MIPS32BE:#define __SIZE_WIDTH__ 32
// MIPS32BE-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 8U
// MIPS32BE:#define __STDC_HOSTED__ 0
// MIPS32BE-C:#define __STDC_VERSION__ 201710L
// MIPS32BE:#define __STDC__ 1
// MIPS32BE:#define __UINT16_C_SUFFIX__
// MIPS32BE:#define __UINT16_MAX__ 65535
// MIPS32BE:#define __UINT16_TYPE__ unsigned short
// MIPS32BE:#define __UINT32_C_SUFFIX__ U
// MIPS32BE:#define __UINT32_MAX__ 4294967295U
// MIPS32BE:#define __UINT32_TYPE__ unsigned int
// MIPS32BE:#define __UINT64_C_SUFFIX__ ULL
// MIPS32BE:#define __UINT64_MAX__ 18446744073709551615ULL
// MIPS32BE:#define __UINT64_TYPE__ long long unsigned int
// MIPS32BE:#define __UINT8_C_SUFFIX__
// MIPS32BE:#define __UINT8_MAX__ 255
// MIPS32BE:#define __UINT8_TYPE__ unsigned char
// MIPS32BE:#define __UINTMAX_C_SUFFIX__ ULL
// MIPS32BE:#define __UINTMAX_MAX__ 18446744073709551615ULL
// MIPS32BE:#define __UINTMAX_TYPE__ long long unsigned int
// MIPS32BE:#define __UINTMAX_WIDTH__ 64
// MIPS32BE:#define __UINTPTR_MAX__ 4294967295UL
// MIPS32BE:#define __UINTPTR_TYPE__ long unsigned int
// MIPS32BE:#define __UINTPTR_WIDTH__ 32
// MIPS32BE:#define __UINT_FAST16_MAX__ 65535
// MIPS32BE:#define __UINT_FAST16_TYPE__ unsigned short
// MIPS32BE:#define __UINT_FAST32_MAX__ 4294967295U
// MIPS32BE:#define __UINT_FAST32_TYPE__ unsigned int
// MIPS32BE:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// MIPS32BE:#define __UINT_FAST64_TYPE__ long long unsigned int
// MIPS32BE:#define __UINT_FAST8_MAX__ 255
// MIPS32BE:#define __UINT_FAST8_TYPE__ unsigned char
// MIPS32BE:#define __UINT_LEAST16_MAX__ 65535
// MIPS32BE:#define __UINT_LEAST16_TYPE__ unsigned short
// MIPS32BE:#define __UINT_LEAST32_MAX__ 4294967295U
// MIPS32BE:#define __UINT_LEAST32_TYPE__ unsigned int
// MIPS32BE:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// MIPS32BE:#define __UINT_LEAST64_TYPE__ long long unsigned int
// MIPS32BE:#define __UINT_LEAST8_MAX__ 255
// MIPS32BE:#define __UINT_LEAST8_TYPE__ unsigned char
// MIPS32BE:#define __USER_LABEL_PREFIX__
// MIPS32BE:#define __WCHAR_MAX__ 2147483647
// MIPS32BE:#define __WCHAR_TYPE__ int
// MIPS32BE:#define __WCHAR_WIDTH__ 32
// MIPS32BE:#define __WINT_TYPE__ int
// MIPS32BE:#define __WINT_WIDTH__ 32
// MIPS32BE:#define __clang__ 1
// MIPS32BE:#define __llvm__ 1
// MIPS32BE:#define __mips 32
// MIPS32BE:#define __mips__ 1
// MIPS32BE:#define __mips_abicalls 1
// MIPS32BE:#define __mips_fpr 0
// MIPS32BE:#define __mips_hard_float 1
// MIPS32BE:#define __mips_o32 1
// MIPS32BE:#define _mips 1
// MIPS32BE:#define mips 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mipsel-none-none < /dev/null | FileCheck -match-full-lines -check-prefix MIPS32EL %s
//
// MIPS32EL:#define MIPSEL 1
// MIPS32EL:#define _ABIO32 1
// MIPS32EL-NOT:#define _LP64
// MIPS32EL:#define _MIPSEL 1
// MIPS32EL:#define _MIPS_ARCH "mips32r2"
// MIPS32EL:#define _MIPS_ARCH_MIPS32R2 1
// MIPS32EL:#define _MIPS_FPSET 16
// MIPS32EL:#define _MIPS_SIM _ABIO32
// MIPS32EL:#define _MIPS_SZINT 32
// MIPS32EL:#define _MIPS_SZLONG 32
// MIPS32EL:#define _MIPS_SZPTR 32
// MIPS32EL:#define __BIGGEST_ALIGNMENT__ 8
// MIPS32EL:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// MIPS32EL:#define __CHAR16_TYPE__ unsigned short
// MIPS32EL:#define __CHAR32_TYPE__ unsigned int
// MIPS32EL:#define __CHAR_BIT__ 8
// MIPS32EL:#define __CONSTANT_CFSTRINGS__ 1
// MIPS32EL:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// MIPS32EL:#define __DBL_DIG__ 15
// MIPS32EL:#define __DBL_EPSILON__ 2.2204460492503131e-16
// MIPS32EL:#define __DBL_HAS_DENORM__ 1
// MIPS32EL:#define __DBL_HAS_INFINITY__ 1
// MIPS32EL:#define __DBL_HAS_QUIET_NAN__ 1
// MIPS32EL:#define __DBL_MANT_DIG__ 53
// MIPS32EL:#define __DBL_MAX_10_EXP__ 308
// MIPS32EL:#define __DBL_MAX_EXP__ 1024
// MIPS32EL:#define __DBL_MAX__ 1.7976931348623157e+308
// MIPS32EL:#define __DBL_MIN_10_EXP__ (-307)
// MIPS32EL:#define __DBL_MIN_EXP__ (-1021)
// MIPS32EL:#define __DBL_MIN__ 2.2250738585072014e-308
// MIPS32EL:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// MIPS32EL:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// MIPS32EL:#define __FLT_DIG__ 6
// MIPS32EL:#define __FLT_EPSILON__ 1.19209290e-7F
// MIPS32EL:#define __FLT_EVAL_METHOD__ 0
// MIPS32EL:#define __FLT_HAS_DENORM__ 1
// MIPS32EL:#define __FLT_HAS_INFINITY__ 1
// MIPS32EL:#define __FLT_HAS_QUIET_NAN__ 1
// MIPS32EL:#define __FLT_MANT_DIG__ 24
// MIPS32EL:#define __FLT_MAX_10_EXP__ 38
// MIPS32EL:#define __FLT_MAX_EXP__ 128
// MIPS32EL:#define __FLT_MAX__ 3.40282347e+38F
// MIPS32EL:#define __FLT_MIN_10_EXP__ (-37)
// MIPS32EL:#define __FLT_MIN_EXP__ (-125)
// MIPS32EL:#define __FLT_MIN__ 1.17549435e-38F
// MIPS32EL:#define __FLT_RADIX__ 2
// MIPS32EL:#define __INT16_C_SUFFIX__
// MIPS32EL:#define __INT16_FMTd__ "hd"
// MIPS32EL:#define __INT16_FMTi__ "hi"
// MIPS32EL:#define __INT16_MAX__ 32767
// MIPS32EL:#define __INT16_TYPE__ short
// MIPS32EL:#define __INT32_C_SUFFIX__
// MIPS32EL:#define __INT32_FMTd__ "d"
// MIPS32EL:#define __INT32_FMTi__ "i"
// MIPS32EL:#define __INT32_MAX__ 2147483647
// MIPS32EL:#define __INT32_TYPE__ int
// MIPS32EL:#define __INT64_C_SUFFIX__ LL
// MIPS32EL:#define __INT64_FMTd__ "lld"
// MIPS32EL:#define __INT64_FMTi__ "lli"
// MIPS32EL:#define __INT64_MAX__ 9223372036854775807LL
// MIPS32EL:#define __INT64_TYPE__ long long int
// MIPS32EL:#define __INT8_C_SUFFIX__
// MIPS32EL:#define __INT8_FMTd__ "hhd"
// MIPS32EL:#define __INT8_FMTi__ "hhi"
// MIPS32EL:#define __INT8_MAX__ 127
// MIPS32EL:#define __INT8_TYPE__ signed char
// MIPS32EL:#define __INTMAX_C_SUFFIX__ LL
// MIPS32EL:#define __INTMAX_FMTd__ "lld"
// MIPS32EL:#define __INTMAX_FMTi__ "lli"
// MIPS32EL:#define __INTMAX_MAX__ 9223372036854775807LL
// MIPS32EL:#define __INTMAX_TYPE__ long long int
// MIPS32EL:#define __INTMAX_WIDTH__ 64
// MIPS32EL:#define __INTPTR_FMTd__ "ld"
// MIPS32EL:#define __INTPTR_FMTi__ "li"
// MIPS32EL:#define __INTPTR_MAX__ 2147483647L
// MIPS32EL:#define __INTPTR_TYPE__ long int
// MIPS32EL:#define __INTPTR_WIDTH__ 32
// MIPS32EL:#define __INT_FAST16_FMTd__ "hd"
// MIPS32EL:#define __INT_FAST16_FMTi__ "hi"
// MIPS32EL:#define __INT_FAST16_MAX__ 32767
// MIPS32EL:#define __INT_FAST16_TYPE__ short
// MIPS32EL:#define __INT_FAST32_FMTd__ "d"
// MIPS32EL:#define __INT_FAST32_FMTi__ "i"
// MIPS32EL:#define __INT_FAST32_MAX__ 2147483647
// MIPS32EL:#define __INT_FAST32_TYPE__ int
// MIPS32EL:#define __INT_FAST64_FMTd__ "lld"
// MIPS32EL:#define __INT_FAST64_FMTi__ "lli"
// MIPS32EL:#define __INT_FAST64_MAX__ 9223372036854775807LL
// MIPS32EL:#define __INT_FAST64_TYPE__ long long int
// MIPS32EL:#define __INT_FAST8_FMTd__ "hhd"
// MIPS32EL:#define __INT_FAST8_FMTi__ "hhi"
// MIPS32EL:#define __INT_FAST8_MAX__ 127
// MIPS32EL:#define __INT_FAST8_TYPE__ signed char
// MIPS32EL:#define __INT_LEAST16_FMTd__ "hd"
// MIPS32EL:#define __INT_LEAST16_FMTi__ "hi"
// MIPS32EL:#define __INT_LEAST16_MAX__ 32767
// MIPS32EL:#define __INT_LEAST16_TYPE__ short
// MIPS32EL:#define __INT_LEAST32_FMTd__ "d"
// MIPS32EL:#define __INT_LEAST32_FMTi__ "i"
// MIPS32EL:#define __INT_LEAST32_MAX__ 2147483647
// MIPS32EL:#define __INT_LEAST32_TYPE__ int
// MIPS32EL:#define __INT_LEAST64_FMTd__ "lld"
// MIPS32EL:#define __INT_LEAST64_FMTi__ "lli"
// MIPS32EL:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// MIPS32EL:#define __INT_LEAST64_TYPE__ long long int
// MIPS32EL:#define __INT_LEAST8_FMTd__ "hhd"
// MIPS32EL:#define __INT_LEAST8_FMTi__ "hhi"
// MIPS32EL:#define __INT_LEAST8_MAX__ 127
// MIPS32EL:#define __INT_LEAST8_TYPE__ signed char
// MIPS32EL:#define __INT_MAX__ 2147483647
// MIPS32EL:#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324L
// MIPS32EL:#define __LDBL_DIG__ 15
// MIPS32EL:#define __LDBL_EPSILON__ 2.2204460492503131e-16L
// MIPS32EL:#define __LDBL_HAS_DENORM__ 1
// MIPS32EL:#define __LDBL_HAS_INFINITY__ 1
// MIPS32EL:#define __LDBL_HAS_QUIET_NAN__ 1
// MIPS32EL:#define __LDBL_MANT_DIG__ 53
// MIPS32EL:#define __LDBL_MAX_10_EXP__ 308
// MIPS32EL:#define __LDBL_MAX_EXP__ 1024
// MIPS32EL:#define __LDBL_MAX__ 1.7976931348623157e+308L
// MIPS32EL:#define __LDBL_MIN_10_EXP__ (-307)
// MIPS32EL:#define __LDBL_MIN_EXP__ (-1021)
// MIPS32EL:#define __LDBL_MIN__ 2.2250738585072014e-308L
// MIPS32EL:#define __LITTLE_ENDIAN__ 1
// MIPS32EL:#define __LONG_LONG_MAX__ 9223372036854775807LL
// MIPS32EL:#define __LONG_MAX__ 2147483647L
// MIPS32EL-NOT:#define __LP64__
// MIPS32EL:#define __MIPSEL 1
// MIPS32EL:#define __MIPSEL__ 1
// MIPS32EL:#define __POINTER_WIDTH__ 32
// MIPS32EL:#define __PRAGMA_REDEFINE_EXTNAME 1
// MIPS32EL:#define __PTRDIFF_TYPE__ int
// MIPS32EL:#define __PTRDIFF_WIDTH__ 32
// MIPS32EL:#define __REGISTER_PREFIX__
// MIPS32EL:#define __SCHAR_MAX__ 127
// MIPS32EL:#define __SHRT_MAX__ 32767
// MIPS32EL:#define __SIG_ATOMIC_MAX__ 2147483647
// MIPS32EL:#define __SIG_ATOMIC_WIDTH__ 32
// MIPS32EL:#define __SIZEOF_DOUBLE__ 8
// MIPS32EL:#define __SIZEOF_FLOAT__ 4
// MIPS32EL:#define __SIZEOF_INT__ 4
// MIPS32EL:#define __SIZEOF_LONG_DOUBLE__ 8
// MIPS32EL:#define __SIZEOF_LONG_LONG__ 8
// MIPS32EL:#define __SIZEOF_LONG__ 4
// MIPS32EL:#define __SIZEOF_POINTER__ 4
// MIPS32EL:#define __SIZEOF_PTRDIFF_T__ 4
// MIPS32EL:#define __SIZEOF_SHORT__ 2
// MIPS32EL:#define __SIZEOF_SIZE_T__ 4
// MIPS32EL:#define __SIZEOF_WCHAR_T__ 4
// MIPS32EL:#define __SIZEOF_WINT_T__ 4
// MIPS32EL:#define __SIZE_MAX__ 4294967295U
// MIPS32EL:#define __SIZE_TYPE__ unsigned int
// MIPS32EL:#define __SIZE_WIDTH__ 32
// MIPS32EL:#define __UINT16_C_SUFFIX__
// MIPS32EL:#define __UINT16_MAX__ 65535
// MIPS32EL:#define __UINT16_TYPE__ unsigned short
// MIPS32EL:#define __UINT32_C_SUFFIX__ U
// MIPS32EL:#define __UINT32_MAX__ 4294967295U
// MIPS32EL:#define __UINT32_TYPE__ unsigned int
// MIPS32EL:#define __UINT64_C_SUFFIX__ ULL
// MIPS32EL:#define __UINT64_MAX__ 18446744073709551615ULL
// MIPS32EL:#define __UINT64_TYPE__ long long unsigned int
// MIPS32EL:#define __UINT8_C_SUFFIX__
// MIPS32EL:#define __UINT8_MAX__ 255
// MIPS32EL:#define __UINT8_TYPE__ unsigned char
// MIPS32EL:#define __UINTMAX_C_SUFFIX__ ULL
// MIPS32EL:#define __UINTMAX_MAX__ 18446744073709551615ULL
// MIPS32EL:#define __UINTMAX_TYPE__ long long unsigned int
// MIPS32EL:#define __UINTMAX_WIDTH__ 64
// MIPS32EL:#define __UINTPTR_MAX__ 4294967295UL
// MIPS32EL:#define __UINTPTR_TYPE__ long unsigned int
// MIPS32EL:#define __UINTPTR_WIDTH__ 32
// MIPS32EL:#define __UINT_FAST16_MAX__ 65535
// MIPS32EL:#define __UINT_FAST16_TYPE__ unsigned short
// MIPS32EL:#define __UINT_FAST32_MAX__ 4294967295U
// MIPS32EL:#define __UINT_FAST32_TYPE__ unsigned int
// MIPS32EL:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// MIPS32EL:#define __UINT_FAST64_TYPE__ long long unsigned int
// MIPS32EL:#define __UINT_FAST8_MAX__ 255
// MIPS32EL:#define __UINT_FAST8_TYPE__ unsigned char
// MIPS32EL:#define __UINT_LEAST16_MAX__ 65535
// MIPS32EL:#define __UINT_LEAST16_TYPE__ unsigned short
// MIPS32EL:#define __UINT_LEAST32_MAX__ 4294967295U
// MIPS32EL:#define __UINT_LEAST32_TYPE__ unsigned int
// MIPS32EL:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// MIPS32EL:#define __UINT_LEAST64_TYPE__ long long unsigned int
// MIPS32EL:#define __UINT_LEAST8_MAX__ 255
// MIPS32EL:#define __UINT_LEAST8_TYPE__ unsigned char
// MIPS32EL:#define __USER_LABEL_PREFIX__
// MIPS32EL:#define __WCHAR_MAX__ 2147483647
// MIPS32EL:#define __WCHAR_TYPE__ int
// MIPS32EL:#define __WCHAR_WIDTH__ 32
// MIPS32EL:#define __WINT_TYPE__ int
// MIPS32EL:#define __WINT_WIDTH__ 32
// MIPS32EL:#define __clang__ 1
// MIPS32EL:#define __llvm__ 1
// MIPS32EL:#define __mips 32
// MIPS32EL:#define __mips__ 1
// MIPS32EL:#define __mips_abicalls 1
// MIPS32EL:#define __mips_fpr 0
// MIPS32EL:#define __mips_hard_float 1
// MIPS32EL:#define __mips_o32 1
// MIPS32EL:#define _mips 1
// MIPS32EL:#define mips 1

// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 \
// RUN:            -triple=mips64-none-none -target-abi n32 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPSN32BE -check-prefix MIPSN32BE-C %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -fgnuc-version=4.2.1 \
// RUN:            -triple=mips64-none-none -target-abi n32 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPSN32BE -check-prefix MIPSN32BE-CXX %s
//
// MIPSN32BE: #define MIPSEB 1
// MIPSN32BE: #define _ABIN32 2
// MIPSN32BE: #define _ILP32 1
// MIPSN32BE: #define _MIPSEB 1
// MIPSN32BE: #define _MIPS_ARCH "mips64r2"
// MIPSN32BE: #define _MIPS_ARCH_MIPS64R2 1
// MIPSN32BE: #define _MIPS_FPSET 32
// MIPSN32BE: #define _MIPS_ISA _MIPS_ISA_MIPS64
// MIPSN32BE: #define _MIPS_SIM _ABIN32
// MIPSN32BE: #define _MIPS_SZINT 32
// MIPSN32BE: #define _MIPS_SZLONG 32
// MIPSN32BE: #define _MIPS_SZPTR 32
// MIPSN32BE: #define __ATOMIC_ACQUIRE 2
// MIPSN32BE: #define __ATOMIC_ACQ_REL 4
// MIPSN32BE: #define __ATOMIC_CONSUME 1
// MIPSN32BE: #define __ATOMIC_RELAXED 0
// MIPSN32BE: #define __ATOMIC_RELEASE 3
// MIPSN32BE: #define __ATOMIC_SEQ_CST 5
// MIPSN32BE: #define __BIG_ENDIAN__ 1
// MIPSN32BE: #define __BYTE_ORDER__ __ORDER_BIG_ENDIAN__
// MIPSN32BE: #define __CHAR16_TYPE__ unsigned short
// MIPSN32BE: #define __CHAR32_TYPE__ unsigned int
// MIPSN32BE: #define __CHAR_BIT__ 8
// MIPSN32BE: #define __CONSTANT_CFSTRINGS__ 1
// MIPSN32BE: #define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// MIPSN32BE: #define __DBL_DIG__ 15
// MIPSN32BE: #define __DBL_EPSILON__ 2.2204460492503131e-16
// MIPSN32BE: #define __DBL_HAS_DENORM__ 1
// MIPSN32BE: #define __DBL_HAS_INFINITY__ 1
// MIPSN32BE: #define __DBL_HAS_QUIET_NAN__ 1
// MIPSN32BE: #define __DBL_MANT_DIG__ 53
// MIPSN32BE: #define __DBL_MAX_10_EXP__ 308
// MIPSN32BE: #define __DBL_MAX_EXP__ 1024
// MIPSN32BE: #define __DBL_MAX__ 1.7976931348623157e+308
// MIPSN32BE: #define __DBL_MIN_10_EXP__ (-307)
// MIPSN32BE: #define __DBL_MIN_EXP__ (-1021)
// MIPSN32BE: #define __DBL_MIN__ 2.2250738585072014e-308
// MIPSN32BE: #define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// MIPSN32BE: #define __FINITE_MATH_ONLY__ 0
// MIPSN32BE: #define __FLT_DENORM_MIN__ 1.40129846e-45F
// MIPSN32BE: #define __FLT_DIG__ 6
// MIPSN32BE: #define __FLT_EPSILON__ 1.19209290e-7F
// MIPSN32BE: #define __FLT_EVAL_METHOD__ 0
// MIPSN32BE: #define __FLT_HAS_DENORM__ 1
// MIPSN32BE: #define __FLT_HAS_INFINITY__ 1
// MIPSN32BE: #define __FLT_HAS_QUIET_NAN__ 1
// MIPSN32BE: #define __FLT_MANT_DIG__ 24
// MIPSN32BE: #define __FLT_MAX_10_EXP__ 38
// MIPSN32BE: #define __FLT_MAX_EXP__ 128
// MIPSN32BE: #define __FLT_MAX__ 3.40282347e+38F
// MIPSN32BE: #define __FLT_MIN_10_EXP__ (-37)
// MIPSN32BE: #define __FLT_MIN_EXP__ (-125)
// MIPSN32BE: #define __FLT_MIN__ 1.17549435e-38F
// MIPSN32BE: #define __FLT_RADIX__ 2
// MIPSN32BE: #define __GCC_ATOMIC_BOOL_LOCK_FREE 2
// MIPSN32BE: #define __GCC_ATOMIC_CHAR16_T_LOCK_FREE 2
// MIPSN32BE: #define __GCC_ATOMIC_CHAR32_T_LOCK_FREE 2
// MIPSN32BE: #define __GCC_ATOMIC_CHAR_LOCK_FREE 2
// MIPSN32BE: #define __GCC_ATOMIC_INT_LOCK_FREE 2
// MIPSN32BE: #define __GCC_ATOMIC_LLONG_LOCK_FREE 2
// MIPSN32BE: #define __GCC_ATOMIC_LONG_LOCK_FREE 2
// MIPSN32BE: #define __GCC_ATOMIC_POINTER_LOCK_FREE 2
// MIPSN32BE: #define __GCC_ATOMIC_SHORT_LOCK_FREE 2
// MIPSN32BE: #define __GCC_ATOMIC_TEST_AND_SET_TRUEVAL 1
// MIPSN32BE: #define __GCC_ATOMIC_WCHAR_T_LOCK_FREE 2
// MIPSN32BE: #define __GNUC_MINOR__ 2
// MIPSN32BE: #define __GNUC_PATCHLEVEL__ 1
// MIPSN32BE-C: #define __GNUC_STDC_INLINE__ 1
// MIPSN32BE: #define __GNUC__ 4
// MIPSN32BE: #define __GXX_ABI_VERSION 1002
// MIPSN32BE: #define __ILP32__ 1
// MIPSN32BE: #define __INT16_C_SUFFIX__
// MIPSN32BE: #define __INT16_FMTd__ "hd"
// MIPSN32BE: #define __INT16_FMTi__ "hi"
// MIPSN32BE: #define __INT16_MAX__ 32767
// MIPSN32BE: #define __INT16_TYPE__ short
// MIPSN32BE: #define __INT32_C_SUFFIX__
// MIPSN32BE: #define __INT32_FMTd__ "d"
// MIPSN32BE: #define __INT32_FMTi__ "i"
// MIPSN32BE: #define __INT32_MAX__ 2147483647
// MIPSN32BE: #define __INT32_TYPE__ int
// MIPSN32BE: #define __INT64_C_SUFFIX__ LL
// MIPSN32BE: #define __INT64_FMTd__ "lld"
// MIPSN32BE: #define __INT64_FMTi__ "lli"
// MIPSN32BE: #define __INT64_MAX__ 9223372036854775807LL
// MIPSN32BE: #define __INT64_TYPE__ long long int
// MIPSN32BE: #define __INT8_C_SUFFIX__
// MIPSN32BE: #define __INT8_FMTd__ "hhd"
// MIPSN32BE: #define __INT8_FMTi__ "hhi"
// MIPSN32BE: #define __INT8_MAX__ 127
// MIPSN32BE: #define __INT8_TYPE__ signed char
// MIPSN32BE: #define __INTMAX_C_SUFFIX__ LL
// MIPSN32BE: #define __INTMAX_FMTd__ "lld"
// MIPSN32BE: #define __INTMAX_FMTi__ "lli"
// MIPSN32BE: #define __INTMAX_MAX__ 9223372036854775807LL
// MIPSN32BE: #define __INTMAX_TYPE__ long long int
// MIPSN32BE: #define __INTMAX_WIDTH__ 64
// MIPSN32BE: #define __INTPTR_FMTd__ "ld"
// MIPSN32BE: #define __INTPTR_FMTi__ "li"
// MIPSN32BE: #define __INTPTR_MAX__ 2147483647L
// MIPSN32BE: #define __INTPTR_TYPE__ long int
// MIPSN32BE: #define __INTPTR_WIDTH__ 32
// MIPSN32BE: #define __INT_FAST16_FMTd__ "hd"
// MIPSN32BE: #define __INT_FAST16_FMTi__ "hi"
// MIPSN32BE: #define __INT_FAST16_MAX__ 32767
// MIPSN32BE: #define __INT_FAST16_TYPE__ short
// MIPSN32BE: #define __INT_FAST32_FMTd__ "d"
// MIPSN32BE: #define __INT_FAST32_FMTi__ "i"
// MIPSN32BE: #define __INT_FAST32_MAX__ 2147483647
// MIPSN32BE: #define __INT_FAST32_TYPE__ int
// MIPSN32BE: #define __INT_FAST64_FMTd__ "lld"
// MIPSN32BE: #define __INT_FAST64_FMTi__ "lli"
// MIPSN32BE: #define __INT_FAST64_MAX__ 9223372036854775807LL
// MIPSN32BE: #define __INT_FAST64_TYPE__ long long int
// MIPSN32BE: #define __INT_FAST8_FMTd__ "hhd"
// MIPSN32BE: #define __INT_FAST8_FMTi__ "hhi"
// MIPSN32BE: #define __INT_FAST8_MAX__ 127
// MIPSN32BE: #define __INT_FAST8_TYPE__ signed char
// MIPSN32BE: #define __INT_LEAST16_FMTd__ "hd"
// MIPSN32BE: #define __INT_LEAST16_FMTi__ "hi"
// MIPSN32BE: #define __INT_LEAST16_MAX__ 32767
// MIPSN32BE: #define __INT_LEAST16_TYPE__ short
// MIPSN32BE: #define __INT_LEAST32_FMTd__ "d"
// MIPSN32BE: #define __INT_LEAST32_FMTi__ "i"
// MIPSN32BE: #define __INT_LEAST32_MAX__ 2147483647
// MIPSN32BE: #define __INT_LEAST32_TYPE__ int
// MIPSN32BE: #define __INT_LEAST64_FMTd__ "lld"
// MIPSN32BE: #define __INT_LEAST64_FMTi__ "lli"
// MIPSN32BE: #define __INT_LEAST64_MAX__ 9223372036854775807LL
// MIPSN32BE: #define __INT_LEAST64_TYPE__ long long int
// MIPSN32BE: #define __INT_LEAST8_FMTd__ "hhd"
// MIPSN32BE: #define __INT_LEAST8_FMTi__ "hhi"
// MIPSN32BE: #define __INT_LEAST8_MAX__ 127
// MIPSN32BE: #define __INT_LEAST8_TYPE__ signed char
// MIPSN32BE: #define __INT_MAX__ 2147483647
// MIPSN32BE: #define __LDBL_DENORM_MIN__ 6.47517511943802511092443895822764655e-4966L
// MIPSN32BE: #define __LDBL_DIG__ 33
// MIPSN32BE: #define __LDBL_EPSILON__ 1.92592994438723585305597794258492732e-34L
// MIPSN32BE: #define __LDBL_HAS_DENORM__ 1
// MIPSN32BE: #define __LDBL_HAS_INFINITY__ 1
// MIPSN32BE: #define __LDBL_HAS_QUIET_NAN__ 1
// MIPSN32BE: #define __LDBL_MANT_DIG__ 113
// MIPSN32BE: #define __LDBL_MAX_10_EXP__ 4932
// MIPSN32BE: #define __LDBL_MAX_EXP__ 16384
// MIPSN32BE: #define __LDBL_MAX__ 1.18973149535723176508575932662800702e+4932L
// MIPSN32BE: #define __LDBL_MIN_10_EXP__ (-4931)
// MIPSN32BE: #define __LDBL_MIN_EXP__ (-16381)
// MIPSN32BE: #define __LDBL_MIN__ 3.36210314311209350626267781732175260e-4932L
// MIPSN32BE: #define __LONG_LONG_MAX__ 9223372036854775807LL
// MIPSN32BE: #define __LONG_MAX__ 2147483647L
// MIPSN32BE: #define __MIPSEB 1
// MIPSN32BE: #define __MIPSEB__ 1
// MIPSN32BE: #define __NO_INLINE__ 1
// MIPSN32BE: #define __ORDER_BIG_ENDIAN__ 4321
// MIPSN32BE: #define __ORDER_LITTLE_ENDIAN__ 1234
// MIPSN32BE: #define __ORDER_PDP_ENDIAN__ 3412
// MIPSN32BE: #define __POINTER_WIDTH__ 32
// MIPSN32BE: #define __PRAGMA_REDEFINE_EXTNAME 1
// MIPSN32BE: #define __PTRDIFF_FMTd__ "d"
// MIPSN32BE: #define __PTRDIFF_FMTi__ "i"
// MIPSN32BE: #define __PTRDIFF_MAX__ 2147483647
// MIPSN32BE: #define __PTRDIFF_TYPE__ int
// MIPSN32BE: #define __PTRDIFF_WIDTH__ 32
// MIPSN32BE: #define __REGISTER_PREFIX__
// MIPSN32BE: #define __SCHAR_MAX__ 127
// MIPSN32BE: #define __SHRT_MAX__ 32767
// MIPSN32BE: #define __SIG_ATOMIC_MAX__ 2147483647
// MIPSN32BE: #define __SIG_ATOMIC_WIDTH__ 32
// MIPSN32BE: #define __SIZEOF_DOUBLE__ 8
// MIPSN32BE: #define __SIZEOF_FLOAT__ 4
// MIPSN32BE: #define __SIZEOF_INT__ 4
// MIPSN32BE: #define __SIZEOF_LONG_DOUBLE__ 16
// MIPSN32BE: #define __SIZEOF_LONG_LONG__ 8
// MIPSN32BE: #define __SIZEOF_LONG__ 4
// MIPSN32BE: #define __SIZEOF_POINTER__ 4
// MIPSN32BE: #define __SIZEOF_PTRDIFF_T__ 4
// MIPSN32BE: #define __SIZEOF_SHORT__ 2
// MIPSN32BE: #define __SIZEOF_SIZE_T__ 4
// MIPSN32BE: #define __SIZEOF_WCHAR_T__ 4
// MIPSN32BE: #define __SIZEOF_WINT_T__ 4
// MIPSN32BE: #define __SIZE_FMTX__ "X"
// MIPSN32BE: #define __SIZE_FMTo__ "o"
// MIPSN32BE: #define __SIZE_FMTu__ "u"
// MIPSN32BE: #define __SIZE_FMTx__ "x"
// MIPSN32BE: #define __SIZE_MAX__ 4294967295U
// MIPSN32BE: #define __SIZE_TYPE__ unsigned int
// MIPSN32BE: #define __SIZE_WIDTH__ 32
// MIPSN32BE-CXX: #define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 16U
// MIPSN32BE: #define __STDC_HOSTED__ 0
// MIPSN32BE: #define __STDC_UTF_16__ 1
// MIPSN32BE: #define __STDC_UTF_32__ 1
// MIPSN32BE-C: #define __STDC_VERSION__ 201710L
// MIPSN32BE: #define __STDC__ 1
// MIPSN32BE: #define __UINT16_C_SUFFIX__
// MIPSN32BE: #define __UINT16_FMTX__ "hX"
// MIPSN32BE: #define __UINT16_FMTo__ "ho"
// MIPSN32BE: #define __UINT16_FMTu__ "hu"
// MIPSN32BE: #define __UINT16_FMTx__ "hx"
// MIPSN32BE: #define __UINT16_MAX__ 65535
// MIPSN32BE: #define __UINT16_TYPE__ unsigned short
// MIPSN32BE: #define __UINT32_C_SUFFIX__ U
// MIPSN32BE: #define __UINT32_FMTX__ "X"
// MIPSN32BE: #define __UINT32_FMTo__ "o"
// MIPSN32BE: #define __UINT32_FMTu__ "u"
// MIPSN32BE: #define __UINT32_FMTx__ "x"
// MIPSN32BE: #define __UINT32_MAX__ 4294967295U
// MIPSN32BE: #define __UINT32_TYPE__ unsigned int
// MIPSN32BE: #define __UINT64_C_SUFFIX__ ULL
// MIPSN32BE: #define __UINT64_FMTX__ "llX"
// MIPSN32BE: #define __UINT64_FMTo__ "llo"
// MIPSN32BE: #define __UINT64_FMTu__ "llu"
// MIPSN32BE: #define __UINT64_FMTx__ "llx"
// MIPSN32BE: #define __UINT64_MAX__ 18446744073709551615ULL
// MIPSN32BE: #define __UINT64_TYPE__ long long unsigned int
// MIPSN32BE: #define __UINT8_C_SUFFIX__
// MIPSN32BE: #define __UINT8_FMTX__ "hhX"
// MIPSN32BE: #define __UINT8_FMTo__ "hho"
// MIPSN32BE: #define __UINT8_FMTu__ "hhu"
// MIPSN32BE: #define __UINT8_FMTx__ "hhx"
// MIPSN32BE: #define __UINT8_MAX__ 255
// MIPSN32BE: #define __UINT8_TYPE__ unsigned char
// MIPSN32BE: #define __UINTMAX_C_SUFFIX__ ULL
// MIPSN32BE: #define __UINTMAX_FMTX__ "llX"
// MIPSN32BE: #define __UINTMAX_FMTo__ "llo"
// MIPSN32BE: #define __UINTMAX_FMTu__ "llu"
// MIPSN32BE: #define __UINTMAX_FMTx__ "llx"
// MIPSN32BE: #define __UINTMAX_MAX__ 18446744073709551615ULL
// MIPSN32BE: #define __UINTMAX_TYPE__ long long unsigned int
// MIPSN32BE: #define __UINTMAX_WIDTH__ 64
// MIPSN32BE: #define __UINTPTR_FMTX__ "lX"
// MIPSN32BE: #define __UINTPTR_FMTo__ "lo"
// MIPSN32BE: #define __UINTPTR_FMTu__ "lu"
// MIPSN32BE: #define __UINTPTR_FMTx__ "lx"
// MIPSN32BE: #define __UINTPTR_MAX__ 4294967295UL
// MIPSN32BE: #define __UINTPTR_TYPE__ long unsigned int
// MIPSN32BE: #define __UINTPTR_WIDTH__ 32
// MIPSN32BE: #define __UINT_FAST16_FMTX__ "hX"
// MIPSN32BE: #define __UINT_FAST16_FMTo__ "ho"
// MIPSN32BE: #define __UINT_FAST16_FMTu__ "hu"
// MIPSN32BE: #define __UINT_FAST16_FMTx__ "hx"
// MIPSN32BE: #define __UINT_FAST16_MAX__ 65535
// MIPSN32BE: #define __UINT_FAST16_TYPE__ unsigned short
// MIPSN32BE: #define __UINT_FAST32_FMTX__ "X"
// MIPSN32BE: #define __UINT_FAST32_FMTo__ "o"
// MIPSN32BE: #define __UINT_FAST32_FMTu__ "u"
// MIPSN32BE: #define __UINT_FAST32_FMTx__ "x"
// MIPSN32BE: #define __UINT_FAST32_MAX__ 4294967295U
// MIPSN32BE: #define __UINT_FAST32_TYPE__ unsigned int
// MIPSN32BE: #define __UINT_FAST64_FMTX__ "llX"
// MIPSN32BE: #define __UINT_FAST64_FMTo__ "llo"
// MIPSN32BE: #define __UINT_FAST64_FMTu__ "llu"
// MIPSN32BE: #define __UINT_FAST64_FMTx__ "llx"
// MIPSN32BE: #define __UINT_FAST64_MAX__ 18446744073709551615ULL
// MIPSN32BE: #define __UINT_FAST64_TYPE__ long long unsigned int
// MIPSN32BE: #define __UINT_FAST8_FMTX__ "hhX"
// MIPSN32BE: #define __UINT_FAST8_FMTo__ "hho"
// MIPSN32BE: #define __UINT_FAST8_FMTu__ "hhu"
// MIPSN32BE: #define __UINT_FAST8_FMTx__ "hhx"
// MIPSN32BE: #define __UINT_FAST8_MAX__ 255
// MIPSN32BE: #define __UINT_FAST8_TYPE__ unsigned char
// MIPSN32BE: #define __UINT_LEAST16_FMTX__ "hX"
// MIPSN32BE: #define __UINT_LEAST16_FMTo__ "ho"
// MIPSN32BE: #define __UINT_LEAST16_FMTu__ "hu"
// MIPSN32BE: #define __UINT_LEAST16_FMTx__ "hx"
// MIPSN32BE: #define __UINT_LEAST16_MAX__ 65535
// MIPSN32BE: #define __UINT_LEAST16_TYPE__ unsigned short
// MIPSN32BE: #define __UINT_LEAST32_FMTX__ "X"
// MIPSN32BE: #define __UINT_LEAST32_FMTo__ "o"
// MIPSN32BE: #define __UINT_LEAST32_FMTu__ "u"
// MIPSN32BE: #define __UINT_LEAST32_FMTx__ "x"
// MIPSN32BE: #define __UINT_LEAST32_MAX__ 4294967295U
// MIPSN32BE: #define __UINT_LEAST32_TYPE__ unsigned int
// MIPSN32BE: #define __UINT_LEAST64_FMTX__ "llX"
// MIPSN32BE: #define __UINT_LEAST64_FMTo__ "llo"
// MIPSN32BE: #define __UINT_LEAST64_FMTu__ "llu"
// MIPSN32BE: #define __UINT_LEAST64_FMTx__ "llx"
// MIPSN32BE: #define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// MIPSN32BE: #define __UINT_LEAST64_TYPE__ long long unsigned int
// MIPSN32BE: #define __UINT_LEAST8_FMTX__ "hhX"
// MIPSN32BE: #define __UINT_LEAST8_FMTo__ "hho"
// MIPSN32BE: #define __UINT_LEAST8_FMTu__ "hhu"
// MIPSN32BE: #define __UINT_LEAST8_FMTx__ "hhx"
// MIPSN32BE: #define __UINT_LEAST8_MAX__ 255
// MIPSN32BE: #define __UINT_LEAST8_TYPE__ unsigned char
// MIPSN32BE: #define __USER_LABEL_PREFIX__
// MIPSN32BE: #define __WCHAR_MAX__ 2147483647
// MIPSN32BE: #define __WCHAR_TYPE__ int
// MIPSN32BE: #define __WCHAR_WIDTH__ 32
// MIPSN32BE: #define __WINT_TYPE__ int
// MIPSN32BE: #define __WINT_WIDTH__ 32
// MIPSN32BE: #define __clang__ 1
// MIPSN32BE: #define __llvm__ 1
// MIPSN32BE: #define __mips 64
// MIPSN32BE: #define __mips64 1
// MIPSN32BE: #define __mips64__ 1
// MIPSN32BE: #define __mips__ 1
// MIPSN32BE: #define __mips_abicalls 1
// MIPSN32BE: #define __mips_fpr 64
// MIPSN32BE: #define __mips_hard_float 1
// MIPSN32BE: #define __mips_isa_rev 2
// MIPSN32BE: #define __mips_n32 1
// MIPSN32BE: #define _mips 1
// MIPSN32BE: #define mips 1

// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 \
// RUN:            -triple=mips64el-none-none -target-abi n32 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPSN32EL %s
//
// MIPSN32EL: #define MIPSEL 1
// MIPSN32EL: #define _ABIN32 2
// MIPSN32EL: #define _ILP32 1
// MIPSN32EL: #define _MIPSEL 1
// MIPSN32EL: #define _MIPS_ARCH "mips64r2"
// MIPSN32EL: #define _MIPS_ARCH_MIPS64R2 1
// MIPSN32EL: #define _MIPS_FPSET 32
// MIPSN32EL: #define _MIPS_ISA _MIPS_ISA_MIPS64
// MIPSN32EL: #define _MIPS_SIM _ABIN32
// MIPSN32EL: #define _MIPS_SZINT 32
// MIPSN32EL: #define _MIPS_SZLONG 32
// MIPSN32EL: #define _MIPS_SZPTR 32
// MIPSN32EL: #define __ATOMIC_ACQUIRE 2
// MIPSN32EL: #define __ATOMIC_ACQ_REL 4
// MIPSN32EL: #define __ATOMIC_CONSUME 1
// MIPSN32EL: #define __ATOMIC_RELAXED 0
// MIPSN32EL: #define __ATOMIC_RELEASE 3
// MIPSN32EL: #define __ATOMIC_SEQ_CST 5
// MIPSN32EL: #define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// MIPSN32EL: #define __CHAR16_TYPE__ unsigned short
// MIPSN32EL: #define __CHAR32_TYPE__ unsigned int
// MIPSN32EL: #define __CHAR_BIT__ 8
// MIPSN32EL: #define __CONSTANT_CFSTRINGS__ 1
// MIPSN32EL: #define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// MIPSN32EL: #define __DBL_DIG__ 15
// MIPSN32EL: #define __DBL_EPSILON__ 2.2204460492503131e-16
// MIPSN32EL: #define __DBL_HAS_DENORM__ 1
// MIPSN32EL: #define __DBL_HAS_INFINITY__ 1
// MIPSN32EL: #define __DBL_HAS_QUIET_NAN__ 1
// MIPSN32EL: #define __DBL_MANT_DIG__ 53
// MIPSN32EL: #define __DBL_MAX_10_EXP__ 308
// MIPSN32EL: #define __DBL_MAX_EXP__ 1024
// MIPSN32EL: #define __DBL_MAX__ 1.7976931348623157e+308
// MIPSN32EL: #define __DBL_MIN_10_EXP__ (-307)
// MIPSN32EL: #define __DBL_MIN_EXP__ (-1021)
// MIPSN32EL: #define __DBL_MIN__ 2.2250738585072014e-308
// MIPSN32EL: #define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// MIPSN32EL: #define __FINITE_MATH_ONLY__ 0
// MIPSN32EL: #define __FLT_DENORM_MIN__ 1.40129846e-45F
// MIPSN32EL: #define __FLT_DIG__ 6
// MIPSN32EL: #define __FLT_EPSILON__ 1.19209290e-7F
// MIPSN32EL: #define __FLT_EVAL_METHOD__ 0
// MIPSN32EL: #define __FLT_HAS_DENORM__ 1
// MIPSN32EL: #define __FLT_HAS_INFINITY__ 1
// MIPSN32EL: #define __FLT_HAS_QUIET_NAN__ 1
// MIPSN32EL: #define __FLT_MANT_DIG__ 24
// MIPSN32EL: #define __FLT_MAX_10_EXP__ 38
// MIPSN32EL: #define __FLT_MAX_EXP__ 128
// MIPSN32EL: #define __FLT_MAX__ 3.40282347e+38F
// MIPSN32EL: #define __FLT_MIN_10_EXP__ (-37)
// MIPSN32EL: #define __FLT_MIN_EXP__ (-125)
// MIPSN32EL: #define __FLT_MIN__ 1.17549435e-38F
// MIPSN32EL: #define __FLT_RADIX__ 2
// MIPSN32EL: #define __GCC_ATOMIC_BOOL_LOCK_FREE 2
// MIPSN32EL: #define __GCC_ATOMIC_CHAR16_T_LOCK_FREE 2
// MIPSN32EL: #define __GCC_ATOMIC_CHAR32_T_LOCK_FREE 2
// MIPSN32EL: #define __GCC_ATOMIC_CHAR_LOCK_FREE 2
// MIPSN32EL: #define __GCC_ATOMIC_INT_LOCK_FREE 2
// MIPSN32EL: #define __GCC_ATOMIC_LLONG_LOCK_FREE 2
// MIPSN32EL: #define __GCC_ATOMIC_LONG_LOCK_FREE 2
// MIPSN32EL: #define __GCC_ATOMIC_POINTER_LOCK_FREE 2
// MIPSN32EL: #define __GCC_ATOMIC_SHORT_LOCK_FREE 2
// MIPSN32EL: #define __GCC_ATOMIC_TEST_AND_SET_TRUEVAL 1
// MIPSN32EL: #define __GCC_ATOMIC_WCHAR_T_LOCK_FREE 2
// MIPSN32EL: #define __GNUC_MINOR__ 2
// MIPSN32EL: #define __GNUC_PATCHLEVEL__ 1
// MIPSN32EL: #define __GNUC_STDC_INLINE__ 1
// MIPSN32EL: #define __GNUC__ 4
// MIPSN32EL: #define __GXX_ABI_VERSION 1002
// MIPSN32EL: #define __ILP32__ 1
// MIPSN32EL: #define __INT16_C_SUFFIX__
// MIPSN32EL: #define __INT16_FMTd__ "hd"
// MIPSN32EL: #define __INT16_FMTi__ "hi"
// MIPSN32EL: #define __INT16_MAX__ 32767
// MIPSN32EL: #define __INT16_TYPE__ short
// MIPSN32EL: #define __INT32_C_SUFFIX__
// MIPSN32EL: #define __INT32_FMTd__ "d"
// MIPSN32EL: #define __INT32_FMTi__ "i"
// MIPSN32EL: #define __INT32_MAX__ 2147483647
// MIPSN32EL: #define __INT32_TYPE__ int
// MIPSN32EL: #define __INT64_C_SUFFIX__ LL
// MIPSN32EL: #define __INT64_FMTd__ "lld"
// MIPSN32EL: #define __INT64_FMTi__ "lli"
// MIPSN32EL: #define __INT64_MAX__ 9223372036854775807LL
// MIPSN32EL: #define __INT64_TYPE__ long long int
// MIPSN32EL: #define __INT8_C_SUFFIX__
// MIPSN32EL: #define __INT8_FMTd__ "hhd"
// MIPSN32EL: #define __INT8_FMTi__ "hhi"
// MIPSN32EL: #define __INT8_MAX__ 127
// MIPSN32EL: #define __INT8_TYPE__ signed char
// MIPSN32EL: #define __INTMAX_C_SUFFIX__ LL
// MIPSN32EL: #define __INTMAX_FMTd__ "lld"
// MIPSN32EL: #define __INTMAX_FMTi__ "lli"
// MIPSN32EL: #define __INTMAX_MAX__ 9223372036854775807LL
// MIPSN32EL: #define __INTMAX_TYPE__ long long int
// MIPSN32EL: #define __INTMAX_WIDTH__ 64
// MIPSN32EL: #define __INTPTR_FMTd__ "ld"
// MIPSN32EL: #define __INTPTR_FMTi__ "li"
// MIPSN32EL: #define __INTPTR_MAX__ 2147483647L
// MIPSN32EL: #define __INTPTR_TYPE__ long int
// MIPSN32EL: #define __INTPTR_WIDTH__ 32
// MIPSN32EL: #define __INT_FAST16_FMTd__ "hd"
// MIPSN32EL: #define __INT_FAST16_FMTi__ "hi"
// MIPSN32EL: #define __INT_FAST16_MAX__ 32767
// MIPSN32EL: #define __INT_FAST16_TYPE__ short
// MIPSN32EL: #define __INT_FAST32_FMTd__ "d"
// MIPSN32EL: #define __INT_FAST32_FMTi__ "i"
// MIPSN32EL: #define __INT_FAST32_MAX__ 2147483647
// MIPSN32EL: #define __INT_FAST32_TYPE__ int
// MIPSN32EL: #define __INT_FAST64_FMTd__ "lld"
// MIPSN32EL: #define __INT_FAST64_FMTi__ "lli"
// MIPSN32EL: #define __INT_FAST64_MAX__ 9223372036854775807LL
// MIPSN32EL: #define __INT_FAST64_TYPE__ long long int
// MIPSN32EL: #define __INT_FAST8_FMTd__ "hhd"
// MIPSN32EL: #define __INT_FAST8_FMTi__ "hhi"
// MIPSN32EL: #define __INT_FAST8_MAX__ 127
// MIPSN32EL: #define __INT_FAST8_TYPE__ signed char
// MIPSN32EL: #define __INT_LEAST16_FMTd__ "hd"
// MIPSN32EL: #define __INT_LEAST16_FMTi__ "hi"
// MIPSN32EL: #define __INT_LEAST16_MAX__ 32767
// MIPSN32EL: #define __INT_LEAST16_TYPE__ short
// MIPSN32EL: #define __INT_LEAST32_FMTd__ "d"
// MIPSN32EL: #define __INT_LEAST32_FMTi__ "i"
// MIPSN32EL: #define __INT_LEAST32_MAX__ 2147483647
// MIPSN32EL: #define __INT_LEAST32_TYPE__ int
// MIPSN32EL: #define __INT_LEAST64_FMTd__ "lld"
// MIPSN32EL: #define __INT_LEAST64_FMTi__ "lli"
// MIPSN32EL: #define __INT_LEAST64_MAX__ 9223372036854775807LL
// MIPSN32EL: #define __INT_LEAST64_TYPE__ long long int
// MIPSN32EL: #define __INT_LEAST8_FMTd__ "hhd"
// MIPSN32EL: #define __INT_LEAST8_FMTi__ "hhi"
// MIPSN32EL: #define __INT_LEAST8_MAX__ 127
// MIPSN32EL: #define __INT_LEAST8_TYPE__ signed char
// MIPSN32EL: #define __INT_MAX__ 2147483647
// MIPSN32EL: #define __LDBL_DENORM_MIN__ 6.47517511943802511092443895822764655e-4966L
// MIPSN32EL: #define __LDBL_DIG__ 33
// MIPSN32EL: #define __LDBL_EPSILON__ 1.92592994438723585305597794258492732e-34L
// MIPSN32EL: #define __LDBL_HAS_DENORM__ 1
// MIPSN32EL: #define __LDBL_HAS_INFINITY__ 1
// MIPSN32EL: #define __LDBL_HAS_QUIET_NAN__ 1
// MIPSN32EL: #define __LDBL_MANT_DIG__ 113
// MIPSN32EL: #define __LDBL_MAX_10_EXP__ 4932
// MIPSN32EL: #define __LDBL_MAX_EXP__ 16384
// MIPSN32EL: #define __LDBL_MAX__ 1.18973149535723176508575932662800702e+4932L
// MIPSN32EL: #define __LDBL_MIN_10_EXP__ (-4931)
// MIPSN32EL: #define __LDBL_MIN_EXP__ (-16381)
// MIPSN32EL: #define __LDBL_MIN__ 3.36210314311209350626267781732175260e-4932L
// MIPSN32EL: #define __LITTLE_ENDIAN__ 1
// MIPSN32EL: #define __LONG_LONG_MAX__ 9223372036854775807LL
// MIPSN32EL: #define __LONG_MAX__ 2147483647L
// MIPSN32EL: #define __MIPSEL 1
// MIPSN32EL: #define __MIPSEL__ 1
// MIPSN32EL: #define __NO_INLINE__ 1
// MIPSN32EL: #define __ORDER_BIG_ENDIAN__ 4321
// MIPSN32EL: #define __ORDER_LITTLE_ENDIAN__ 1234
// MIPSN32EL: #define __ORDER_PDP_ENDIAN__ 3412
// MIPSN32EL: #define __POINTER_WIDTH__ 32
// MIPSN32EL: #define __PRAGMA_REDEFINE_EXTNAME 1
// MIPSN32EL: #define __PTRDIFF_FMTd__ "d"
// MIPSN32EL: #define __PTRDIFF_FMTi__ "i"
// MIPSN32EL: #define __PTRDIFF_MAX__ 2147483647
// MIPSN32EL: #define __PTRDIFF_TYPE__ int
// MIPSN32EL: #define __PTRDIFF_WIDTH__ 32
// MIPSN32EL: #define __REGISTER_PREFIX__
// MIPSN32EL: #define __SCHAR_MAX__ 127
// MIPSN32EL: #define __SHRT_MAX__ 32767
// MIPSN32EL: #define __SIG_ATOMIC_MAX__ 2147483647
// MIPSN32EL: #define __SIG_ATOMIC_WIDTH__ 32
// MIPSN32EL: #define __SIZEOF_DOUBLE__ 8
// MIPSN32EL: #define __SIZEOF_FLOAT__ 4
// MIPSN32EL: #define __SIZEOF_INT__ 4
// MIPSN32EL: #define __SIZEOF_LONG_DOUBLE__ 16
// MIPSN32EL: #define __SIZEOF_LONG_LONG__ 8
// MIPSN32EL: #define __SIZEOF_LONG__ 4
// MIPSN32EL: #define __SIZEOF_POINTER__ 4
// MIPSN32EL: #define __SIZEOF_PTRDIFF_T__ 4
// MIPSN32EL: #define __SIZEOF_SHORT__ 2
// MIPSN32EL: #define __SIZEOF_SIZE_T__ 4
// MIPSN32EL: #define __SIZEOF_WCHAR_T__ 4
// MIPSN32EL: #define __SIZEOF_WINT_T__ 4
// MIPSN32EL: #define __SIZE_FMTX__ "X"
// MIPSN32EL: #define __SIZE_FMTo__ "o"
// MIPSN32EL: #define __SIZE_FMTu__ "u"
// MIPSN32EL: #define __SIZE_FMTx__ "x"
// MIPSN32EL: #define __SIZE_MAX__ 4294967295U
// MIPSN32EL: #define __SIZE_TYPE__ unsigned int
// MIPSN32EL: #define __SIZE_WIDTH__ 32
// MIPSN32EL: #define __STDC_HOSTED__ 0
// MIPSN32EL: #define __STDC_UTF_16__ 1
// MIPSN32EL: #define __STDC_UTF_32__ 1
// MIPSN32EL: #define __STDC_VERSION__ 201710L
// MIPSN32EL: #define __STDC__ 1
// MIPSN32EL: #define __UINT16_C_SUFFIX__
// MIPSN32EL: #define __UINT16_FMTX__ "hX"
// MIPSN32EL: #define __UINT16_FMTo__ "ho"
// MIPSN32EL: #define __UINT16_FMTu__ "hu"
// MIPSN32EL: #define __UINT16_FMTx__ "hx"
// MIPSN32EL: #define __UINT16_MAX__ 65535
// MIPSN32EL: #define __UINT16_TYPE__ unsigned short
// MIPSN32EL: #define __UINT32_C_SUFFIX__ U
// MIPSN32EL: #define __UINT32_FMTX__ "X"
// MIPSN32EL: #define __UINT32_FMTo__ "o"
// MIPSN32EL: #define __UINT32_FMTu__ "u"
// MIPSN32EL: #define __UINT32_FMTx__ "x"
// MIPSN32EL: #define __UINT32_MAX__ 4294967295U
// MIPSN32EL: #define __UINT32_TYPE__ unsigned int
// MIPSN32EL: #define __UINT64_C_SUFFIX__ ULL
// MIPSN32EL: #define __UINT64_FMTX__ "llX"
// MIPSN32EL: #define __UINT64_FMTo__ "llo"
// MIPSN32EL: #define __UINT64_FMTu__ "llu"
// MIPSN32EL: #define __UINT64_FMTx__ "llx"
// MIPSN32EL: #define __UINT64_MAX__ 18446744073709551615ULL
// MIPSN32EL: #define __UINT64_TYPE__ long long unsigned int
// MIPSN32EL: #define __UINT8_C_SUFFIX__
// MIPSN32EL: #define __UINT8_FMTX__ "hhX"
// MIPSN32EL: #define __UINT8_FMTo__ "hho"
// MIPSN32EL: #define __UINT8_FMTu__ "hhu"
// MIPSN32EL: #define __UINT8_FMTx__ "hhx"
// MIPSN32EL: #define __UINT8_MAX__ 255
// MIPSN32EL: #define __UINT8_TYPE__ unsigned char
// MIPSN32EL: #define __UINTMAX_C_SUFFIX__ ULL
// MIPSN32EL: #define __UINTMAX_FMTX__ "llX"
// MIPSN32EL: #define __UINTMAX_FMTo__ "llo"
// MIPSN32EL: #define __UINTMAX_FMTu__ "llu"
// MIPSN32EL: #define __UINTMAX_FMTx__ "llx"
// MIPSN32EL: #define __UINTMAX_MAX__ 18446744073709551615ULL
// MIPSN32EL: #define __UINTMAX_TYPE__ long long unsigned int
// MIPSN32EL: #define __UINTMAX_WIDTH__ 64
// MIPSN32EL: #define __UINTPTR_FMTX__ "lX"
// MIPSN32EL: #define __UINTPTR_FMTo__ "lo"
// MIPSN32EL: #define __UINTPTR_FMTu__ "lu"
// MIPSN32EL: #define __UINTPTR_FMTx__ "lx"
// MIPSN32EL: #define __UINTPTR_MAX__ 4294967295UL
// MIPSN32EL: #define __UINTPTR_TYPE__ long unsigned int
// MIPSN32EL: #define __UINTPTR_WIDTH__ 32
// MIPSN32EL: #define __UINT_FAST16_FMTX__ "hX"
// MIPSN32EL: #define __UINT_FAST16_FMTo__ "ho"
// MIPSN32EL: #define __UINT_FAST16_FMTu__ "hu"
// MIPSN32EL: #define __UINT_FAST16_FMTx__ "hx"
// MIPSN32EL: #define __UINT_FAST16_MAX__ 65535
// MIPSN32EL: #define __UINT_FAST16_TYPE__ unsigned short
// MIPSN32EL: #define __UINT_FAST32_FMTX__ "X"
// MIPSN32EL: #define __UINT_FAST32_FMTo__ "o"
// MIPSN32EL: #define __UINT_FAST32_FMTu__ "u"
// MIPSN32EL: #define __UINT_FAST32_FMTx__ "x"
// MIPSN32EL: #define __UINT_FAST32_MAX__ 4294967295U
// MIPSN32EL: #define __UINT_FAST32_TYPE__ unsigned int
// MIPSN32EL: #define __UINT_FAST64_FMTX__ "llX"
// MIPSN32EL: #define __UINT_FAST64_FMTo__ "llo"
// MIPSN32EL: #define __UINT_FAST64_FMTu__ "llu"
// MIPSN32EL: #define __UINT_FAST64_FMTx__ "llx"
// MIPSN32EL: #define __UINT_FAST64_MAX__ 18446744073709551615ULL
// MIPSN32EL: #define __UINT_FAST64_TYPE__ long long unsigned int
// MIPSN32EL: #define __UINT_FAST8_FMTX__ "hhX"
// MIPSN32EL: #define __UINT_FAST8_FMTo__ "hho"
// MIPSN32EL: #define __UINT_FAST8_FMTu__ "hhu"
// MIPSN32EL: #define __UINT_FAST8_FMTx__ "hhx"
// MIPSN32EL: #define __UINT_FAST8_MAX__ 255
// MIPSN32EL: #define __UINT_FAST8_TYPE__ unsigned char
// MIPSN32EL: #define __UINT_LEAST16_FMTX__ "hX"
// MIPSN32EL: #define __UINT_LEAST16_FMTo__ "ho"
// MIPSN32EL: #define __UINT_LEAST16_FMTu__ "hu"
// MIPSN32EL: #define __UINT_LEAST16_FMTx__ "hx"
// MIPSN32EL: #define __UINT_LEAST16_MAX__ 65535
// MIPSN32EL: #define __UINT_LEAST16_TYPE__ unsigned short
// MIPSN32EL: #define __UINT_LEAST32_FMTX__ "X"
// MIPSN32EL: #define __UINT_LEAST32_FMTo__ "o"
// MIPSN32EL: #define __UINT_LEAST32_FMTu__ "u"
// MIPSN32EL: #define __UINT_LEAST32_FMTx__ "x"
// MIPSN32EL: #define __UINT_LEAST32_MAX__ 4294967295U
// MIPSN32EL: #define __UINT_LEAST32_TYPE__ unsigned int
// MIPSN32EL: #define __UINT_LEAST64_FMTX__ "llX"
// MIPSN32EL: #define __UINT_LEAST64_FMTo__ "llo"
// MIPSN32EL: #define __UINT_LEAST64_FMTu__ "llu"
// MIPSN32EL: #define __UINT_LEAST64_FMTx__ "llx"
// MIPSN32EL: #define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// MIPSN32EL: #define __UINT_LEAST64_TYPE__ long long unsigned int
// MIPSN32EL: #define __UINT_LEAST8_FMTX__ "hhX"
// MIPSN32EL: #define __UINT_LEAST8_FMTo__ "hho"
// MIPSN32EL: #define __UINT_LEAST8_FMTu__ "hhu"
// MIPSN32EL: #define __UINT_LEAST8_FMTx__ "hhx"
// MIPSN32EL: #define __UINT_LEAST8_MAX__ 255
// MIPSN32EL: #define __UINT_LEAST8_TYPE__ unsigned char
// MIPSN32EL: #define __USER_LABEL_PREFIX__
// MIPSN32EL: #define __WCHAR_MAX__ 2147483647
// MIPSN32EL: #define __WCHAR_TYPE__ int
// MIPSN32EL: #define __WCHAR_WIDTH__ 32
// MIPSN32EL: #define __WINT_TYPE__ int
// MIPSN32EL: #define __WINT_WIDTH__ 32
// MIPSN32EL: #define __clang__ 1
// MIPSN32EL: #define __llvm__ 1
// MIPSN32EL: #define __mips 64
// MIPSN32EL: #define __mips64 1
// MIPSN32EL: #define __mips64__ 1
// MIPSN32EL: #define __mips__ 1
// MIPSN32EL: #define __mips_abicalls 1
// MIPSN32EL: #define __mips_fpr 64
// MIPSN32EL: #define __mips_hard_float 1
// MIPSN32EL: #define __mips_isa_rev 2
// MIPSN32EL: #define __mips_n32 1
// MIPSN32EL: #define _mips 1
// MIPSN32EL: #define mips 1

// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=mips64-none-none < /dev/null | FileCheck -match-full-lines -check-prefix MIPS64BE %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=mips64-none-none < /dev/null | FileCheck -match-full-lines -check-prefix MIPS64BE -check-prefix MIPS64BE-CXX %s
//
// MIPS64BE:#define MIPSEB 1
// MIPS64BE:#define _ABI64 3
// MIPS64BE:#define _LP64 1
// MIPS64BE:#define _MIPSEB 1
// MIPS64BE:#define _MIPS_ARCH "mips64r2"
// MIPS64BE:#define _MIPS_ARCH_MIPS64R2 1
// MIPS64BE:#define _MIPS_FPSET 32
// MIPS64BE:#define _MIPS_SIM _ABI64
// MIPS64BE:#define _MIPS_SZINT 32
// MIPS64BE:#define _MIPS_SZLONG 64
// MIPS64BE:#define _MIPS_SZPTR 64
// MIPS64BE:#define __BIGGEST_ALIGNMENT__ 16
// MIPS64BE:#define __BIG_ENDIAN__ 1
// MIPS64BE:#define __BYTE_ORDER__ __ORDER_BIG_ENDIAN__
// MIPS64BE:#define __CHAR16_TYPE__ unsigned short
// MIPS64BE:#define __CHAR32_TYPE__ unsigned int
// MIPS64BE:#define __CHAR_BIT__ 8
// MIPS64BE:#define __CONSTANT_CFSTRINGS__ 1
// MIPS64BE:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// MIPS64BE:#define __DBL_DIG__ 15
// MIPS64BE:#define __DBL_EPSILON__ 2.2204460492503131e-16
// MIPS64BE:#define __DBL_HAS_DENORM__ 1
// MIPS64BE:#define __DBL_HAS_INFINITY__ 1
// MIPS64BE:#define __DBL_HAS_QUIET_NAN__ 1
// MIPS64BE:#define __DBL_MANT_DIG__ 53
// MIPS64BE:#define __DBL_MAX_10_EXP__ 308
// MIPS64BE:#define __DBL_MAX_EXP__ 1024
// MIPS64BE:#define __DBL_MAX__ 1.7976931348623157e+308
// MIPS64BE:#define __DBL_MIN_10_EXP__ (-307)
// MIPS64BE:#define __DBL_MIN_EXP__ (-1021)
// MIPS64BE:#define __DBL_MIN__ 2.2250738585072014e-308
// MIPS64BE:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// MIPS64BE:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// MIPS64BE:#define __FLT_DIG__ 6
// MIPS64BE:#define __FLT_EPSILON__ 1.19209290e-7F
// MIPS64BE:#define __FLT_EVAL_METHOD__ 0
// MIPS64BE:#define __FLT_HAS_DENORM__ 1
// MIPS64BE:#define __FLT_HAS_INFINITY__ 1
// MIPS64BE:#define __FLT_HAS_QUIET_NAN__ 1
// MIPS64BE:#define __FLT_MANT_DIG__ 24
// MIPS64BE:#define __FLT_MAX_10_EXP__ 38
// MIPS64BE:#define __FLT_MAX_EXP__ 128
// MIPS64BE:#define __FLT_MAX__ 3.40282347e+38F
// MIPS64BE:#define __FLT_MIN_10_EXP__ (-37)
// MIPS64BE:#define __FLT_MIN_EXP__ (-125)
// MIPS64BE:#define __FLT_MIN__ 1.17549435e-38F
// MIPS64BE:#define __FLT_RADIX__ 2
// MIPS64BE:#define __INT16_C_SUFFIX__
// MIPS64BE:#define __INT16_FMTd__ "hd"
// MIPS64BE:#define __INT16_FMTi__ "hi"
// MIPS64BE:#define __INT16_MAX__ 32767
// MIPS64BE:#define __INT16_TYPE__ short
// MIPS64BE:#define __INT32_C_SUFFIX__
// MIPS64BE:#define __INT32_FMTd__ "d"
// MIPS64BE:#define __INT32_FMTi__ "i"
// MIPS64BE:#define __INT32_MAX__ 2147483647
// MIPS64BE:#define __INT32_TYPE__ int
// MIPS64BE:#define __INT64_C_SUFFIX__ L
// MIPS64BE:#define __INT64_FMTd__ "ld"
// MIPS64BE:#define __INT64_FMTi__ "li"
// MIPS64BE:#define __INT64_MAX__ 9223372036854775807L
// MIPS64BE:#define __INT64_TYPE__ long int
// MIPS64BE:#define __INT8_C_SUFFIX__
// MIPS64BE:#define __INT8_FMTd__ "hhd"
// MIPS64BE:#define __INT8_FMTi__ "hhi"
// MIPS64BE:#define __INT8_MAX__ 127
// MIPS64BE:#define __INT8_TYPE__ signed char
// MIPS64BE:#define __INTMAX_C_SUFFIX__ L
// MIPS64BE:#define __INTMAX_FMTd__ "ld"
// MIPS64BE:#define __INTMAX_FMTi__ "li"
// MIPS64BE:#define __INTMAX_MAX__ 9223372036854775807L
// MIPS64BE:#define __INTMAX_TYPE__ long int
// MIPS64BE:#define __INTMAX_WIDTH__ 64
// MIPS64BE:#define __INTPTR_FMTd__ "ld"
// MIPS64BE:#define __INTPTR_FMTi__ "li"
// MIPS64BE:#define __INTPTR_MAX__ 9223372036854775807L
// MIPS64BE:#define __INTPTR_TYPE__ long int
// MIPS64BE:#define __INTPTR_WIDTH__ 64
// MIPS64BE:#define __INT_FAST16_FMTd__ "hd"
// MIPS64BE:#define __INT_FAST16_FMTi__ "hi"
// MIPS64BE:#define __INT_FAST16_MAX__ 32767
// MIPS64BE:#define __INT_FAST16_TYPE__ short
// MIPS64BE:#define __INT_FAST32_FMTd__ "d"
// MIPS64BE:#define __INT_FAST32_FMTi__ "i"
// MIPS64BE:#define __INT_FAST32_MAX__ 2147483647
// MIPS64BE:#define __INT_FAST32_TYPE__ int
// MIPS64BE:#define __INT_FAST64_FMTd__ "ld"
// MIPS64BE:#define __INT_FAST64_FMTi__ "li"
// MIPS64BE:#define __INT_FAST64_MAX__ 9223372036854775807L
// MIPS64BE:#define __INT_FAST64_TYPE__ long int
// MIPS64BE:#define __INT_FAST8_FMTd__ "hhd"
// MIPS64BE:#define __INT_FAST8_FMTi__ "hhi"
// MIPS64BE:#define __INT_FAST8_MAX__ 127
// MIPS64BE:#define __INT_FAST8_TYPE__ signed char
// MIPS64BE:#define __INT_LEAST16_FMTd__ "hd"
// MIPS64BE:#define __INT_LEAST16_FMTi__ "hi"
// MIPS64BE:#define __INT_LEAST16_MAX__ 32767
// MIPS64BE:#define __INT_LEAST16_TYPE__ short
// MIPS64BE:#define __INT_LEAST32_FMTd__ "d"
// MIPS64BE:#define __INT_LEAST32_FMTi__ "i"
// MIPS64BE:#define __INT_LEAST32_MAX__ 2147483647
// MIPS64BE:#define __INT_LEAST32_TYPE__ int
// MIPS64BE:#define __INT_LEAST64_FMTd__ "ld"
// MIPS64BE:#define __INT_LEAST64_FMTi__ "li"
// MIPS64BE:#define __INT_LEAST64_MAX__ 9223372036854775807L
// MIPS64BE:#define __INT_LEAST64_TYPE__ long int
// MIPS64BE:#define __INT_LEAST8_FMTd__ "hhd"
// MIPS64BE:#define __INT_LEAST8_FMTi__ "hhi"
// MIPS64BE:#define __INT_LEAST8_MAX__ 127
// MIPS64BE:#define __INT_LEAST8_TYPE__ signed char
// MIPS64BE:#define __INT_MAX__ 2147483647
// MIPS64BE:#define __LDBL_DENORM_MIN__ 6.47517511943802511092443895822764655e-4966L
// MIPS64BE:#define __LDBL_DIG__ 33
// MIPS64BE:#define __LDBL_EPSILON__ 1.92592994438723585305597794258492732e-34L
// MIPS64BE:#define __LDBL_HAS_DENORM__ 1
// MIPS64BE:#define __LDBL_HAS_INFINITY__ 1
// MIPS64BE:#define __LDBL_HAS_QUIET_NAN__ 1
// MIPS64BE:#define __LDBL_MANT_DIG__ 113
// MIPS64BE:#define __LDBL_MAX_10_EXP__ 4932
// MIPS64BE:#define __LDBL_MAX_EXP__ 16384
// MIPS64BE:#define __LDBL_MAX__ 1.18973149535723176508575932662800702e+4932L
// MIPS64BE:#define __LDBL_MIN_10_EXP__ (-4931)
// MIPS64BE:#define __LDBL_MIN_EXP__ (-16381)
// MIPS64BE:#define __LDBL_MIN__ 3.36210314311209350626267781732175260e-4932L
// MIPS64BE:#define __LONG_LONG_MAX__ 9223372036854775807LL
// MIPS64BE:#define __LONG_MAX__ 9223372036854775807L
// MIPS64BE:#define __LP64__ 1
// MIPS64BE:#define __MIPSEB 1
// MIPS64BE:#define __MIPSEB__ 1
// MIPS64BE:#define __POINTER_WIDTH__ 64
// MIPS64BE:#define __PRAGMA_REDEFINE_EXTNAME 1
// MIPS64BE:#define __PTRDIFF_TYPE__ long int
// MIPS64BE:#define __PTRDIFF_WIDTH__ 64
// MIPS64BE:#define __REGISTER_PREFIX__
// MIPS64BE:#define __SCHAR_MAX__ 127
// MIPS64BE:#define __SHRT_MAX__ 32767
// MIPS64BE:#define __SIG_ATOMIC_MAX__ 2147483647
// MIPS64BE:#define __SIG_ATOMIC_WIDTH__ 32
// MIPS64BE:#define __SIZEOF_DOUBLE__ 8
// MIPS64BE:#define __SIZEOF_FLOAT__ 4
// MIPS64BE:#define __SIZEOF_INT128__ 16
// MIPS64BE:#define __SIZEOF_INT__ 4
// MIPS64BE:#define __SIZEOF_LONG_DOUBLE__ 16
// MIPS64BE:#define __SIZEOF_LONG_LONG__ 8
// MIPS64BE:#define __SIZEOF_LONG__ 8
// MIPS64BE:#define __SIZEOF_POINTER__ 8
// MIPS64BE:#define __SIZEOF_PTRDIFF_T__ 8
// MIPS64BE:#define __SIZEOF_SHORT__ 2
// MIPS64BE:#define __SIZEOF_SIZE_T__ 8
// MIPS64BE:#define __SIZEOF_WCHAR_T__ 4
// MIPS64BE:#define __SIZEOF_WINT_T__ 4
// MIPS64BE:#define __SIZE_MAX__ 18446744073709551615UL
// MIPS64BE:#define __SIZE_TYPE__ long unsigned int
// MIPS64BE:#define __SIZE_WIDTH__ 64
// MIPS64BE-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 16UL
// MIPS64BE:#define __UINT16_C_SUFFIX__
// MIPS64BE:#define __UINT16_MAX__ 65535
// MIPS64BE:#define __UINT16_TYPE__ unsigned short
// MIPS64BE:#define __UINT32_C_SUFFIX__ U
// MIPS64BE:#define __UINT32_MAX__ 4294967295U
// MIPS64BE:#define __UINT32_TYPE__ unsigned int
// MIPS64BE:#define __UINT64_C_SUFFIX__ UL
// MIPS64BE:#define __UINT64_MAX__ 18446744073709551615UL
// MIPS64BE:#define __UINT64_TYPE__ long unsigned int
// MIPS64BE:#define __UINT8_C_SUFFIX__
// MIPS64BE:#define __UINT8_MAX__ 255
// MIPS64BE:#define __UINT8_TYPE__ unsigned char
// MIPS64BE:#define __UINTMAX_C_SUFFIX__ UL
// MIPS64BE:#define __UINTMAX_MAX__ 18446744073709551615UL
// MIPS64BE:#define __UINTMAX_TYPE__ long unsigned int
// MIPS64BE:#define __UINTMAX_WIDTH__ 64
// MIPS64BE:#define __UINTPTR_MAX__ 18446744073709551615UL
// MIPS64BE:#define __UINTPTR_TYPE__ long unsigned int
// MIPS64BE:#define __UINTPTR_WIDTH__ 64
// MIPS64BE:#define __UINT_FAST16_MAX__ 65535
// MIPS64BE:#define __UINT_FAST16_TYPE__ unsigned short
// MIPS64BE:#define __UINT_FAST32_MAX__ 4294967295U
// MIPS64BE:#define __UINT_FAST32_TYPE__ unsigned int
// MIPS64BE:#define __UINT_FAST64_MAX__ 18446744073709551615UL
// MIPS64BE:#define __UINT_FAST64_TYPE__ long unsigned int
// MIPS64BE:#define __UINT_FAST8_MAX__ 255
// MIPS64BE:#define __UINT_FAST8_TYPE__ unsigned char
// MIPS64BE:#define __UINT_LEAST16_MAX__ 65535
// MIPS64BE:#define __UINT_LEAST16_TYPE__ unsigned short
// MIPS64BE:#define __UINT_LEAST32_MAX__ 4294967295U
// MIPS64BE:#define __UINT_LEAST32_TYPE__ unsigned int
// MIPS64BE:#define __UINT_LEAST64_MAX__ 18446744073709551615UL
// MIPS64BE:#define __UINT_LEAST64_TYPE__ long unsigned int
// MIPS64BE:#define __UINT_LEAST8_MAX__ 255
// MIPS64BE:#define __UINT_LEAST8_TYPE__ unsigned char
// MIPS64BE:#define __USER_LABEL_PREFIX__
// MIPS64BE:#define __WCHAR_MAX__ 2147483647
// MIPS64BE:#define __WCHAR_TYPE__ int
// MIPS64BE:#define __WCHAR_WIDTH__ 32
// MIPS64BE:#define __WINT_TYPE__ int
// MIPS64BE:#define __WINT_WIDTH__ 32
// MIPS64BE:#define __clang__ 1
// MIPS64BE:#define __llvm__ 1
// MIPS64BE:#define __mips 64
// MIPS64BE:#define __mips64 1
// MIPS64BE:#define __mips64__ 1
// MIPS64BE:#define __mips__ 1
// MIPS64BE:#define __mips_abicalls 1
// MIPS64BE:#define __mips_fpr 64
// MIPS64BE:#define __mips_hard_float 1
// MIPS64BE:#define __mips_n64 1
// MIPS64BE:#define _mips 1
// MIPS64BE:#define mips 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips64el-none-none < /dev/null | FileCheck -match-full-lines -check-prefix MIPS64EL %s
//
// MIPS64EL:#define MIPSEL 1
// MIPS64EL:#define _ABI64 3
// MIPS64EL:#define _LP64 1
// MIPS64EL:#define _MIPSEL 1
// MIPS64EL:#define _MIPS_ARCH "mips64r2"
// MIPS64EL:#define _MIPS_ARCH_MIPS64R2 1
// MIPS64EL:#define _MIPS_FPSET 32
// MIPS64EL:#define _MIPS_SIM _ABI64
// MIPS64EL:#define _MIPS_SZINT 32
// MIPS64EL:#define _MIPS_SZLONG 64
// MIPS64EL:#define _MIPS_SZPTR 64
// MIPS64EL:#define __BIGGEST_ALIGNMENT__ 16
// MIPS64EL:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// MIPS64EL:#define __CHAR16_TYPE__ unsigned short
// MIPS64EL:#define __CHAR32_TYPE__ unsigned int
// MIPS64EL:#define __CHAR_BIT__ 8
// MIPS64EL:#define __CONSTANT_CFSTRINGS__ 1
// MIPS64EL:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// MIPS64EL:#define __DBL_DIG__ 15
// MIPS64EL:#define __DBL_EPSILON__ 2.2204460492503131e-16
// MIPS64EL:#define __DBL_HAS_DENORM__ 1
// MIPS64EL:#define __DBL_HAS_INFINITY__ 1
// MIPS64EL:#define __DBL_HAS_QUIET_NAN__ 1
// MIPS64EL:#define __DBL_MANT_DIG__ 53
// MIPS64EL:#define __DBL_MAX_10_EXP__ 308
// MIPS64EL:#define __DBL_MAX_EXP__ 1024
// MIPS64EL:#define __DBL_MAX__ 1.7976931348623157e+308
// MIPS64EL:#define __DBL_MIN_10_EXP__ (-307)
// MIPS64EL:#define __DBL_MIN_EXP__ (-1021)
// MIPS64EL:#define __DBL_MIN__ 2.2250738585072014e-308
// MIPS64EL:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// MIPS64EL:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// MIPS64EL:#define __FLT_DIG__ 6
// MIPS64EL:#define __FLT_EPSILON__ 1.19209290e-7F
// MIPS64EL:#define __FLT_EVAL_METHOD__ 0
// MIPS64EL:#define __FLT_HAS_DENORM__ 1
// MIPS64EL:#define __FLT_HAS_INFINITY__ 1
// MIPS64EL:#define __FLT_HAS_QUIET_NAN__ 1
// MIPS64EL:#define __FLT_MANT_DIG__ 24
// MIPS64EL:#define __FLT_MAX_10_EXP__ 38
// MIPS64EL:#define __FLT_MAX_EXP__ 128
// MIPS64EL:#define __FLT_MAX__ 3.40282347e+38F
// MIPS64EL:#define __FLT_MIN_10_EXP__ (-37)
// MIPS64EL:#define __FLT_MIN_EXP__ (-125)
// MIPS64EL:#define __FLT_MIN__ 1.17549435e-38F
// MIPS64EL:#define __FLT_RADIX__ 2
// MIPS64EL:#define __INT16_C_SUFFIX__
// MIPS64EL:#define __INT16_FMTd__ "hd"
// MIPS64EL:#define __INT16_FMTi__ "hi"
// MIPS64EL:#define __INT16_MAX__ 32767
// MIPS64EL:#define __INT16_TYPE__ short
// MIPS64EL:#define __INT32_C_SUFFIX__
// MIPS64EL:#define __INT32_FMTd__ "d"
// MIPS64EL:#define __INT32_FMTi__ "i"
// MIPS64EL:#define __INT32_MAX__ 2147483647
// MIPS64EL:#define __INT32_TYPE__ int
// MIPS64EL:#define __INT64_C_SUFFIX__ L
// MIPS64EL:#define __INT64_FMTd__ "ld"
// MIPS64EL:#define __INT64_FMTi__ "li"
// MIPS64EL:#define __INT64_MAX__ 9223372036854775807L
// MIPS64EL:#define __INT64_TYPE__ long int
// MIPS64EL:#define __INT8_C_SUFFIX__
// MIPS64EL:#define __INT8_FMTd__ "hhd"
// MIPS64EL:#define __INT8_FMTi__ "hhi"
// MIPS64EL:#define __INT8_MAX__ 127
// MIPS64EL:#define __INT8_TYPE__ signed char
// MIPS64EL:#define __INTMAX_C_SUFFIX__ L
// MIPS64EL:#define __INTMAX_FMTd__ "ld"
// MIPS64EL:#define __INTMAX_FMTi__ "li"
// MIPS64EL:#define __INTMAX_MAX__ 9223372036854775807L
// MIPS64EL:#define __INTMAX_TYPE__ long int
// MIPS64EL:#define __INTMAX_WIDTH__ 64
// MIPS64EL:#define __INTPTR_FMTd__ "ld"
// MIPS64EL:#define __INTPTR_FMTi__ "li"
// MIPS64EL:#define __INTPTR_MAX__ 9223372036854775807L
// MIPS64EL:#define __INTPTR_TYPE__ long int
// MIPS64EL:#define __INTPTR_WIDTH__ 64
// MIPS64EL:#define __INT_FAST16_FMTd__ "hd"
// MIPS64EL:#define __INT_FAST16_FMTi__ "hi"
// MIPS64EL:#define __INT_FAST16_MAX__ 32767
// MIPS64EL:#define __INT_FAST16_TYPE__ short
// MIPS64EL:#define __INT_FAST32_FMTd__ "d"
// MIPS64EL:#define __INT_FAST32_FMTi__ "i"
// MIPS64EL:#define __INT_FAST32_MAX__ 2147483647
// MIPS64EL:#define __INT_FAST32_TYPE__ int
// MIPS64EL:#define __INT_FAST64_FMTd__ "ld"
// MIPS64EL:#define __INT_FAST64_FMTi__ "li"
// MIPS64EL:#define __INT_FAST64_MAX__ 9223372036854775807L
// MIPS64EL:#define __INT_FAST64_TYPE__ long int
// MIPS64EL:#define __INT_FAST8_FMTd__ "hhd"
// MIPS64EL:#define __INT_FAST8_FMTi__ "hhi"
// MIPS64EL:#define __INT_FAST8_MAX__ 127
// MIPS64EL:#define __INT_FAST8_TYPE__ signed char
// MIPS64EL:#define __INT_LEAST16_FMTd__ "hd"
// MIPS64EL:#define __INT_LEAST16_FMTi__ "hi"
// MIPS64EL:#define __INT_LEAST16_MAX__ 32767
// MIPS64EL:#define __INT_LEAST16_TYPE__ short
// MIPS64EL:#define __INT_LEAST32_FMTd__ "d"
// MIPS64EL:#define __INT_LEAST32_FMTi__ "i"
// MIPS64EL:#define __INT_LEAST32_MAX__ 2147483647
// MIPS64EL:#define __INT_LEAST32_TYPE__ int
// MIPS64EL:#define __INT_LEAST64_FMTd__ "ld"
// MIPS64EL:#define __INT_LEAST64_FMTi__ "li"
// MIPS64EL:#define __INT_LEAST64_MAX__ 9223372036854775807L
// MIPS64EL:#define __INT_LEAST64_TYPE__ long int
// MIPS64EL:#define __INT_LEAST8_FMTd__ "hhd"
// MIPS64EL:#define __INT_LEAST8_FMTi__ "hhi"
// MIPS64EL:#define __INT_LEAST8_MAX__ 127
// MIPS64EL:#define __INT_LEAST8_TYPE__ signed char
// MIPS64EL:#define __INT_MAX__ 2147483647
// MIPS64EL:#define __LDBL_DENORM_MIN__ 6.47517511943802511092443895822764655e-4966L
// MIPS64EL:#define __LDBL_DIG__ 33
// MIPS64EL:#define __LDBL_EPSILON__ 1.92592994438723585305597794258492732e-34L
// MIPS64EL:#define __LDBL_HAS_DENORM__ 1
// MIPS64EL:#define __LDBL_HAS_INFINITY__ 1
// MIPS64EL:#define __LDBL_HAS_QUIET_NAN__ 1
// MIPS64EL:#define __LDBL_MANT_DIG__ 113
// MIPS64EL:#define __LDBL_MAX_10_EXP__ 4932
// MIPS64EL:#define __LDBL_MAX_EXP__ 16384
// MIPS64EL:#define __LDBL_MAX__ 1.18973149535723176508575932662800702e+4932L
// MIPS64EL:#define __LDBL_MIN_10_EXP__ (-4931)
// MIPS64EL:#define __LDBL_MIN_EXP__ (-16381)
// MIPS64EL:#define __LDBL_MIN__ 3.36210314311209350626267781732175260e-4932L
// MIPS64EL:#define __LITTLE_ENDIAN__ 1
// MIPS64EL:#define __LONG_LONG_MAX__ 9223372036854775807LL
// MIPS64EL:#define __LONG_MAX__ 9223372036854775807L
// MIPS64EL:#define __LP64__ 1
// MIPS64EL:#define __MIPSEL 1
// MIPS64EL:#define __MIPSEL__ 1
// MIPS64EL:#define __POINTER_WIDTH__ 64
// MIPS64EL:#define __PRAGMA_REDEFINE_EXTNAME 1
// MIPS64EL:#define __PTRDIFF_TYPE__ long int
// MIPS64EL:#define __PTRDIFF_WIDTH__ 64
// MIPS64EL:#define __REGISTER_PREFIX__
// MIPS64EL:#define __SCHAR_MAX__ 127
// MIPS64EL:#define __SHRT_MAX__ 32767
// MIPS64EL:#define __SIG_ATOMIC_MAX__ 2147483647
// MIPS64EL:#define __SIG_ATOMIC_WIDTH__ 32
// MIPS64EL:#define __SIZEOF_DOUBLE__ 8
// MIPS64EL:#define __SIZEOF_FLOAT__ 4
// MIPS64EL:#define __SIZEOF_INT128__ 16
// MIPS64EL:#define __SIZEOF_INT__ 4
// MIPS64EL:#define __SIZEOF_LONG_DOUBLE__ 16
// MIPS64EL:#define __SIZEOF_LONG_LONG__ 8
// MIPS64EL:#define __SIZEOF_LONG__ 8
// MIPS64EL:#define __SIZEOF_POINTER__ 8
// MIPS64EL:#define __SIZEOF_PTRDIFF_T__ 8
// MIPS64EL:#define __SIZEOF_SHORT__ 2
// MIPS64EL:#define __SIZEOF_SIZE_T__ 8
// MIPS64EL:#define __SIZEOF_WCHAR_T__ 4
// MIPS64EL:#define __SIZEOF_WINT_T__ 4
// MIPS64EL:#define __SIZE_MAX__ 18446744073709551615UL
// MIPS64EL:#define __SIZE_TYPE__ long unsigned int
// MIPS64EL:#define __SIZE_WIDTH__ 64
// MIPS64EL:#define __UINT16_C_SUFFIX__
// MIPS64EL:#define __UINT16_MAX__ 65535
// MIPS64EL:#define __UINT16_TYPE__ unsigned short
// MIPS64EL:#define __UINT32_C_SUFFIX__ U
// MIPS64EL:#define __UINT32_MAX__ 4294967295U
// MIPS64EL:#define __UINT32_TYPE__ unsigned int
// MIPS64EL:#define __UINT64_C_SUFFIX__ UL
// MIPS64EL:#define __UINT64_MAX__ 18446744073709551615UL
// MIPS64EL:#define __UINT64_TYPE__ long unsigned int
// MIPS64EL:#define __UINT8_C_SUFFIX__
// MIPS64EL:#define __UINT8_MAX__ 255
// MIPS64EL:#define __UINT8_TYPE__ unsigned char
// MIPS64EL:#define __UINTMAX_C_SUFFIX__ UL
// MIPS64EL:#define __UINTMAX_MAX__ 18446744073709551615UL
// MIPS64EL:#define __UINTMAX_TYPE__ long unsigned int
// MIPS64EL:#define __UINTMAX_WIDTH__ 64
// MIPS64EL:#define __UINTPTR_MAX__ 18446744073709551615UL
// MIPS64EL:#define __UINTPTR_TYPE__ long unsigned int
// MIPS64EL:#define __UINTPTR_WIDTH__ 64
// MIPS64EL:#define __UINT_FAST16_MAX__ 65535
// MIPS64EL:#define __UINT_FAST16_TYPE__ unsigned short
// MIPS64EL:#define __UINT_FAST32_MAX__ 4294967295U
// MIPS64EL:#define __UINT_FAST32_TYPE__ unsigned int
// MIPS64EL:#define __UINT_FAST64_MAX__ 18446744073709551615UL
// MIPS64EL:#define __UINT_FAST64_TYPE__ long unsigned int
// MIPS64EL:#define __UINT_FAST8_MAX__ 255
// MIPS64EL:#define __UINT_FAST8_TYPE__ unsigned char
// MIPS64EL:#define __UINT_LEAST16_MAX__ 65535
// MIPS64EL:#define __UINT_LEAST16_TYPE__ unsigned short
// MIPS64EL:#define __UINT_LEAST32_MAX__ 4294967295U
// MIPS64EL:#define __UINT_LEAST32_TYPE__ unsigned int
// MIPS64EL:#define __UINT_LEAST64_MAX__ 18446744073709551615UL
// MIPS64EL:#define __UINT_LEAST64_TYPE__ long unsigned int
// MIPS64EL:#define __UINT_LEAST8_MAX__ 255
// MIPS64EL:#define __UINT_LEAST8_TYPE__ unsigned char
// MIPS64EL:#define __USER_LABEL_PREFIX__
// MIPS64EL:#define __WCHAR_MAX__ 2147483647
// MIPS64EL:#define __WCHAR_TYPE__ int
// MIPS64EL:#define __WCHAR_WIDTH__ 32
// MIPS64EL:#define __WINT_TYPE__ int
// MIPS64EL:#define __WINT_WIDTH__ 32
// MIPS64EL:#define __clang__ 1
// MIPS64EL:#define __llvm__ 1
// MIPS64EL:#define __mips 64
// MIPS64EL:#define __mips64 1
// MIPS64EL:#define __mips64__ 1
// MIPS64EL:#define __mips__ 1
// MIPS64EL:#define __mips_abicalls 1
// MIPS64EL:#define __mips_fpr 64
// MIPS64EL:#define __mips_hard_float 1
// MIPS64EL:#define __mips_n64 1
// MIPS64EL:#define _mips 1
// MIPS64EL:#define mips 1

// Check MIPS arch and isa macros
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips-none-none \
// RUN:            < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-DEF32 %s
//
// MIPS-ARCH-DEF32:#define _MIPS_ARCH "mips32r2"
// MIPS-ARCH-DEF32:#define _MIPS_ARCH_MIPS32R2 1
// MIPS-ARCH-DEF32:#define _MIPS_ISA _MIPS_ISA_MIPS32
// MIPS-ARCH-DEF32:#define __mips_isa_rev 2

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips-none-nones \
// RUN:            -target-cpu mips32 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-32 %s
//
// MIPS-ARCH-32:#define _MIPS_ARCH "mips32"
// MIPS-ARCH-32:#define _MIPS_ARCH_MIPS32 1
// MIPS-ARCH-32:#define _MIPS_ISA _MIPS_ISA_MIPS32
// MIPS-ARCH-32:#define __mips_isa_rev 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips-none-none \
// RUN:            -target-cpu mips32r2 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-32R2 %s
//
// MIPS-ARCH-32R2:#define _MIPS_ARCH "mips32r2"
// MIPS-ARCH-32R2:#define _MIPS_ARCH_MIPS32R2 1
// MIPS-ARCH-32R2:#define _MIPS_ISA _MIPS_ISA_MIPS32
// MIPS-ARCH-32R2:#define __mips_isa_rev 2

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips-none-none \
// RUN:            -target-cpu mips32r3 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-32R3 %s
//
// MIPS-ARCH-32R3:#define _MIPS_ARCH "mips32r3"
// MIPS-ARCH-32R3:#define _MIPS_ARCH_MIPS32R3 1
// MIPS-ARCH-32R3:#define _MIPS_ISA _MIPS_ISA_MIPS32
// MIPS-ARCH-32R3:#define __mips_isa_rev 3

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips-none-none \
// RUN:            -target-cpu mips32r5 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-32R5 %s
//
// MIPS-ARCH-32R5:#define _MIPS_ARCH "mips32r5"
// MIPS-ARCH-32R5:#define _MIPS_ARCH_MIPS32R5 1
// MIPS-ARCH-32R5:#define _MIPS_ISA _MIPS_ISA_MIPS32
// MIPS-ARCH-32R5:#define __mips_isa_rev 5

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips-none-none \
// RUN:            -target-cpu mips32r6 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-32R6 %s
//
// MIPS-ARCH-32R6:#define _MIPS_ARCH "mips32r6"
// MIPS-ARCH-32R6:#define _MIPS_ARCH_MIPS32R6 1
// MIPS-ARCH-32R6:#define _MIPS_ISA _MIPS_ISA_MIPS32
// MIPS-ARCH-32R6:#define __mips_isa_rev 6

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips64-none-none \
// RUN:            < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-DEF64 %s
//
// MIPS-ARCH-DEF64:#define _MIPS_ARCH "mips64r2"
// MIPS-ARCH-DEF64:#define _MIPS_ARCH_MIPS64R2 1
// MIPS-ARCH-DEF64:#define _MIPS_ISA _MIPS_ISA_MIPS64
// MIPS-ARCH-DEF64:#define __mips_isa_rev 2

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips64-none-none \
// RUN:            -target-cpu mips64 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-64 %s
//
// MIPS-ARCH-64:#define _MIPS_ARCH "mips64"
// MIPS-ARCH-64:#define _MIPS_ARCH_MIPS64 1
// MIPS-ARCH-64:#define _MIPS_ISA _MIPS_ISA_MIPS64
// MIPS-ARCH-64:#define __mips_isa_rev 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips64-none-none \
// RUN:            -target-cpu mips64r2 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-64R2 %s
//
// MIPS-ARCH-64R2:#define _MIPS_ARCH "mips64r2"
// MIPS-ARCH-64R2:#define _MIPS_ARCH_MIPS64R2 1
// MIPS-ARCH-64R2:#define _MIPS_ISA _MIPS_ISA_MIPS64
// MIPS-ARCH-64R2:#define __mips_isa_rev 2

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips64-none-none \
// RUN:            -target-cpu mips64r3 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-64R3 %s
//
// MIPS-ARCH-64R3:#define _MIPS_ARCH "mips64r3"
// MIPS-ARCH-64R3:#define _MIPS_ARCH_MIPS64R3 1
// MIPS-ARCH-64R3:#define _MIPS_ISA _MIPS_ISA_MIPS64
// MIPS-ARCH-64R3:#define __mips_isa_rev 3

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips64-none-none \
// RUN:            -target-cpu mips64r5 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-64R5 %s
//
// MIPS-ARCH-64R5:#define _MIPS_ARCH "mips64r5"
// MIPS-ARCH-64R5:#define _MIPS_ARCH_MIPS64R5 1
// MIPS-ARCH-64R5:#define _MIPS_ISA _MIPS_ISA_MIPS64
// MIPS-ARCH-64R5:#define __mips_isa_rev 5

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips64-none-none \
// RUN:            -target-cpu mips64r6 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-64R6 %s
//
// MIPS-ARCH-64R6:#define _MIPS_ARCH "mips64r6"
// MIPS-ARCH-64R6:#define _MIPS_ARCH_MIPS64R6 1
// MIPS-ARCH-64R6:#define _MIPS_ISA _MIPS_ISA_MIPS64
// MIPS-ARCH-64R6:#define __mips_isa_rev 6

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips64-none-none \
// RUN:            -target-cpu octeon < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-OCTEON %s
//
// MIPS-ARCH-OCTEON:#define _MIPS_ARCH "octeon"
// MIPS-ARCH-OCTEON:#define _MIPS_ARCH_OCTEON 1
// MIPS-ARCH-OCTEON:#define _MIPS_ISA _MIPS_ISA_MIPS64
// MIPS-ARCH-OCTEON:#define __OCTEON__ 1
// MIPS-ARCH-OCTEON:#define __mips_isa_rev 2

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips64-none-none \
// RUN:            -target-cpu octeon+ < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-OCTEONP %s
//
// MIPS-ARCH-OCTEONP:#define _MIPS_ARCH "octeon+"
// MIPS-ARCH-OCTEONP:#define _MIPS_ARCH_OCTEONP 1
// MIPS-ARCH-OCTEONP:#define _MIPS_ISA _MIPS_ISA_MIPS64
// MIPS-ARCH-OCTEONP:#define __OCTEON__ 1
// MIPS-ARCH-OCTEONP:#define __mips_isa_rev 2

// Check MIPS float ABI macros
//
// RUN: %clang_cc1 -E -dM -ffreestanding \
// RUN:   -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-FABI-HARD %s
// MIPS-FABI-HARD:#define __mips_hard_float 1

// RUN: %clang_cc1 -target-feature +soft-float -E -dM -ffreestanding \
// RUN:   -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-FABI-SOFT %s
// MIPS-FABI-SOFT:#define __mips_soft_float 1

// RUN: %clang_cc1 -target-feature +single-float -E -dM -ffreestanding \
// RUN:   -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-FABI-SINGLE %s
// MIPS-FABI-SINGLE:#define __mips_hard_float 1
// MIPS-FABI-SINGLE:#define __mips_single_float 1

// RUN: %clang_cc1 -target-feature +soft-float -target-feature +single-float \
// RUN:   -E -dM -ffreestanding -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-FABI-SINGLE-SOFT %s
// MIPS-FABI-SINGLE-SOFT:#define __mips_single_float 1
// MIPS-FABI-SINGLE-SOFT:#define __mips_soft_float 1

// Check MIPS features macros
//
// RUN: %clang_cc1 -target-feature +mips16 \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS16 %s
// MIPS16:#define __mips16 1

// RUN: %clang_cc1 -target-feature -mips16 \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix NOMIPS16 %s
// NOMIPS16-NOT:#define __mips16 1

// RUN: %clang_cc1 -target-feature +micromips \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MICROMIPS %s
// MICROMIPS:#define __mips_micromips 1

// RUN: %clang_cc1 -target-feature -micromips \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix NOMICROMIPS %s
// NOMICROMIPS-NOT:#define __mips_micromips 1

// RUN: %clang_cc1 -target-feature +dsp \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-DSP %s
// MIPS-DSP:#define __mips_dsp 1
// MIPS-DSP:#define __mips_dsp_rev 1
// MIPS-DSP-NOT:#define __mips_dspr2 1

// RUN: %clang_cc1 -target-feature +dspr2 \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-DSPR2 %s
// MIPS-DSPR2:#define __mips_dsp 1
// MIPS-DSPR2:#define __mips_dsp_rev 2
// MIPS-DSPR2:#define __mips_dspr2 1

// RUN: %clang_cc1 -target-feature +msa \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-MSA %s
// MIPS-MSA:#define __mips_msa 1

// RUN: %clang_cc1 -target-feature +nomadd4 \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-NOMADD4 %s
// MIPS-NOMADD4:#define __mips_no_madd4 1

// RUN: %clang_cc1 \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-MADD4 %s
// MIPS-MADD4-NOT:#define __mips_no_madd4 1

// RUN: %clang_cc1 -target-cpu mips32r3 -target-feature +nan2008 \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-NAN2008 %s
// MIPS-NAN2008:#define __mips_nan2008 1

// RUN: %clang_cc1 -target-cpu mips32r3 -target-feature -nan2008 \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix NOMIPS-NAN2008 %s
// NOMIPS-NAN2008-NOT:#define __mips_nan2008 1

// RUN: %clang_cc1 -target-cpu mips32r3 -target-feature +abs2008 \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ABS2008 %s
// MIPS-ABS2008:#define __mips_abs2008 1

// RUN: %clang_cc1 -target-cpu mips32r3 -target-feature -abs2008 \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix NOMIPS-ABS2008 %s
// NOMIPS-ABS2008-NOT:#define __mips_abs2008 1

// RUN: %clang_cc1  \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS32-NOFP %s
// MIPS32-NOFP:#define __mips_fpr 0

// RUN: %clang_cc1 -target-feature +fpxx \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS32-MFPXX %s
// MIPS32-MFPXX:#define __mips_fpr 0

// RUN: %clang_cc1 -target-cpu mips32r6 -target-feature +fpxx \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS32R6-MFPXX %s
// MIPS32R6-MFPXX:#define __mips_fpr 0

// RUN: %clang_cc1  \
// RUN:   -E -dM -triple=mips64-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS64-NOFP %s
// MIPS64-NOFP:#define __mips_fpr 64

// RUN: not %clang_cc1 -target-feature -fp64 \
// RUN:   -E -dM -triple=mips64-none-none < /dev/null 2>&1 \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS64-MFP32 %s
// MIPS64-MFP32:error: option '-mfpxx' cannot be specified with 'mips64r2'

// RUN: not %clang_cc1 -target-feature +fpxx \
// RUN:   -E -dM -triple=mips64-none-none < /dev/null 2>&1 \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS64-MFPXX %s
// MIPS64-MFPXX:error: '-mfpxx' can only be used with the 'o32' ABI

// RUN: not %clang_cc1 -target-cpu mips64r6 -target-feature +fpxx \
// RUN:   -E -dM -triple=mips64-none-none < /dev/null 2>&1 \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS64R6-MFPXX %s
// MIPS64R6-MFPXX:error: '-mfpxx' can only be used with the 'o32' ABI

// RUN: %clang_cc1 -target-feature -fp64 \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS32-MFP32 %s
// MIPS32-MFP32:#define _MIPS_FPSET 16
// MIPS32-MFP32:#define __mips_fpr 32

// RUN: %clang_cc1 -target-feature +fp64 \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS32-MFP64 %s
// MIPS32-MFP64:#define _MIPS_FPSET 32
// MIPS32-MFP64:#define __mips_fpr 64
//
// RUN: %clang_cc1 -target-feature +single-float \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS32-MFP32SF %s
// MIPS32-MFP32SF:#define _MIPS_FPSET 32
// MIPS32-MFP32SF:#define __mips_fpr 0

// RUN: %clang_cc1 -target-feature +fp64 \
// RUN:   -E -dM -triple=mips64-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS64-MFP64 %s
// MIPS64-MFP64:#define _MIPS_FPSET 32
// MIPS64-MFP64:#define __mips_fpr 64

// RUN: %clang_cc1 -target-feature -fp64 -target-feature +single-float \
// RUN:   -E -dM -triple=mips64-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS64-NOMFP64 %s
// MIPS64-NOMFP64:#define _MIPS_FPSET 32
// MIPS64-NOMFP64:#define __mips_fpr 32

// RUN: %clang_cc1 -target-cpu mips32r6 \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-XXR6 %s
// RUN: %clang_cc1 -target-cpu mips64r6 \
// RUN:   -E -dM -triple=mips64-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-XXR6 %s
// MIPS-XXR6:#define _MIPS_FPSET 32
// MIPS-XXR6:#define __mips_fpr 64
// MIPS-XXR6:#define __mips_nan2008 1

// RUN: %clang_cc1 -target-cpu mips32 \
// RUN:   -E -dM -triple=mips-unknown-netbsd -mrelocation-model pic < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ABICALLS-NETBSD %s
// MIPS-ABICALLS-NETBSD-NOT: #define __ABICALLS__ 1
// MIPS-ABICALLS-NETBSD: #define __mips_abicalls 1

// RUN: %clang_cc1 -target-cpu mips64 \
// RUN:   -E -dM -triple=mips64-unknown-netbsd -mrelocation-model pic < \
// RUN:   /dev/null | FileCheck -match-full-lines \
// RUN:   -check-prefix MIPS-ABICALLS-NETBSD64 %s
// MIPS-ABICALLS-NETBSD64-NOT: #define __ABICALLS__ 1
// MIPS-ABICALLS-NETBSD64: #define __mips_abicalls 1

// RUN: %clang_cc1 -target-cpu mips32 \
// RUN:   -E -dM -triple=mips-unknown-freebsd -mrelocation-model pic < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ABICALLS-FREEBSD %s
// MIPS-ABICALLS-FREEBSD: #define __ABICALLS__ 1
// MIPS-ABICALLS-FREEBSD: #define __mips_abicalls 1

// RUN: %clang_cc1 -target-cpu mips64 \
// RUN:   -E -dM -triple=mips64-unknown-freebsd -mrelocation-model pic < \
// RUN:   /dev/null | FileCheck -match-full-lines \
// RUN:   -check-prefix MIPS-ABICALLS-FREEBSD64 %s
// MIPS-ABICALLS-FREEBSD64: #define __ABICALLS__ 1
// MIPS-ABICALLS-FREEBSD64: #define __mips_abicalls 1

// RUN: %clang_cc1 -target-cpu mips32 \
// RUN:   -E -dM -triple=mips-unknown-openbsd -mrelocation-model pic < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ABICALLS-OPENBSD %s
// MIPS-ABICALLS-OPENBSD: #define __ABICALLS__ 1
// MIPS-ABICALLS-OPENBSD: #define __mips_abicalls 1

// RUN: %clang_cc1 -target-cpu mips64 \
// RUN:   -E -dM -triple=mips64-unknown-openbsd -mrelocation-model pic < \
// RUN:   /dev/null | FileCheck -match-full-lines \
// RUN:   -check-prefix MIPS-ABICALLS-OPENBSD64 %s
// MIPS-ABICALLS-OPENBSD64: #define __ABICALLS__ 1
// MIPS-ABICALLS-OPENBSD64: #define __mips_abicalls 1
