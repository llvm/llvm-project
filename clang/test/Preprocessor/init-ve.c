/// Check predefinitions for NEC Aurora VE
/// REQUIRES: ve-registered-target

// RUN: %clang_cc1 -E -dM -triple=ve < /dev/null | \
// RUN:     FileCheck -match-full-lines -check-prefix VE %s
// RUN: %clang_cc1 -x c++ -E -dM -triple=ve < /dev/null | \
// RUN:     FileCheck -match-full-lines -check-prefix VE -check-prefix VE-CXX %s
//
// VE:#define _LP64 1
// VE:#define __BIGGEST_ALIGNMENT__ 8
// VE:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// VE:#define __CHAR16_TYPE__ unsigned short
// VE:#define __CHAR32_TYPE__ unsigned int
// VE:#define __CHAR_BIT__ 8
// VE:#define __DBL_DECIMAL_DIG__ 17
// VE:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// VE:#define __DBL_DIG__ 15
// VE:#define __DBL_EPSILON__ 2.2204460492503131e-16
// VE:#define __DBL_HAS_DENORM__ 1
// VE:#define __DBL_HAS_INFINITY__ 1
// VE:#define __DBL_HAS_QUIET_NAN__ 1
// VE:#define __DBL_MANT_DIG__ 53
// VE:#define __DBL_MAX_10_EXP__ 308
// VE:#define __DBL_MAX_EXP__ 1024
// VE:#define __DBL_MAX__ 1.7976931348623157e+308
// VE:#define __DBL_MIN_10_EXP__ (-307)
// VE:#define __DBL_MIN_EXP__ (-1021)
// VE:#define __DBL_MIN__ 2.2250738585072014e-308
// VE:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// VE-NOT:#define __FAST_MATH__ 1
// VE:#define __FLT_DECIMAL_DIG__ 9
// VE:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// VE:#define __FLT_DIG__ 6
// VE:#define __FLT_EPSILON__ 1.19209290e-7F
// VE:#define __FLT_HAS_DENORM__ 1
// VE:#define __FLT_HAS_INFINITY__ 1
// VE:#define __FLT_HAS_QUIET_NAN__ 1
// VE:#define __FLT_MANT_DIG__ 24
// VE:#define __FLT_MAX_10_EXP__ 38
// VE:#define __FLT_MAX_EXP__ 128
// VE:#define __FLT_MAX__ 3.40282347e+38F
// VE:#define __FLT_MIN_10_EXP__ (-37)
// VE:#define __FLT_MIN_EXP__ (-125)
// VE:#define __FLT_MIN__ 1.17549435e-38F
// VE:#define __FLT_RADIX__ 2
// VE:#define __INT16_C_SUFFIX__
// VE:#define __INT16_FMTd__ "hd"
// VE:#define __INT16_FMTi__ "hi"
// VE:#define __INT16_MAX__ 32767
// VE:#define __INT16_TYPE__ short
// VE:#define __INT32_C_SUFFIX__
// VE:#define __INT32_FMTd__ "d"
// VE:#define __INT32_FMTi__ "i"
// VE:#define __INT32_MAX__ 2147483647
// VE:#define __INT32_TYPE__ int
// VE:#define __INT64_C_SUFFIX__ L
// VE:#define __INT64_FMTd__ "ld"
// VE:#define __INT64_FMTi__ "li"
// VE:#define __INT64_MAX__ 9223372036854775807L
// VE:#define __INT64_TYPE__ long int
// VE:#define __INT8_C_SUFFIX__
// VE:#define __INT8_FMTd__ "hhd"
// VE:#define __INT8_FMTi__ "hhi"
// VE:#define __INT8_MAX__ 127
// VE:#define __INT8_TYPE__ signed char
// VE:#define __INTMAX_C_SUFFIX__ L
// VE:#define __INTMAX_FMTd__ "ld"
// VE:#define __INTMAX_FMTi__ "li"
// VE:#define __INTMAX_MAX__ 9223372036854775807L
// VE:#define __INTMAX_TYPE__ long int
// VE:#define __INTMAX_WIDTH__ 64
// VE:#define __INTPTR_FMTd__ "ld"
// VE:#define __INTPTR_FMTi__ "li"
// VE:#define __INTPTR_MAX__ 9223372036854775807L
// VE:#define __INTPTR_TYPE__ long int
// VE:#define __INTPTR_WIDTH__ 64
// VE:#define __INT_FAST16_FMTd__ "hd"
// VE:#define __INT_FAST16_FMTi__ "hi"
// VE:#define __INT_FAST16_MAX__ 32767
// VE:#define __INT_FAST16_TYPE__ short
// VE:#define __INT_FAST32_FMTd__ "d"
// VE:#define __INT_FAST32_FMTi__ "i"
// VE:#define __INT_FAST32_MAX__ 2147483647
// VE:#define __INT_FAST32_TYPE__ int
// VE:#define __INT_FAST64_FMTd__ "ld"
// VE:#define __INT_FAST64_FMTi__ "li"
// VE:#define __INT_FAST64_MAX__ 9223372036854775807L
// VE:#define __INT_FAST64_TYPE__ long int
// VE:#define __INT_FAST8_FMTd__ "hhd"
// VE:#define __INT_FAST8_FMTi__ "hhi"
// VE:#define __INT_FAST8_MAX__ 127
// VE:#define __INT_FAST8_TYPE__ signed char
// VE:#define __INT_LEAST16_FMTd__ "hd"
// VE:#define __INT_LEAST16_FMTi__ "hi"
// VE:#define __INT_LEAST16_MAX__ 32767
// VE:#define __INT_LEAST16_TYPE__ short
// VE:#define __INT_LEAST32_FMTd__ "d"
// VE:#define __INT_LEAST32_FMTi__ "i"
// VE:#define __INT_LEAST32_MAX__ 2147483647
// VE:#define __INT_LEAST32_TYPE__ int
// VE:#define __INT_LEAST64_FMTd__ "ld"
// VE:#define __INT_LEAST64_FMTi__ "li"
// VE:#define __INT_LEAST64_MAX__ 9223372036854775807L
// VE:#define __INT_LEAST64_TYPE__ long int
// VE:#define __INT_LEAST8_FMTd__ "hhd"
// VE:#define __INT_LEAST8_FMTi__ "hhi"
// VE:#define __INT_LEAST8_MAX__ 127
// VE:#define __INT_LEAST8_TYPE__ signed char
// VE:#define __INT_MAX__ 2147483647
// VE:#define __LDBL_DECIMAL_DIG__ 36
// VE:#define __LDBL_DENORM_MIN__ 6.47517511943802511092443895822764655e-4966L
// VE:#define __LDBL_DIG__ 33
// VE:#define __LDBL_EPSILON__ 1.92592994438723585305597794258492732e-34L
// VE:#define __LDBL_HAS_DENORM__ 1
// VE:#define __LDBL_HAS_INFINITY__ 1
// VE:#define __LDBL_HAS_QUIET_NAN__ 1
// VE:#define __LDBL_MANT_DIG__ 113
// VE:#define __LDBL_MAX_10_EXP__ 4932
// VE:#define __LDBL_MAX_EXP__ 16384
// VE:#define __LDBL_MAX__ 1.18973149535723176508575932662800702e+4932L
// VE:#define __LDBL_MIN_10_EXP__ (-4931)
// VE:#define __LDBL_MIN_EXP__ (-16381)
// VE:#define __LDBL_MIN__ 3.36210314311209350626267781732175260e-4932L
// VE:#define __LITTLE_ENDIAN__ 1
// VE-NOT:#define __LONGDOUBLE128 1
// VE-NOT:#define __LONG_DOUBLE_128__ 1
// VE:#define __LONG_LONG_MAX__ 9223372036854775807LL
// VE:#define __LONG_MAX__ 9223372036854775807L
// VE:#define __LP64__ 1
// VE:#define __NEC__ 1
// VE-NOT:#define __OPTIMIZE__
// VE:#define __POINTER_WIDTH__ 64
// VE:#define __PTRDIFF_FMTd__ "ld"
// VE:#define __PTRDIFF_FMTi__ "li"
// VE:#define __PTRDIFF_MAX__ 9223372036854775807L
// VE:#define __PTRDIFF_TYPE__ long int
// VE:#define __PTRDIFF_WIDTH__ 64
// VE:#define __SCHAR_MAX__ 127
// VE:#define __SHRT_MAX__ 32767
// VE:#define __SIG_ATOMIC_MAX__ 2147483647
// VE:#define __SIG_ATOMIC_WIDTH__ 32
// VE:#define __SIZEOF_DOUBLE__ 8
// VE:#define __SIZEOF_FLOAT__ 4
// VE:#define __SIZEOF_INT128__ 16
// VE:#define __SIZEOF_INT__ 4
// VE:#define __SIZEOF_LONG_DOUBLE__ 16
// VE:#define __SIZEOF_LONG_LONG__ 8
// VE:#define __SIZEOF_LONG__ 8
// VE:#define __SIZEOF_POINTER__ 8
// VE:#define __SIZEOF_PTRDIFF_T__ 8
// VE:#define __SIZEOF_SHORT__ 2
// VE:#define __SIZEOF_SIZE_T__ 8
// VE:#define __SIZEOF_WCHAR_T__ 4
// VE:#define __SIZEOF_WINT_T__ 4
// VE:#define __SIZE_FMTX__ "lX"
// VE:#define __SIZE_FMTo__ "lo"
// VE:#define __SIZE_FMTu__ "lu"
// VE:#define __SIZE_FMTx__ "lx"
// VE:#define __SIZE_MAX__ 18446744073709551615UL
// VE:#define __SIZE_TYPE__ long unsigned int
// VE:#define __SIZE_WIDTH__ 64
// VE-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 16UL
// VE:#define __STDC_HOSTED__ 1
// VE:#define __UINT16_C_SUFFIX__
// VE:#define __UINT16_FMTX__ "hX"
// VE:#define __UINT16_FMTo__ "ho"
// VE:#define __UINT16_FMTu__ "hu"
// VE:#define __UINT16_FMTx__ "hx"
// VE:#define __UINT16_MAX__ 65535
// VE:#define __UINT16_TYPE__ unsigned short
// VE:#define __UINT32_C_SUFFIX__ U
// VE:#define __UINT32_FMTX__ "X"
// VE:#define __UINT32_FMTo__ "o"
// VE:#define __UINT32_FMTu__ "u"
// VE:#define __UINT32_FMTx__ "x"
// VE:#define __UINT32_MAX__ 4294967295U
// VE:#define __UINT32_TYPE__ unsigned int
// VE:#define __UINT64_C_SUFFIX__ UL
// VE:#define __UINT64_FMTX__ "lX"
// VE:#define __UINT64_FMTo__ "lo"
// VE:#define __UINT64_FMTu__ "lu"
// VE:#define __UINT64_FMTx__ "lx"
// VE:#define __UINT64_MAX__ 18446744073709551615UL
// VE:#define __UINT64_TYPE__ long unsigned int
// VE:#define __UINT8_C_SUFFIX__
// VE:#define __UINT8_FMTX__ "hhX"
// VE:#define __UINT8_FMTo__ "hho"
// VE:#define __UINT8_FMTu__ "hhu"
// VE:#define __UINT8_FMTx__ "hhx"
// VE:#define __UINT8_MAX__ 255
// VE:#define __UINT8_TYPE__ unsigned char
// VE:#define __UINTMAX_C_SUFFIX__ UL
// VE:#define __UINTMAX_FMTX__ "lX"
// VE:#define __UINTMAX_FMTo__ "lo"
// VE:#define __UINTMAX_FMTu__ "lu"
// VE:#define __UINTMAX_FMTx__ "lx"
// VE:#define __UINTMAX_MAX__ 18446744073709551615UL
// VE:#define __UINTMAX_TYPE__ long unsigned int
// VE:#define __UINTMAX_WIDTH__ 64
// VE:#define __UINTPTR_FMTX__ "lX"
// VE:#define __UINTPTR_FMTo__ "lo"
// VE:#define __UINTPTR_FMTu__ "lu"
// VE:#define __UINTPTR_FMTx__ "lx"
// VE:#define __UINTPTR_MAX__ 18446744073709551615UL
// VE:#define __UINTPTR_TYPE__ long unsigned int
// VE:#define __UINTPTR_WIDTH__ 64
// VE:#define __UINT_FAST16_FMTX__ "hX"
// VE:#define __UINT_FAST16_FMTo__ "ho"
// VE:#define __UINT_FAST16_FMTu__ "hu"
// VE:#define __UINT_FAST16_FMTx__ "hx"
// VE:#define __UINT_FAST16_MAX__ 65535
// VE:#define __UINT_FAST16_TYPE__ unsigned short
// VE:#define __UINT_FAST32_FMTX__ "X"
// VE:#define __UINT_FAST32_FMTo__ "o"
// VE:#define __UINT_FAST32_FMTu__ "u"
// VE:#define __UINT_FAST32_FMTx__ "x"
// VE:#define __UINT_FAST32_MAX__ 4294967295U
// VE:#define __UINT_FAST32_TYPE__ unsigned int
// VE:#define __UINT_FAST64_FMTX__ "lX"
// VE:#define __UINT_FAST64_FMTo__ "lo"
// VE:#define __UINT_FAST64_FMTu__ "lu"
// VE:#define __UINT_FAST64_FMTx__ "lx"
// VE:#define __UINT_FAST64_MAX__ 18446744073709551615UL
// VE:#define __UINT_FAST64_TYPE__ long unsigned int
// VE:#define __UINT_FAST8_FMTX__ "hhX"
// VE:#define __UINT_FAST8_FMTo__ "hho"
// VE:#define __UINT_FAST8_FMTu__ "hhu"
// VE:#define __UINT_FAST8_FMTx__ "hhx"
// VE:#define __UINT_FAST8_MAX__ 255
// VE:#define __UINT_FAST8_TYPE__ unsigned char
// VE:#define __UINT_LEAST16_FMTX__ "hX"
// VE:#define __UINT_LEAST16_FMTo__ "ho"
// VE:#define __UINT_LEAST16_FMTu__ "hu"
// VE:#define __UINT_LEAST16_FMTx__ "hx"
// VE:#define __UINT_LEAST16_MAX__ 65535
// VE:#define __UINT_LEAST16_TYPE__ unsigned short
// VE:#define __UINT_LEAST32_FMTX__ "X"
// VE:#define __UINT_LEAST32_FMTo__ "o"
// VE:#define __UINT_LEAST32_FMTu__ "u"
// VE:#define __UINT_LEAST32_FMTx__ "x"
// VE:#define __UINT_LEAST32_MAX__ 4294967295U
// VE:#define __UINT_LEAST32_TYPE__ unsigned int
// VE:#define __UINT_LEAST64_FMTX__ "lX"
// VE:#define __UINT_LEAST64_FMTo__ "lo"
// VE:#define __UINT_LEAST64_FMTu__ "lu"
// VE:#define __UINT_LEAST64_FMTx__ "lx"
// VE:#define __UINT_LEAST64_MAX__ 18446744073709551615UL
// VE:#define __UINT_LEAST64_TYPE__ long unsigned int
// VE:#define __UINT_LEAST8_FMTX__ "hhX"
// VE:#define __UINT_LEAST8_FMTo__ "hho"
// VE:#define __UINT_LEAST8_FMTu__ "hhu"
// VE:#define __UINT_LEAST8_FMTx__ "hhx"
// VE:#define __UINT_LEAST8_MAX__ 255
// VE:#define __UINT_LEAST8_TYPE__ unsigned char
// VE:#define __USER_LABEL_PREFIX__
// VE-NOT:#define __VECTOR__
// VE:#define __WCHAR_MAX__ 4294967295U
// VE:#define __WCHAR_TYPE__ unsigned int
// VE:#define __WCHAR_UNSIGNED__ 1
// VE:#define __WCHAR_WIDTH__ 32
// VE:#define __WINT_MAX__ 4294967295U
// VE:#define __WINT_TYPE__ unsigned int
// VE:#define __WINT_UNSIGNED__ 1
// VE:#define __WINT_WIDTH__ 32
// VE:#define __linux 1
// VE:#define __linux__ 1
// VE:#define __llvm__ 1
// VE:#define __unix 1
// VE:#define __unix__ 1
// VE:#define __ve 1
// VE:#define __ve__ 1
// VE:#define linux 1
// VE:#define unix 1
