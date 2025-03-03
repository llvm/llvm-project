// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple loongarch32 /dev/null \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA32 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple loongarch32-unknown-linux /dev/null \
// RUN:   | FileCheck --match-full-lines --check-prefixes=LA32,LA32-LINUX %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple loongarch32 \
// RUN:   -fforce-enable-int128 /dev/null | FileCheck --match-full-lines \
// RUN:   --check-prefixes=LA32,LA32-INT128 %s

// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple loongarch64 /dev/null \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA64 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple loongarch64-unknown-linux /dev/null \
// RUN:   | FileCheck --match-full-lines --check-prefixes=LA64,LA64-LINUX %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple loongarch64 \
// RUN:   -fforce-enable-int128 /dev/null | FileCheck --match-full-lines \
// RUN:   --check-prefixes=LA64,LA64-INT128 %s

//// Note that common macros are tested in init.c, such as __VERSION__. So they're not listed here.

// LA32: #define _ILP32 1
// LA32: #define __ATOMIC_ACQUIRE 2
// LA32-NEXT: #define __ATOMIC_ACQ_REL 4
// LA32-NEXT: #define __ATOMIC_CONSUME 1
// LA32-NEXT: #define __ATOMIC_RELAXED 0
// LA32-NEXT: #define __ATOMIC_RELEASE 3
// LA32-NEXT: #define __ATOMIC_SEQ_CST 5
// LA32: #define __BIGGEST_ALIGNMENT__ 16
// LA32: #define __BITINT_MAXWIDTH__ 128
// LA32: #define __BOOL_WIDTH__ 1
// LA32: #define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// LA32: #define __CHAR16_TYPE__ unsigned short
// LA32: #define __CHAR32_TYPE__ unsigned int
// LA32: #define __CHAR_BIT__ 8
// LA32: #define __CLANG_ATOMIC_BOOL_LOCK_FREE 2
// LA32: #define __CLANG_ATOMIC_CHAR16_T_LOCK_FREE 2
// LA32: #define __CLANG_ATOMIC_CHAR32_T_LOCK_FREE 2
// LA32: #define __CLANG_ATOMIC_CHAR_LOCK_FREE 2
// LA32: #define __CLANG_ATOMIC_INT_LOCK_FREE 2
// LA32: #define __CLANG_ATOMIC_LLONG_LOCK_FREE 1
// LA32: #define __CLANG_ATOMIC_LONG_LOCK_FREE 2
// LA32: #define __CLANG_ATOMIC_POINTER_LOCK_FREE 2
// LA32: #define __CLANG_ATOMIC_SHORT_LOCK_FREE 2
// LA32: #define __CLANG_ATOMIC_WCHAR_T_LOCK_FREE 2
// LA32: #define __DBL_DECIMAL_DIG__ 17
// LA32: #define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// LA32: #define __DBL_DIG__ 15
// LA32: #define __DBL_EPSILON__ 2.2204460492503131e-16
// LA32: #define __DBL_HAS_DENORM__ 1
// LA32: #define __DBL_HAS_INFINITY__ 1
// LA32: #define __DBL_HAS_QUIET_NAN__ 1
// LA32: #define __DBL_MANT_DIG__ 53
// LA32: #define __DBL_MAX_10_EXP__ 308
// LA32: #define __DBL_MAX_EXP__ 1024
// LA32: #define __DBL_MAX__ 1.7976931348623157e+308
// LA32: #define __DBL_MIN_10_EXP__ (-307)
// LA32: #define __DBL_MIN_EXP__ (-1021)
// LA32: #define __DBL_MIN__ 2.2250738585072014e-308
// LA32: #define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// LA32: #define __FLT_DECIMAL_DIG__ 9
// LA32: #define __FLT_DENORM_MIN__ 1.40129846e-45F
// LA32: #define __FLT_DIG__ 6
// LA32: #define __FLT_EPSILON__ 1.19209290e-7F
// LA32: #define __FLT_HAS_DENORM__ 1
// LA32: #define __FLT_HAS_INFINITY__ 1
// LA32: #define __FLT_HAS_QUIET_NAN__ 1
// LA32: #define __FLT_MANT_DIG__ 24
// LA32: #define __FLT_MAX_10_EXP__ 38
// LA32: #define __FLT_MAX_EXP__ 128
// LA32: #define __FLT_MAX__ 3.40282347e+38F
// LA32: #define __FLT_MIN_10_EXP__ (-37)
// LA32: #define __FLT_MIN_EXP__ (-125)
// LA32: #define __FLT_MIN__ 1.17549435e-38F
// LA32: #define __FLT_RADIX__ 2
// LA32: #define __GCC_ATOMIC_BOOL_LOCK_FREE 2
// LA32: #define __GCC_ATOMIC_CHAR16_T_LOCK_FREE 2
// LA32: #define __GCC_ATOMIC_CHAR32_T_LOCK_FREE 2
// LA32: #define __GCC_ATOMIC_CHAR_LOCK_FREE 2
// LA32: #define __GCC_ATOMIC_INT_LOCK_FREE 2
// LA32: #define __GCC_ATOMIC_LLONG_LOCK_FREE 1
// LA32: #define __GCC_ATOMIC_LONG_LOCK_FREE 2
// LA32: #define __GCC_ATOMIC_POINTER_LOCK_FREE 2
// LA32: #define __GCC_ATOMIC_SHORT_LOCK_FREE 2
// LA32: #define __GCC_ATOMIC_TEST_AND_SET_TRUEVAL 1
// LA32: #define __GCC_ATOMIC_WCHAR_T_LOCK_FREE 2
// LA32: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1
// LA32: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 1
// LA32: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
// LA32: #define __ILP32__ 1
// LA32: #define __INT16_C(c) c
// LA32: #define __INT16_C_SUFFIX__
// LA32: #define __INT16_FMTd__ "hd"
// LA32: #define __INT16_FMTi__ "hi"
// LA32: #define __INT16_MAX__ 32767
// LA32: #define __INT16_TYPE__ short
// LA32: #define __INT32_C(c) c
// LA32: #define __INT32_C_SUFFIX__
// LA32: #define __INT32_FMTd__ "d"
// LA32: #define __INT32_FMTi__ "i"
// LA32: #define __INT32_MAX__ 2147483647
// LA32: #define __INT32_TYPE__ int
// LA32: #define __INT64_C(c) c##LL
// LA32: #define __INT64_C_SUFFIX__ LL
// LA32: #define __INT64_FMTd__ "lld"
// LA32: #define __INT64_FMTi__ "lli"
// LA32: #define __INT64_MAX__ 9223372036854775807LL
// LA32: #define __INT64_TYPE__ long long int
// LA32: #define __INT8_C(c) c
// LA32: #define __INT8_C_SUFFIX__
// LA32: #define __INT8_FMTd__ "hhd"
// LA32: #define __INT8_FMTi__ "hhi"
// LA32: #define __INT8_MAX__ 127
// LA32: #define __INT8_TYPE__ signed char
// LA32: #define __INTMAX_C(c) c##LL
// LA32: #define __INTMAX_C_SUFFIX__ LL
// LA32: #define __INTMAX_FMTd__ "lld"
// LA32: #define __INTMAX_FMTi__ "lli"
// LA32: #define __INTMAX_MAX__ 9223372036854775807LL
// LA32: #define __INTMAX_TYPE__ long long int
// LA32: #define __INTMAX_WIDTH__ 64
// LA32: #define __INTPTR_FMTd__ "d"
// LA32: #define __INTPTR_FMTi__ "i"
// LA32: #define __INTPTR_MAX__ 2147483647
// LA32: #define __INTPTR_TYPE__ int
// LA32: #define __INTPTR_WIDTH__ 32
// LA32: #define __INT_FAST16_FMTd__ "hd"
// LA32: #define __INT_FAST16_FMTi__ "hi"
// LA32: #define __INT_FAST16_MAX__ 32767
// LA32: #define __INT_FAST16_TYPE__ short
// LA32: #define __INT_FAST16_WIDTH__ 16
// LA32: #define __INT_FAST32_FMTd__ "d"
// LA32: #define __INT_FAST32_FMTi__ "i"
// LA32: #define __INT_FAST32_MAX__ 2147483647
// LA32: #define __INT_FAST32_TYPE__ int
// LA32: #define __INT_FAST32_WIDTH__ 32
// LA32: #define __INT_FAST64_FMTd__ "lld"
// LA32: #define __INT_FAST64_FMTi__ "lli"
// LA32: #define __INT_FAST64_MAX__ 9223372036854775807LL
// LA32: #define __INT_FAST64_TYPE__ long long int
// LA32: #define __INT_FAST64_WIDTH__ 64
// LA32: #define __INT_FAST8_FMTd__ "hhd"
// LA32: #define __INT_FAST8_FMTi__ "hhi"
// LA32: #define __INT_FAST8_MAX__ 127
// LA32: #define __INT_FAST8_TYPE__ signed char
// LA32: #define __INT_FAST8_WIDTH__ 8
// LA32: #define __INT_LEAST16_FMTd__ "hd"
// LA32: #define __INT_LEAST16_FMTi__ "hi"
// LA32: #define __INT_LEAST16_MAX__ 32767
// LA32: #define __INT_LEAST16_TYPE__ short
// LA32: #define __INT_LEAST16_WIDTH__ 16
// LA32: #define __INT_LEAST32_FMTd__ "d"
// LA32: #define __INT_LEAST32_FMTi__ "i"
// LA32: #define __INT_LEAST32_MAX__ 2147483647
// LA32: #define __INT_LEAST32_TYPE__ int
// LA32: #define __INT_LEAST32_WIDTH__ 32
// LA32: #define __INT_LEAST64_FMTd__ "lld"
// LA32: #define __INT_LEAST64_FMTi__ "lli"
// LA32: #define __INT_LEAST64_MAX__ 9223372036854775807LL
// LA32: #define __INT_LEAST64_TYPE__ long long int
// LA32: #define __INT_LEAST64_WIDTH__ 64
// LA32: #define __INT_LEAST8_FMTd__ "hhd"
// LA32: #define __INT_LEAST8_FMTi__ "hhi"
// LA32: #define __INT_LEAST8_MAX__ 127
// LA32: #define __INT_LEAST8_TYPE__ signed char
// LA32: #define __INT_LEAST8_WIDTH__ 8
// LA32: #define __INT_MAX__ 2147483647
// LA32: #define __INT_WIDTH__ 32
// LA32: #define __LDBL_DECIMAL_DIG__ 36
// LA32: #define __LDBL_DENORM_MIN__ 6.47517511943802511092443895822764655e-4966L
// LA32: #define __LDBL_DIG__ 33
// LA32: #define __LDBL_EPSILON__ 1.92592994438723585305597794258492732e-34L
// LA32: #define __LDBL_HAS_DENORM__ 1
// LA32: #define __LDBL_HAS_INFINITY__ 1
// LA32: #define __LDBL_HAS_QUIET_NAN__ 1
// LA32: #define __LDBL_MANT_DIG__ 113
// LA32: #define __LDBL_MAX_10_EXP__ 4932
// LA32: #define __LDBL_MAX_EXP__ 16384
// LA32: #define __LDBL_MAX__ 1.18973149535723176508575932662800702e+4932L
// LA32: #define __LDBL_MIN_10_EXP__ (-4931)
// LA32: #define __LDBL_MIN_EXP__ (-16381)
// LA32: #define __LDBL_MIN__ 3.36210314311209350626267781732175260e-4932L
// LA32: #define __LITTLE_ENDIAN__ 1
// LA32: #define __LLONG_WIDTH__ 64
// LA32: #define __LONG_LONG_MAX__ 9223372036854775807LL
// LA32: #define __LONG_MAX__ 2147483647L
// LA32: #define __LONG_WIDTH__ 32
// LA32: #define __MEMORY_SCOPE_DEVICE 1 
// LA32: #define __MEMORY_SCOPE_SINGLE 4 
// LA32: #define __MEMORY_SCOPE_SYSTEM 0 
// LA32: #define __MEMORY_SCOPE_WRKGRP 2 
// LA32: #define __MEMORY_SCOPE_WVFRNT 3 
// LA32: #define __NO_INLINE__ 1
// LA32: #define __NO_MATH_ERRNO__ 1
// LA32: #define __OBJC_BOOL_IS_BOOL 0
// LA32: #define __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES 3
// LA32: #define __OPENCL_MEMORY_SCOPE_DEVICE 2
// LA32: #define __OPENCL_MEMORY_SCOPE_SUB_GROUP 4
// LA32: #define __OPENCL_MEMORY_SCOPE_WORK_GROUP 1
// LA32: #define __OPENCL_MEMORY_SCOPE_WORK_ITEM 0
// LA32: #define __POINTER_WIDTH__ 32
// LA32: #define __PRAGMA_REDEFINE_EXTNAME 1
// LA32: #define __PTRDIFF_FMTd__ "d"
// LA32: #define __PTRDIFF_FMTi__ "i"
// LA32: #define __PTRDIFF_MAX__ 2147483647
// LA32: #define __PTRDIFF_TYPE__ int
// LA32: #define __PTRDIFF_WIDTH__ 32
// LA32: #define __SCHAR_MAX__ 127
// LA32: #define __SHRT_MAX__ 32767
// LA32: #define __SHRT_WIDTH__ 16
// LA32: #define __SIG_ATOMIC_MAX__ 2147483647
// LA32: #define __SIG_ATOMIC_WIDTH__ 32
// LA32: #define __SIZEOF_DOUBLE__ 8
// LA32: #define __SIZEOF_FLOAT__ 4
// LA32-INT128: #define __SIZEOF_INT128__ 16
// LA32: #define __SIZEOF_INT__ 4
// LA32: #define __SIZEOF_LONG_DOUBLE__ 16
// LA32: #define __SIZEOF_LONG_LONG__ 8
// LA32: #define __SIZEOF_LONG__ 4
// LA32: #define __SIZEOF_POINTER__ 4
// LA32: #define __SIZEOF_PTRDIFF_T__ 4
// LA32: #define __SIZEOF_SHORT__ 2
// LA32: #define __SIZEOF_SIZE_T__ 4
// LA32: #define __SIZEOF_WCHAR_T__ 4
// LA32: #define __SIZEOF_WINT_T__ 4
// LA32: #define __SIZE_FMTX__ "X"
// LA32: #define __SIZE_FMTo__ "o"
// LA32: #define __SIZE_FMTu__ "u"
// LA32: #define __SIZE_FMTx__ "x"
// LA32: #define __SIZE_MAX__ 4294967295U
// LA32: #define __SIZE_TYPE__ unsigned int
// LA32: #define __SIZE_WIDTH__ 32
// LA32: #define __STDC_HOSTED__ 0
// LA32: #define __STDC_UTF_16__ 1
// LA32: #define __STDC_UTF_32__ 1
// LA32: #define __STDC_VERSION__ 201710L
// LA32: #define __STDC__ 1
// LA32: #define __UINT16_C(c) c
// LA32: #define __UINT16_C_SUFFIX__
// LA32: #define __UINT16_FMTX__ "hX"
// LA32: #define __UINT16_FMTo__ "ho"
// LA32: #define __UINT16_FMTu__ "hu"
// LA32: #define __UINT16_FMTx__ "hx"
// LA32: #define __UINT16_MAX__ 65535
// LA32: #define __UINT16_TYPE__ unsigned short
// LA32: #define __UINT32_C(c) c##U
// LA32: #define __UINT32_C_SUFFIX__ U
// LA32: #define __UINT32_FMTX__ "X"
// LA32: #define __UINT32_FMTo__ "o"
// LA32: #define __UINT32_FMTu__ "u"
// LA32: #define __UINT32_FMTx__ "x"
// LA32: #define __UINT32_MAX__ 4294967295U
// LA32: #define __UINT32_TYPE__ unsigned int
// LA32: #define __UINT64_C(c) c##ULL
// LA32: #define __UINT64_C_SUFFIX__ ULL
// LA32: #define __UINT64_FMTX__ "llX"
// LA32: #define __UINT64_FMTo__ "llo"
// LA32: #define __UINT64_FMTu__ "llu"
// LA32: #define __UINT64_FMTx__ "llx"
// LA32: #define __UINT64_MAX__ 18446744073709551615ULL
// LA32: #define __UINT64_TYPE__ long long unsigned int
// LA32: #define __UINT8_C(c) c
// LA32: #define __UINT8_C_SUFFIX__
// LA32: #define __UINT8_FMTX__ "hhX"
// LA32: #define __UINT8_FMTo__ "hho"
// LA32: #define __UINT8_FMTu__ "hhu"
// LA32: #define __UINT8_FMTx__ "hhx"
// LA32: #define __UINT8_MAX__ 255
// LA32: #define __UINT8_TYPE__ unsigned char
// LA32: #define __UINTMAX_C(c) c##ULL
// LA32: #define __UINTMAX_C_SUFFIX__ ULL
// LA32: #define __UINTMAX_FMTX__ "llX"
// LA32: #define __UINTMAX_FMTo__ "llo"
// LA32: #define __UINTMAX_FMTu__ "llu"
// LA32: #define __UINTMAX_FMTx__ "llx"
// LA32: #define __UINTMAX_MAX__ 18446744073709551615ULL
// LA32: #define __UINTMAX_TYPE__ long long unsigned int
// LA32: #define __UINTMAX_WIDTH__ 64
// LA32: #define __UINTPTR_FMTX__ "X"
// LA32: #define __UINTPTR_FMTo__ "o"
// LA32: #define __UINTPTR_FMTu__ "u"
// LA32: #define __UINTPTR_FMTx__ "x"
// LA32: #define __UINTPTR_MAX__ 4294967295U
// LA32: #define __UINTPTR_TYPE__ unsigned int
// LA32: #define __UINTPTR_WIDTH__ 32
// LA32: #define __UINT_FAST16_FMTX__ "hX"
// LA32: #define __UINT_FAST16_FMTo__ "ho"
// LA32: #define __UINT_FAST16_FMTu__ "hu"
// LA32: #define __UINT_FAST16_FMTx__ "hx"
// LA32: #define __UINT_FAST16_MAX__ 65535
// TODO: LoongArch GCC defines UINT_FAST16 to be long unsigned int
// LA32: #define __UINT_FAST16_TYPE__ unsigned short
// LA32: #define __UINT_FAST32_FMTX__ "X"
// LA32: #define __UINT_FAST32_FMTo__ "o"
// LA32: #define __UINT_FAST32_FMTu__ "u"
// LA32: #define __UINT_FAST32_FMTx__ "x"
// LA32: #define __UINT_FAST32_MAX__ 4294967295U
// LA32: #define __UINT_FAST32_TYPE__ unsigned int
// LA32: #define __UINT_FAST64_FMTX__ "llX"
// LA32: #define __UINT_FAST64_FMTo__ "llo"
// LA32: #define __UINT_FAST64_FMTu__ "llu"
// LA32: #define __UINT_FAST64_FMTx__ "llx"
// LA32: #define __UINT_FAST64_MAX__ 18446744073709551615ULL
// LA32: #define __UINT_FAST64_TYPE__ long long unsigned int
// LA32: #define __UINT_FAST8_FMTX__ "hhX"
// LA32: #define __UINT_FAST8_FMTo__ "hho"
// LA32: #define __UINT_FAST8_FMTu__ "hhu"
// LA32: #define __UINT_FAST8_FMTx__ "hhx"
// LA32: #define __UINT_FAST8_MAX__ 255
// LA32: #define __UINT_FAST8_TYPE__ unsigned char
// LA32: #define __UINT_LEAST16_FMTX__ "hX"
// LA32: #define __UINT_LEAST16_FMTo__ "ho"
// LA32: #define __UINT_LEAST16_FMTu__ "hu"
// LA32: #define __UINT_LEAST16_FMTx__ "hx"
// LA32: #define __UINT_LEAST16_MAX__ 65535
// LA32: #define __UINT_LEAST16_TYPE__ unsigned short
// LA32: #define __UINT_LEAST32_FMTX__ "X"
// LA32: #define __UINT_LEAST32_FMTo__ "o"
// LA32: #define __UINT_LEAST32_FMTu__ "u"
// LA32: #define __UINT_LEAST32_FMTx__ "x"
// LA32: #define __UINT_LEAST32_MAX__ 4294967295U
// LA32: #define __UINT_LEAST32_TYPE__ unsigned int
// LA32: #define __UINT_LEAST64_FMTX__ "llX"
// LA32: #define __UINT_LEAST64_FMTo__ "llo"
// LA32: #define __UINT_LEAST64_FMTu__ "llu"
// LA32: #define __UINT_LEAST64_FMTx__ "llx"
// LA32: #define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// LA32: #define __UINT_LEAST64_TYPE__ long long unsigned int
// LA32: #define __UINT_LEAST8_FMTX__ "hhX"
// LA32: #define __UINT_LEAST8_FMTo__ "hho"
// LA32: #define __UINT_LEAST8_FMTu__ "hhu"
// LA32: #define __UINT_LEAST8_FMTx__ "hhx"
// LA32: #define __UINT_LEAST8_MAX__ 255
// LA32: #define __UINT_LEAST8_TYPE__ unsigned char
// LA32: #define __USER_LABEL_PREFIX__
// LA32: #define __WCHAR_MAX__ 2147483647
// LA32: #define __WCHAR_TYPE__ int
// LA32: #define __WCHAR_WIDTH__ 32
// LA32: #define __WINT_MAX__ 4294967295U
// LA32: #define __WINT_TYPE__ unsigned int
// LA32: #define __WINT_UNSIGNED__ 1
// LA32: #define __WINT_WIDTH__ 32
// LA32-LINUX: #define __gnu_linux__ 1
// LA32-LINUX: #define __linux 1
// LA32-LINUX: #define __linux__ 1
// LA32-NOT: #define __loongarch64 1
// LA32: #define __loongarch__ 1
// LA32-LINUX: #define __unix 1
// LA32-LINUX: #define __unix__ 1
// LA32-LINUX: #define linux 1
// LA32-LINUX: #define unix 1

// LA64: #define _LP64 1
// LA64: #define __ATOMIC_ACQUIRE 2
// LA64-NEXT: #define __ATOMIC_ACQ_REL 4
// LA64-NEXT: #define __ATOMIC_CONSUME 1
// LA64-NEXT: #define __ATOMIC_RELAXED 0
// LA64-NEXT: #define __ATOMIC_RELEASE 3
// LA64-NEXT: #define __ATOMIC_SEQ_CST 5
// LA64: #define __BIGGEST_ALIGNMENT__ 16
// LA64: #define __BITINT_MAXWIDTH__ 128
// LA64: #define __BOOL_WIDTH__ 1
// LA64: #define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// LA64: #define __CHAR16_TYPE__ unsigned short
// LA64: #define __CHAR32_TYPE__ unsigned int
// LA64: #define __CHAR_BIT__ 8
// LA64: #define __CLANG_ATOMIC_BOOL_LOCK_FREE 2
// LA64: #define __CLANG_ATOMIC_CHAR16_T_LOCK_FREE 2
// LA64: #define __CLANG_ATOMIC_CHAR32_T_LOCK_FREE 2
// LA64: #define __CLANG_ATOMIC_CHAR_LOCK_FREE 2
// LA64: #define __CLANG_ATOMIC_INT_LOCK_FREE 2
// LA64: #define __CLANG_ATOMIC_LLONG_LOCK_FREE 2
// LA64: #define __CLANG_ATOMIC_LONG_LOCK_FREE 2
// LA64: #define __CLANG_ATOMIC_POINTER_LOCK_FREE 2
// LA64: #define __CLANG_ATOMIC_SHORT_LOCK_FREE 2
// LA64: #define __CLANG_ATOMIC_WCHAR_T_LOCK_FREE 2
// LA64: #define __DBL_DECIMAL_DIG__ 17
// LA64: #define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// LA64: #define __DBL_DIG__ 15
// LA64: #define __DBL_EPSILON__ 2.2204460492503131e-16
// LA64: #define __DBL_HAS_DENORM__ 1
// LA64: #define __DBL_HAS_INFINITY__ 1
// LA64: #define __DBL_HAS_QUIET_NAN__ 1
// LA64: #define __DBL_MANT_DIG__ 53
// LA64: #define __DBL_MAX_10_EXP__ 308
// LA64: #define __DBL_MAX_EXP__ 1024
// LA64: #define __DBL_MAX__ 1.7976931348623157e+308
// LA64: #define __DBL_MIN_10_EXP__ (-307)
// LA64: #define __DBL_MIN_EXP__ (-1021)
// LA64: #define __DBL_MIN__ 2.2250738585072014e-308
// LA64: #define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// LA64: #define __FLT_DECIMAL_DIG__ 9
// LA64: #define __FLT_DENORM_MIN__ 1.40129846e-45F
// LA64: #define __FLT_DIG__ 6
// LA64: #define __FLT_EPSILON__ 1.19209290e-7F
// LA64: #define __FLT_HAS_DENORM__ 1
// LA64: #define __FLT_HAS_INFINITY__ 1
// LA64: #define __FLT_HAS_QUIET_NAN__ 1
// LA64: #define __FLT_MANT_DIG__ 24
// LA64: #define __FLT_MAX_10_EXP__ 38
// LA64: #define __FLT_MAX_EXP__ 128
// LA64: #define __FLT_MAX__ 3.40282347e+38F
// LA64: #define __FLT_MIN_10_EXP__ (-37)
// LA64: #define __FLT_MIN_EXP__ (-125)
// LA64: #define __FLT_MIN__ 1.17549435e-38F
// LA64: #define __FLT_RADIX__ 2
// LA64: #define __GCC_ATOMIC_BOOL_LOCK_FREE 2
// LA64: #define __GCC_ATOMIC_CHAR16_T_LOCK_FREE 2
// LA64: #define __GCC_ATOMIC_CHAR32_T_LOCK_FREE 2
// LA64: #define __GCC_ATOMIC_CHAR_LOCK_FREE 2
// LA64: #define __GCC_ATOMIC_INT_LOCK_FREE 2
// LA64: #define __GCC_ATOMIC_LLONG_LOCK_FREE 2
// LA64: #define __GCC_ATOMIC_LONG_LOCK_FREE 2
// LA64: #define __GCC_ATOMIC_POINTER_LOCK_FREE 2
// LA64: #define __GCC_ATOMIC_SHORT_LOCK_FREE 2
// LA64: #define __GCC_ATOMIC_TEST_AND_SET_TRUEVAL 1
// LA64: #define __GCC_ATOMIC_WCHAR_T_LOCK_FREE 2
// LA64: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1
// LA64: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 1
// LA64: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
// LA64: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 1
// LA64: #define __INT16_C(c) c
// LA64: #define __INT16_C_SUFFIX__
// LA64: #define __INT16_FMTd__ "hd"
// LA64: #define __INT16_FMTi__ "hi"
// LA64: #define __INT16_MAX__ 32767
// LA64: #define __INT16_TYPE__ short
// LA64: #define __INT32_C(c) c
// LA64: #define __INT32_C_SUFFIX__
// LA64: #define __INT32_FMTd__ "d"
// LA64: #define __INT32_FMTi__ "i"
// LA64: #define __INT32_MAX__ 2147483647
// LA64: #define __INT32_TYPE__ int
// LA64: #define __INT64_C(c) c##L
// LA64: #define __INT64_C_SUFFIX__ L
// LA64: #define __INT64_FMTd__ "ld"
// LA64: #define __INT64_FMTi__ "li"
// LA64: #define __INT64_MAX__ 9223372036854775807L
// LA64: #define __INT64_TYPE__ long int
// LA64: #define __INT8_C(c) c
// LA64: #define __INT8_C_SUFFIX__
// LA64: #define __INT8_FMTd__ "hhd"
// LA64: #define __INT8_FMTi__ "hhi"
// LA64: #define __INT8_MAX__ 127
// LA64: #define __INT8_TYPE__ signed char
// LA64: #define __INTMAX_C(c) c##L
// LA64: #define __INTMAX_C_SUFFIX__ L
// LA64: #define __INTMAX_FMTd__ "ld"
// LA64: #define __INTMAX_FMTi__ "li"
// LA64: #define __INTMAX_MAX__ 9223372036854775807L
// LA64: #define __INTMAX_TYPE__ long int
// LA64: #define __INTMAX_WIDTH__ 64
// LA64: #define __INTPTR_FMTd__ "ld"
// LA64: #define __INTPTR_FMTi__ "li"
// LA64: #define __INTPTR_MAX__ 9223372036854775807L
// LA64: #define __INTPTR_TYPE__ long int
// LA64: #define __INTPTR_WIDTH__ 64
// LA64: #define __INT_FAST16_FMTd__ "hd"
// LA64: #define __INT_FAST16_FMTi__ "hi"
// LA64: #define __INT_FAST16_MAX__ 32767
// LA64: #define __INT_FAST16_TYPE__ short
// LA64: #define __INT_FAST16_WIDTH__ 16
// LA64: #define __INT_FAST32_FMTd__ "d"
// LA64: #define __INT_FAST32_FMTi__ "i"
// LA64: #define __INT_FAST32_MAX__ 2147483647
// LA64: #define __INT_FAST32_TYPE__ int
// LA64: #define __INT_FAST32_WIDTH__ 32
// LA64: #define __INT_FAST64_FMTd__ "ld"
// LA64: #define __INT_FAST64_FMTi__ "li"
// LA64: #define __INT_FAST64_MAX__ 9223372036854775807L
// LA64: #define __INT_FAST64_TYPE__ long int
// LA64: #define __INT_FAST64_WIDTH__ 64
// LA64: #define __INT_FAST8_FMTd__ "hhd"
// LA64: #define __INT_FAST8_FMTi__ "hhi"
// LA64: #define __INT_FAST8_MAX__ 127
// LA64: #define __INT_FAST8_TYPE__ signed char
// LA64: #define __INT_FAST8_WIDTH__ 8
// LA64: #define __INT_LEAST16_FMTd__ "hd"
// LA64: #define __INT_LEAST16_FMTi__ "hi"
// LA64: #define __INT_LEAST16_MAX__ 32767
// LA64: #define __INT_LEAST16_TYPE__ short
// LA64: #define __INT_LEAST16_WIDTH__ 16
// LA64: #define __INT_LEAST32_FMTd__ "d"
// LA64: #define __INT_LEAST32_FMTi__ "i"
// LA64: #define __INT_LEAST32_MAX__ 2147483647
// LA64: #define __INT_LEAST32_TYPE__ int
// LA64: #define __INT_LEAST32_WIDTH__ 32
// LA64: #define __INT_LEAST64_FMTd__ "ld"
// LA64: #define __INT_LEAST64_FMTi__ "li"
// LA64: #define __INT_LEAST64_MAX__ 9223372036854775807L
// LA64: #define __INT_LEAST64_TYPE__ long int
// LA64: #define __INT_LEAST64_WIDTH__ 64
// LA64: #define __INT_LEAST8_FMTd__ "hhd"
// LA64: #define __INT_LEAST8_FMTi__ "hhi"
// LA64: #define __INT_LEAST8_MAX__ 127
// LA64: #define __INT_LEAST8_TYPE__ signed char
// LA64: #define __INT_LEAST8_WIDTH__ 8
// LA64: #define __INT_MAX__ 2147483647
// LA64: #define __INT_WIDTH__ 32
// LA64: #define __LDBL_DECIMAL_DIG__ 36
// LA64: #define __LDBL_DENORM_MIN__ 6.47517511943802511092443895822764655e-4966L
// LA64: #define __LDBL_DIG__ 33
// LA64: #define __LDBL_EPSILON__ 1.92592994438723585305597794258492732e-34L
// LA64: #define __LDBL_HAS_DENORM__ 1
// LA64: #define __LDBL_HAS_INFINITY__ 1
// LA64: #define __LDBL_HAS_QUIET_NAN__ 1
// LA64: #define __LDBL_MANT_DIG__ 113
// LA64: #define __LDBL_MAX_10_EXP__ 4932
// LA64: #define __LDBL_MAX_EXP__ 16384
// LA64: #define __LDBL_MAX__ 1.18973149535723176508575932662800702e+4932L
// LA64: #define __LDBL_MIN_10_EXP__ (-4931)
// LA64: #define __LDBL_MIN_EXP__ (-16381)
// LA64: #define __LDBL_MIN__ 3.36210314311209350626267781732175260e-4932L
// LA64: #define __LITTLE_ENDIAN__ 1
// LA64: #define __LLONG_WIDTH__ 64
// LA64: #define __LONG_LONG_MAX__ 9223372036854775807LL
// LA64: #define __LONG_MAX__ 9223372036854775807L
// LA64: #define __LONG_WIDTH__ 64
// LA64: #define __LP64__ 1
// LA64: #define __MEMORY_SCOPE_DEVICE 1 
// LA64: #define __MEMORY_SCOPE_SINGLE 4 
// LA64: #define __MEMORY_SCOPE_SYSTEM 0 
// LA64: #define __MEMORY_SCOPE_WRKGRP 2 
// LA64: #define __MEMORY_SCOPE_WVFRNT 3 
// LA64: #define __NO_INLINE__ 1
// LA64: #define __NO_MATH_ERRNO__ 1
// LA64: #define __OBJC_BOOL_IS_BOOL 0
// LA64: #define __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES 3
// LA64: #define __OPENCL_MEMORY_SCOPE_DEVICE 2
// LA64: #define __OPENCL_MEMORY_SCOPE_SUB_GROUP 4
// LA64: #define __OPENCL_MEMORY_SCOPE_WORK_GROUP 1
// LA64: #define __OPENCL_MEMORY_SCOPE_WORK_ITEM 0
// LA64: #define __POINTER_WIDTH__ 64
// LA64: #define __PRAGMA_REDEFINE_EXTNAME 1
// LA64: #define __PTRDIFF_FMTd__ "ld"
// LA64: #define __PTRDIFF_FMTi__ "li"
// LA64: #define __PTRDIFF_MAX__ 9223372036854775807L
// LA64: #define __PTRDIFF_TYPE__ long int
// LA64: #define __PTRDIFF_WIDTH__ 64
// LA64: #define __SCHAR_MAX__ 127
// LA64: #define __SHRT_MAX__ 32767
// LA64: #define __SHRT_WIDTH__ 16
// LA64: #define __SIG_ATOMIC_MAX__ 2147483647
// LA64: #define __SIG_ATOMIC_WIDTH__ 32
// LA64: #define __SIZEOF_DOUBLE__ 8
// LA64: #define __SIZEOF_FLOAT__ 4
// LA64-INT128: #define __SIZEOF_INT128__ 16
// LA64: #define __SIZEOF_INT__ 4
// LA64: #define __SIZEOF_LONG_DOUBLE__ 16
// LA64: #define __SIZEOF_LONG_LONG__ 8
// LA64: #define __SIZEOF_LONG__ 8
// LA64: #define __SIZEOF_POINTER__ 8
// LA64: #define __SIZEOF_PTRDIFF_T__ 8
// LA64: #define __SIZEOF_SHORT__ 2
// LA64: #define __SIZEOF_SIZE_T__ 8
// LA64: #define __SIZEOF_WCHAR_T__ 4
// LA64: #define __SIZEOF_WINT_T__ 4
// LA64: #define __SIZE_FMTX__ "lX"
// LA64: #define __SIZE_FMTo__ "lo"
// LA64: #define __SIZE_FMTu__ "lu"
// LA64: #define __SIZE_FMTx__ "lx"
// LA64: #define __SIZE_MAX__ 18446744073709551615UL
// LA64: #define __SIZE_TYPE__ long unsigned int
// LA64: #define __SIZE_WIDTH__ 64
// LA64: #define __STDC_HOSTED__ 0
// LA64: #define __STDC_UTF_16__ 1
// LA64: #define __STDC_UTF_32__ 1
// LA64: #define __STDC_VERSION__ 201710L
// LA64: #define __STDC__ 1
// LA64: #define __UINT16_C(c) c
// LA64: #define __UINT16_C_SUFFIX__
// LA64: #define __UINT16_FMTX__ "hX"
// LA64: #define __UINT16_FMTo__ "ho"
// LA64: #define __UINT16_FMTu__ "hu"
// LA64: #define __UINT16_FMTx__ "hx"
// LA64: #define __UINT16_MAX__ 65535
// LA64: #define __UINT16_TYPE__ unsigned short
// LA64: #define __UINT32_C(c) c##U
// LA64: #define __UINT32_C_SUFFIX__ U
// LA64: #define __UINT32_FMTX__ "X"
// LA64: #define __UINT32_FMTo__ "o"
// LA64: #define __UINT32_FMTu__ "u"
// LA64: #define __UINT32_FMTx__ "x"
// LA64: #define __UINT32_MAX__ 4294967295U
// LA64: #define __UINT32_TYPE__ unsigned int
// LA64: #define __UINT64_C(c) c##UL
// LA64: #define __UINT64_C_SUFFIX__ UL
// LA64: #define __UINT64_FMTX__ "lX"
// LA64: #define __UINT64_FMTo__ "lo"
// LA64: #define __UINT64_FMTu__ "lu"
// LA64: #define __UINT64_FMTx__ "lx"
// LA64: #define __UINT64_MAX__ 18446744073709551615UL
// LA64: #define __UINT64_TYPE__ long unsigned int
// LA64: #define __UINT8_C(c) c
// LA64: #define __UINT8_C_SUFFIX__
// LA64: #define __UINT8_FMTX__ "hhX"
// LA64: #define __UINT8_FMTo__ "hho"
// LA64: #define __UINT8_FMTu__ "hhu"
// LA64: #define __UINT8_FMTx__ "hhx"
// LA64: #define __UINT8_MAX__ 255
// LA64: #define __UINT8_TYPE__ unsigned char
// LA64: #define __UINTMAX_C(c) c##UL
// LA64: #define __UINTMAX_C_SUFFIX__ UL
// LA64: #define __UINTMAX_FMTX__ "lX"
// LA64: #define __UINTMAX_FMTo__ "lo"
// LA64: #define __UINTMAX_FMTu__ "lu"
// LA64: #define __UINTMAX_FMTx__ "lx"
// LA64: #define __UINTMAX_MAX__ 18446744073709551615UL
// LA64: #define __UINTMAX_TYPE__ long unsigned int
// LA64: #define __UINTMAX_WIDTH__ 64
// LA64: #define __UINTPTR_FMTX__ "lX"
// LA64: #define __UINTPTR_FMTo__ "lo"
// LA64: #define __UINTPTR_FMTu__ "lu"
// LA64: #define __UINTPTR_FMTx__ "lx"
// LA64: #define __UINTPTR_MAX__ 18446744073709551615UL
// LA64: #define __UINTPTR_TYPE__ long unsigned int
// LA64: #define __UINTPTR_WIDTH__ 64
// LA64: #define __UINT_FAST16_FMTX__ "hX"
// LA64: #define __UINT_FAST16_FMTo__ "ho"
// LA64: #define __UINT_FAST16_FMTu__ "hu"
// LA64: #define __UINT_FAST16_FMTx__ "hx"
// LA64: #define __UINT_FAST16_MAX__ 65535
// TODO: LoongArch GCC defines UINT_FAST16 to be long unsigned int
// LA64: #define __UINT_FAST16_TYPE__ unsigned short
// LA64: #define __UINT_FAST32_FMTX__ "X"
// LA64: #define __UINT_FAST32_FMTo__ "o"
// LA64: #define __UINT_FAST32_FMTu__ "u"
// LA64: #define __UINT_FAST32_FMTx__ "x"
// LA64: #define __UINT_FAST32_MAX__ 4294967295U
// LA64: #define __UINT_FAST32_TYPE__ unsigned int
// LA64: #define __UINT_FAST64_FMTX__ "lX"
// LA64: #define __UINT_FAST64_FMTo__ "lo"
// LA64: #define __UINT_FAST64_FMTu__ "lu"
// LA64: #define __UINT_FAST64_FMTx__ "lx"
// LA64: #define __UINT_FAST64_MAX__ 18446744073709551615UL
// LA64: #define __UINT_FAST64_TYPE__ long unsigned int
// LA64: #define __UINT_FAST8_FMTX__ "hhX"
// LA64: #define __UINT_FAST8_FMTo__ "hho"
// LA64: #define __UINT_FAST8_FMTu__ "hhu"
// LA64: #define __UINT_FAST8_FMTx__ "hhx"
// LA64: #define __UINT_FAST8_MAX__ 255
// LA64: #define __UINT_FAST8_TYPE__ unsigned char
// LA64: #define __UINT_LEAST16_FMTX__ "hX"
// LA64: #define __UINT_LEAST16_FMTo__ "ho"
// LA64: #define __UINT_LEAST16_FMTu__ "hu"
// LA64: #define __UINT_LEAST16_FMTx__ "hx"
// LA64: #define __UINT_LEAST16_MAX__ 65535
// LA64: #define __UINT_LEAST16_TYPE__ unsigned short
// LA64: #define __UINT_LEAST32_FMTX__ "X"
// LA64: #define __UINT_LEAST32_FMTo__ "o"
// LA64: #define __UINT_LEAST32_FMTu__ "u"
// LA64: #define __UINT_LEAST32_FMTx__ "x"
// LA64: #define __UINT_LEAST32_MAX__ 4294967295U
// LA64: #define __UINT_LEAST32_TYPE__ unsigned int
// LA64: #define __UINT_LEAST64_FMTX__ "lX"
// LA64: #define __UINT_LEAST64_FMTo__ "lo"
// LA64: #define __UINT_LEAST64_FMTu__ "lu"
// LA64: #define __UINT_LEAST64_FMTx__ "lx"
// LA64: #define __UINT_LEAST64_MAX__ 18446744073709551615UL
// LA64: #define __UINT_LEAST64_TYPE__ long unsigned int
// LA64: #define __UINT_LEAST8_FMTX__ "hhX"
// LA64: #define __UINT_LEAST8_FMTo__ "hho"
// LA64: #define __UINT_LEAST8_FMTu__ "hhu"
// LA64: #define __UINT_LEAST8_FMTx__ "hhx"
// LA64: #define __UINT_LEAST8_MAX__ 255
// LA64: #define __UINT_LEAST8_TYPE__ unsigned char
// LA64: #define __USER_LABEL_PREFIX__
// LA64: #define __WCHAR_MAX__ 2147483647
// LA64: #define __WCHAR_TYPE__ int
// LA64: #define __WCHAR_WIDTH__ 32
// LA64: #define __WINT_MAX__ 4294967295U
// LA64: #define __WINT_TYPE__ unsigned int
// LA64: #define __WINT_UNSIGNED__ 1
// LA64: #define __WINT_WIDTH__ 32
// LA64-LINUX: #define __gnu_linux__ 1
// LA64-LINUX: #define __linux 1
// LA64-LINUX: #define __linux__ 1
// LA64: #define __loongarch64 1
// LA64: #define __loongarch__ 1
// LA64-LINUX: #define __unix 1
// LA64-LINUX: #define __unix__ 1
// LA64-LINUX: #define linux 1
// LA64-LINUX: #define unix 1


/// Check __loongarch_{double,single,hard,soft}_float, __loongarch_{gr,fr}len, __loongarch_lp64.

// RUN: %clang --target=loongarch32 -mfpu=64 -mabi=ilp32d -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA32-FPU64-ILP32D %s
// RUN: %clang --target=loongarch32 -mdouble-float -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA32-FPU64-ILP32D %s
// LA32-FPU64-ILP32D: #define __loongarch_double_float 1
// LA32-FPU64-ILP32D: #define __loongarch_frlen 64
// LA32-FPU64-ILP32D: #define __loongarch_grlen 32
// LA32-FPU64-ILP32D: #define __loongarch_hard_float 1
// LA32-FPU64-ILP32D-NOT: #define __loongarch_lp64
// LA32-FPU64-ILP32D-NOT: #define __loongarch_single_float
// LA32-FPU64-ILP32D-NOT: #define __loongarch_soft_float

// RUN: %clang --target=loongarch32 -mfpu=64 -mabi=ilp32f -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA32-FPU64-ILP32F %s
// LA32-FPU64-ILP32F-NOT: #define __loongarch_double_float
// LA32-FPU64-ILP32F: #define __loongarch_frlen 64
// LA32-FPU64-ILP32F: #define __loongarch_grlen 32
// LA32-FPU64-ILP32F: #define __loongarch_hard_float 1
// LA32-FPU64-ILP32F-NOT: #define __loongarch_lp64
// LA32-FPU64-ILP32F: #define __loongarch_single_float 1
// LA32-FPU64-ILP32F-NOT: #define __loongarch_soft_float

// RUN: %clang --target=loongarch32 -mfpu=64 -mabi=ilp32s -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA32-FPU64-ILP32S %s
// LA32-FPU64-ILP32S-NOT: #define __loongarch_double_float
// LA32-FPU64-ILP32S: #define __loongarch_frlen 64
// LA32-FPU64-ILP32S: #define __loongarch_grlen 32
// LA32-FPU64-ILP32S-NOT: #define __loongarch_hard_float
// LA32-FPU64-ILP32S-NOT: #define __loongarch_lp64
// LA32-FPU64-ILP32S-NOT: #define __loongarch_single_float
// LA32-FPU64-ILP32S: #define __loongarch_soft_float 1

// RUN: %clang --target=loongarch32 -mfpu=32 -mabi=ilp32f -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA32-FPU32-ILP32F %s
// RUN: %clang --target=loongarch32 -msingle-float -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA32-FPU32-ILP32F %s
// LA32-FPU32-ILP32F-NOT: #define __loongarch_double_float
// LA32-FPU32-ILP32F: #define __loongarch_frlen 32
// LA32-FPU32-ILP32F: #define __loongarch_grlen 32
// LA32-FPU32-ILP32F: #define __loongarch_hard_float 1
// LA32-FPU32-ILP32F-NOT: #define __loongarch_lp64
// LA32-FPU32-ILP32F: #define __loongarch_single_float 1
// LA32-FPU32-ILP32F-NOT: #define __loongarch_soft_float

// RUN: %clang --target=loongarch32 -mfpu=32 -mabi=ilp32s -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA32-FPU32-ILP32S %s
// LA32-FPU32-ILP32S-NOT: #define __loongarch_double_float
// LA32-FPU32-ILP32S: #define __loongarch_frlen 32
// LA32-FPU32-ILP32S: #define __loongarch_grlen 32
// LA32-FPU32-ILP32S-NOT: #define __loongarch_hard_float
// LA32-FPU32-ILP32S-NOT: #define __loongarch_lp64
// LA32-FPU32-ILP32S-NOT: #define __loongarch_single_float
// LA32-FPU32-ILP32S: #define __loongarch_soft_float 1

// RUN: %clang --target=loongarch32 -mfpu=0 -mabi=ilp32s -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA32-FPU0-ILP32S %s
// RUN: %clang --target=loongarch32 -mfpu=none -mabi=ilp32s -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA32-FPU0-ILP32S %s
// RUN: %clang --target=loongarch32 -msoft-float -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA32-FPU0-ILP32S %s
// LA32-FPU0-ILP32S-NOT: #define __loongarch_double_float
// LA32-FPU0-ILP32S: #define __loongarch_frlen 0
// LA32-FPU0-ILP32S: #define __loongarch_grlen 32
// LA32-FPU0-ILP32S-NOT: #define __loongarch_hard_float
// LA32-FPU0-ILP32S-NOT: #define __loongarch_lp64
// LA32-FPU0-ILP32S-NOT: #define __loongarch_single_float
// LA32-FPU0-ILP32S: #define __loongarch_soft_float 1

// RUN: %clang --target=loongarch64 -mfpu=64 -mabi=lp64d -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA64-FPU64-LP64D %s
// RUN: %clang --target=loongarch64 -mdouble-float -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA64-FPU64-LP64D %s
// LA64-FPU64-LP64D: #define __loongarch_double_float 1
// LA64-FPU64-LP64D: #define __loongarch_frlen 64
// LA64-FPU64-LP64D: #define __loongarch_grlen 64
// LA64-FPU64-LP64D: #define __loongarch_hard_float 1
// LA64-FPU64-LP64D: #define __loongarch_lp64 1
// LA64-FPU64-LP64D-NOT: #define __loongarch_single_float
// LA64-FPU64-LP64D-NOT: #define __loongarch_soft_float

// RUN: %clang --target=loongarch64 -mfpu=64 -mabi=lp64f -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA64-FPU64-LP64F %s
// LA64-FPU64-LP64F-NOT: #define __loongarch_double_float
// LA64-FPU64-LP64F: #define __loongarch_frlen 64
// LA64-FPU64-LP64F: #define __loongarch_grlen 64
// LA64-FPU64-LP64F: #define __loongarch_hard_float 1
// LA64-FPU64-LP64F: #define __loongarch_lp64 1
// LA64-FPU64-LP64F: #define __loongarch_single_float 1
// LA64-FPU64-LP64F-NOT: #define __loongarch_soft_float

// RUN: %clang --target=loongarch64 -mfpu=64 -mabi=lp64s -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA64-FPU64-LP64S %s
// LA64-FPU64-LP64S-NOT: #define __loongarch_double_float
// LA64-FPU64-LP64S: #define __loongarch_frlen 64
// LA64-FPU64-LP64S: #define __loongarch_grlen 64
// LA64-FPU64-LP64S-NOT: #define __loongarch_hard_float
// LA64-FPU64-LP64S: #define __loongarch_lp64 1
// LA64-FPU64-LP64S-NOT: #define __loongarch_single_float
// LA64-FPU64-LP64S: #define __loongarch_soft_float 1

// RUN: %clang --target=loongarch64 -mfpu=32 -mabi=lp64f -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA64-FPU32-LP64F %s
// RUN: %clang --target=loongarch64 -msingle-float -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA64-FPU32-LP64F %s
// LA64-FPU32-LP64F-NOT: #define __loongarch_double_float
// LA64-FPU32-LP64F: #define __loongarch_frlen 32
// LA64-FPU32-LP64F: #define __loongarch_grlen 64
// LA64-FPU32-LP64F: #define __loongarch_hard_float 1
// LA64-FPU32-LP64F: #define __loongarch_lp64 1
// LA64-FPU32-LP64F: #define __loongarch_single_float 1
// LA64-FPU32-LP64F-NOT: #define __loongarch_soft_float

// RUN: %clang --target=loongarch64 -mfpu=32 -mabi=lp64s -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA64-FPU32-LP64S %s
// LA64-FPU32-LP64S-NOT: #define __loongarch_double_float
// LA64-FPU32-LP64S: #define __loongarch_frlen 32
// LA64-FPU32-LP64S: #define __loongarch_grlen 64
// LA64-FPU32-LP64S-NOT: #define __loongarch_hard_float
// LA64-FPU32-LP64S: #define __loongarch_lp64 1
// LA64-FPU32-LP64S-NOT: #define __loongarch_single_float
// LA64-FPU32-LP64S: #define __loongarch_soft_float 1

// RUN: %clang --target=loongarch64 -mfpu=0 -mabi=lp64s -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA64-FPU0-LP64S %s
// RUN: %clang --target=loongarch64 -mfpu=none -mabi=lp64s -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA64-FPU0-LP64S %s
// RUN: %clang --target=loongarch64 -msoft-float -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=LA64-FPU0-LP64S %s
// LA64-FPU0-LP64S-NOT: #define __loongarch_double_float
// LA64-FPU0-LP64S: #define __loongarch_frlen 0
// LA64-FPU0-LP64S: #define __loongarch_grlen 64
// LA64-FPU0-LP64S-NOT: #define __loongarch_hard_float
// LA64-FPU0-LP64S: #define __loongarch_lp64 1
// LA64-FPU0-LP64S-NOT: #define __loongarch_single_float
// LA64-FPU0-LP64S: #define __loongarch_soft_float 1

/// Check __loongarch_arch{_tune/_frecipe/_lam_bh/_lamcas/_ld_seq_sa/_div32/_scq}.

// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - | \
// RUN:   FileCheck --match-full-lines --check-prefix=ARCH-TUNE -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=loongarch64 | \
// RUN:   FileCheck --match-full-lines --check-prefix=ARCH-TUNE -DARCH=loongarch64 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la464 | \
// RUN:   FileCheck --match-full-lines --check-prefix=ARCH-TUNE -DARCH=la464 -DTUNE=la464 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -mtune=loongarch64 | \
// RUN:   FileCheck --match-full-lines --check-prefix=ARCH-TUNE -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -mtune=la464 | \
// RUN:   FileCheck --match-full-lines --check-prefix=ARCH-TUNE -DARCH=la64v1.0 -DTUNE=la464 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=loongarch64 -mtune=la464 | \
// RUN:   FileCheck --match-full-lines --check-prefix=ARCH-TUNE -DARCH=loongarch64 -DTUNE=la464 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la464 -mtune=loongarch64 | \
// RUN:   FileCheck --match-full-lines --check-prefix=ARCH-TUNE -DARCH=la464 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la64v1.0 | \
// RUN:   FileCheck --match-full-lines --check-prefix=ARCH-TUNE -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la64v1.0 -Xclang -target-feature -Xclang -lsx | \
// RUN:   FileCheck --match-full-lines --check-prefix=ARCH-TUNE -DARCH=loongarch64 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la64v1.0 -Xclang -target-feature -Xclang +frecipe | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,FRECIPE -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=loongarch64 -Xclang -target-feature -Xclang +lsx | \
// RUN:   FileCheck --match-full-lines --check-prefix=ARCH-TUNE -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la64v1.1 | \
// RUN:   FileCheck --match-full-lines  --check-prefixes=ARCH-TUNE,FRECIPE,LAM-BH,LAMCAS,LD-SEQ-SA,DIV32,SCQ -DARCH=la64v1.1 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la64v1.1 -Xclang -target-feature -Xclang -frecipe | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,LAM-BH,LAMCAS,LD-SEQ-SA,DIV32,SCQ -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la64v1.1 -Xclang -target-feature -Xclang -lsx | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,FRECIPE,LAM-BH,LAMCAS,LD-SEQ-SA,DIV32,SCQ -DARCH=loongarch64 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=loongarch64 -Xclang -target-feature -Xclang +frecipe | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,FRECIPE -DARCH=loongarch64 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=loongarch64 -Xclang -target-feature -Xclang +lsx -Xclang -target-feature -Xclang +frecipe | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,FRECIPE -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la64v1.0 -Xclang -target-feature -Xclang +lam-bh | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,LAM-BH -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la64v1.1 -Xclang -target-feature -Xclang -lam-bh | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,FRECIPE,LAMCAS,LD-SEQ-SA,DIV32,SCQ -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=loongarch64 -Xclang -target-feature -Xclang +lam-bh | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,LAM-BH -DARCH=loongarch64 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=loongarch64 -Xclang -target-feature -Xclang +lsx -Xclang -target-feature -Xclang +lam-bh | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,LAM-BH -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la64v1.0 -Xclang -target-feature -Xclang +lamcas | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,LAMCAS -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la64v1.1 -Xclang -target-feature -Xclang -lamcas | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,FRECIPE,LAM-BH,LD-SEQ-SA,DIV32,SCQ -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=loongarch64 -Xclang -target-feature -Xclang +lamcas | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,LAMCAS -DARCH=loongarch64 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=loongarch64 -Xclang -target-feature -Xclang +lsx -Xclang -target-feature -Xclang +lamcas | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,LAMCAS -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la64v1.0 -Xclang -target-feature -Xclang +ld-seq-sa | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,LD-SEQ-SA -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la64v1.1 -Xclang -target-feature -Xclang -ld-seq-sa | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,FRECIPE,LAM-BH,LAMCAS,DIV32,SCQ -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=loongarch64 -Xclang -target-feature -Xclang +ld-seq-sa | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,LD-SEQ-SA -DARCH=loongarch64 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=loongarch64 -Xclang -target-feature -Xclang +lsx -Xclang -target-feature -Xclang +ld-seq-sa | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,LD-SEQ-SA -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la64v1.0 -Xclang -target-feature -Xclang +div32 | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,DIV32 -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la64v1.1 -Xclang -target-feature -Xclang -div32| \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,FRECIPE,LAM-BH,LAMCAS,LD-SEQ-SA,SCQ -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=loongarch64 -Xclang -target-feature -Xclang +div32 | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,DIV32 -DARCH=loongarch64 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=loongarch64 -Xclang -target-feature -Xclang +lsx -Xclang -target-feature -Xclang +div32 | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,DIV32 -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la64v1.0 -Xclang -target-feature -Xclang +scq | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,SCQ -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la64v1.1 -Xclang -target-feature -Xclang -scq | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,FRECIPE,LAM-BH,LAMCAS,LD-SEQ-SA,DIV32 -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=loongarch64 -Xclang -target-feature -Xclang +scq | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,SCQ -DARCH=loongarch64 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=loongarch64 -Xclang -target-feature -Xclang +lsx -Xclang -target-feature -Xclang +scq | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,SCQ -DARCH=la64v1.0 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la64v1.0 -Xclang -target-feature -Xclang +frecipe -Xclang -target-feature -Xclang +lam-bh  -Xclang -target-feature -Xclang +lamcas -Xclang -target-feature -Xclang +ld-seq-sa -Xclang -target-feature -Xclang +div32 -Xclang -target-feature -Xclang +scq | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE -DARCH=la64v1.1 -DTUNE=loongarch64 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la664 | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,FRECIPE,LAM-BH,LAMCAS,LD-SEQ-SA,DIV32,SCQ -DARCH=la664 -DTUNE=la664 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -mtune=la664 | \
// RUN:   FileCheck --match-full-lines --check-prefix=ARCH-TUNE -DARCH=la64v1.0 -DTUNE=la664 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=loongarch64 -mtune=la664 | \
// RUN:   FileCheck --match-full-lines --check-prefix=ARCH-TUNE -DARCH=loongarch64 -DTUNE=la664 %s
// RUN: %clang --target=loongarch64 -x c -E -dM %s -o - -march=la664 -mtune=loongarch64 | \
// RUN:   FileCheck --match-full-lines --check-prefixes=ARCH-TUNE,FRECIPE,LAM-BH,LAMCAS,LD-SEQ-SA,DIV32,SCQ -DARCH=la664 -DTUNE=loongarch64 %s

// ARCH-TUNE: #define __loongarch_arch "[[ARCH]]"
// DIV32: #define __loongarch_div32 1
// FRECIPE: #define __loongarch_frecipe 1
// LAM-BH: #define __loongarch_lam_bh 1
// LAMCAS: #define __loongarch_lamcas 1
// LD-SEQ-SA: #define __loongarch_ld_seq_sa 1
// SCQ: #define __loongarch_scq 1
// ARCH-TUNE: #define __loongarch_tune "[[TUNE]]"

// RUN: %clang --target=loongarch64 -mlsx -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=MLSX %s
// RUN: %clang --target=loongarch64 -mno-lsx -mlsx -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=MLSX %s
// RUN: %clang --target=loongarch64 -mlsx -mno-lasx -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=MLSX %s
// RUN: %clang --target=loongarch64 -mno-lasx -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=MLSX %s
// RUN: %clang --target=loongarch64 -mno-lasx -mlsx -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=MLSX %s
// MLSX-NOT: #define __loongarch_asx
// MLSX: #define __loongarch_simd_width 128
// MLSX: #define __loongarch_sx 1

// RUN: %clang --target=loongarch64 -mlasx -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=MLASX %s
// RUN: %clang --target=loongarch64 -mlsx -mlasx -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=MLASX %s
// RUN: %clang --target=loongarch64 -mlasx -mlsx -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=MLASX %s
// RUN: %clang --target=loongarch64 -mno-lasx -mlasx -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=MLASX %s
// MLASX: #define __loongarch_asx 1
// MLASX: #define __loongarch_simd_width 256
// MLASX: #define __loongarch_sx 1

// RUN: %clang --target=loongarch64 -mno-lsx -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=MNO-LSX %s
// RUN: %clang --target=loongarch64 -mlsx -mno-lsx -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=MNO-LSX %s
// RUN: %clang --target=loongarch64 -mno-lsx -mno-lasx -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=MNO-LSX %s
// RUN: %clang --target=loongarch64 -mno-lasx -mno-lsx -x c -E -dM %s -o - \
// RUN:   | FileCheck --match-full-lines --check-prefix=MNO-LSX %s
// MNO-LSX-NOT: #define __loongarch_asx
// MNO-LSX-NOT: #define __loongarch_simd_width
// MNO-LSX-NOT: #define __loongarch_sx
