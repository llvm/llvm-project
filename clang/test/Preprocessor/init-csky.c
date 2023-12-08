// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=csky < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=CSKY %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=csky-unknown-linux < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefixes=CSKY,CSKY-LINUX %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=csky \
// RUN: -fforce-enable-int128 < /dev/null | FileCheck -match-full-lines \
// RUN: -check-prefixes=CSKY,CSKY-INT128 %s
// CSKY: #define _ILP32 1
// CSKY: #define __ATOMIC_ACQUIRE 2
// CSKY: #define __ATOMIC_ACQ_REL 4
// CSKY: #define __ATOMIC_CONSUME 1
// CSKY: #define __ATOMIC_RELAXED 0
// CSKY: #define __ATOMIC_RELEASE 3
// CSKY: #define __ATOMIC_SEQ_CST 5
// CSKY: #define __BIGGEST_ALIGNMENT__ 4
// CSKY: #define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// CSKY: #define __CHAR16_TYPE__ unsigned short
// CSKY: #define __CHAR32_TYPE__ unsigned int
// CSKY: #define __CHAR_BIT__ 8
// CSKY: #define __DBL_DECIMAL_DIG__ 17
// CSKY: #define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// CSKY: #define __DBL_DIG__ 15
// CSKY: #define __DBL_EPSILON__ 2.2204460492503131e-16
// CSKY: #define __DBL_HAS_DENORM__ 1
// CSKY: #define __DBL_HAS_INFINITY__ 1
// CSKY: #define __DBL_HAS_QUIET_NAN__ 1
// CSKY: #define __DBL_MANT_DIG__ 53
// CSKY: #define __DBL_MAX_10_EXP__ 308
// CSKY: #define __DBL_MAX_EXP__ 1024
// CSKY: #define __DBL_MAX__ 1.7976931348623157e+308
// CSKY: #define __DBL_MIN_10_EXP__ (-307)
// CSKY: #define __DBL_MIN_EXP__ (-1021)
// CSKY: #define __DBL_MIN__ 2.2250738585072014e-308
// CSKY: #define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// CSKY: #define __ELF__ 1
// CSKY: #define __FINITE_MATH_ONLY__ 0
// CSKY: #define __FLT_DECIMAL_DIG__ 9
// CSKY: #define __FLT_DENORM_MIN__ 1.40129846e-45F
// CSKY: #define __FLT_DIG__ 6
// CSKY: #define __FLT_EPSILON__ 1.19209290e-7F
// CSKY: #define __FLT_HAS_DENORM__ 1
// CSKY: #define __FLT_HAS_INFINITY__ 1
// CSKY: #define __FLT_HAS_QUIET_NAN__ 1
// CSKY: #define __FLT_MANT_DIG__ 24
// CSKY: #define __FLT_MAX_10_EXP__ 38
// CSKY: #define __FLT_MAX_EXP__ 128
// CSKY: #define __FLT_MAX__ 3.40282347e+38F
// CSKY: #define __FLT_MIN_10_EXP__ (-37)
// CSKY: #define __FLT_MIN_EXP__ (-125)
// CSKY: #define __FLT_MIN__ 1.17549435e-38F
// CSKY: #define __FLT_RADIX__ 2
// CSKY: #define __GCC_ATOMIC_BOOL_LOCK_FREE 2
// CSKY: #define __GCC_ATOMIC_CHAR16_T_LOCK_FREE 2
// CSKY: #define __GCC_ATOMIC_CHAR32_T_LOCK_FREE 2
// CSKY: #define __GCC_ATOMIC_CHAR_LOCK_FREE 2
// CSKY: #define __GCC_ATOMIC_INT_LOCK_FREE 2
// CSKY: #define __GCC_ATOMIC_LLONG_LOCK_FREE 1
// CSKY: #define __GCC_ATOMIC_LONG_LOCK_FREE 2
// CSKY: #define __GCC_ATOMIC_POINTER_LOCK_FREE 2
// CSKY: #define __GCC_ATOMIC_SHORT_LOCK_FREE 2
// CSKY: #define __GCC_ATOMIC_TEST_AND_SET_TRUEVAL 1
// CSKY: #define __GCC_ATOMIC_WCHAR_T_LOCK_FREE 2
// CSKY: #define __GNUC_MINOR__ {{.*}}
// CSKY: #define __GNUC_PATCHLEVEL__ {{.*}}
// CSKY: #define __GNUC_STDC_INLINE__ 1
// CSKY: #define __GNUC__ {{.*}}
// CSKY: #define __GXX_ABI_VERSION {{.*}}
// CSKY: #define __ILP32__ 1
// CSKY: #define __INT16_C_SUFFIX__
// CSKY: #define __INT16_MAX__ 32767
// CSKY: #define __INT16_TYPE__ short
// CSKY: #define __INT32_C_SUFFIX__
// CSKY: #define __INT32_MAX__ 2147483647
// CSKY: #define __INT32_TYPE__ int
// CSKY: #define __INT64_C_SUFFIX__ LL
// CSKY: #define __INT64_MAX__ 9223372036854775807LL
// CSKY: #define __INT64_TYPE__ long long int
// CSKY: #define __INT8_C_SUFFIX__
// CSKY: #define __INT8_MAX__ 127
// CSKY: #define __INT8_TYPE__ signed char
// CSKY: #define __INTMAX_C_SUFFIX__ LL
// CSKY: #define __INTMAX_MAX__ 9223372036854775807LL
// CSKY: #define __INTMAX_TYPE__ long long int
// CSKY: #define __INTMAX_WIDTH__ 64
// CSKY: #define __INTPTR_MAX__ 2147483647
// CSKY: #define __INTPTR_TYPE__ int
// CSKY: #define __INTPTR_WIDTH__ 32
// TODO: C-SKY GCC defines INT_FAST16 as int
// CSKY: #define __INT_FAST16_MAX__ 32767
// CSKY: #define __INT_FAST16_TYPE__ short
// CSKY: #define __INT_FAST32_MAX__ 2147483647
// CSKY: #define __INT_FAST32_TYPE__ int
// CSKY: #define __INT_FAST64_MAX__ 9223372036854775807LL
// CSKY: #define __INT_FAST64_TYPE__ long long int
// TODO: C-SKY GCC defines INT_FAST8 as int
// CSKY: #define __INT_FAST8_MAX__ 127
// CSKY: #define __INT_FAST8_TYPE__ signed char
// CSKY: #define __INT_LEAST16_MAX__ 32767
// CSKY: #define __INT_LEAST16_TYPE__ short
// CSKY: #define __INT_LEAST32_MAX__ 2147483647
// CSKY: #define __INT_LEAST32_TYPE__ int
// CSKY: #define __INT_LEAST64_MAX__ 9223372036854775807LL
// CSKY: #define __INT_LEAST64_TYPE__ long long int
// CSKY: #define __INT_LEAST8_MAX__ 127
// CSKY: #define __INT_LEAST8_TYPE__ signed char
// CSKY: #define __INT_MAX__ 2147483647
// CSKY: #define __LDBL_DECIMAL_DIG__ 17
// CSKY: #define __LDBL_DENORM_MIN__ 4.9406564584124654e-324L
// CSKY: #define __LDBL_DIG__ 15
// CSKY: #define __LDBL_EPSILON__ 2.2204460492503131e-16L
// CSKY: #define __LDBL_HAS_DENORM__ 1
// CSKY: #define __LDBL_HAS_INFINITY__ 1
// CSKY: #define __LDBL_HAS_QUIET_NAN__ 1
// CSKY: #define __LDBL_MANT_DIG__ 53
// CSKY: #define __LDBL_MAX_10_EXP__ 308
// CSKY: #define __LDBL_MAX_EXP__ 1024
// CSKY: #define __LDBL_MAX__ 1.7976931348623157e+308L
// CSKY: #define __LDBL_MIN_10_EXP__ (-307)
// CSKY: #define __LDBL_MIN_EXP__ (-1021)
// CSKY: #define __LDBL_MIN__ 2.2250738585072014e-308L
// CSKY: #define __LITTLE_ENDIAN__ 1
// CSKY: #define __LONG_LONG_MAX__ 9223372036854775807LL
// CSKY: #define __LONG_MAX__ 2147483647L
// CSKY: #define __NO_INLINE__ 1
// CSKY: #define __POINTER_WIDTH__ 32
// CSKY: #define __PRAGMA_REDEFINE_EXTNAME 1
// CSKY: #define __PTRDIFF_MAX__ 2147483647
// CSKY: #define __PTRDIFF_TYPE__ int
// CSKY: #define __PTRDIFF_WIDTH__ 32
// CSKY: #define __SCHAR_MAX__ 127
// CSKY: #define __SHRT_MAX__ 32767
// CSKY: #define __SIG_ATOMIC_MAX__ 2147483647
// CSKY: #define __SIG_ATOMIC_WIDTH__ 32
// CSKY: #define __SIZEOF_DOUBLE__ 8
// CSKY: #define __SIZEOF_FLOAT__ 4
// CSKY-INT128: #define __SIZEOF_INT128__ 16
// CSKY: #define __SIZEOF_INT__ 4
// CSKY: #define __SIZEOF_LONG_DOUBLE__ 8
// CSKY: #define __SIZEOF_LONG_LONG__ 8
// CSKY: #define __SIZEOF_LONG__ 4
// CSKY: #define __SIZEOF_POINTER__ 4
// CSKY: #define __SIZEOF_PTRDIFF_T__ 4
// CSKY: #define __SIZEOF_SHORT__ 2
// CSKY: #define __SIZEOF_SIZE_T__ 4
// CSKY: #define __SIZEOF_WCHAR_T__ 4
// CSKY: #define __SIZEOF_WINT_T__ 4
// CSKY: #define __SIZE_MAX__ 4294967295U
// CSKY: #define __SIZE_TYPE__ unsigned int
// CSKY: #define __SIZE_WIDTH__ 32
// CSKY: #define __STDC_HOSTED__ 0
// CSKY: #define __STDC_UTF_16__ 1
// CSKY: #define __STDC_UTF_32__ 1
// CSKY: #define __STDC_VERSION__ 201710L
// CSKY: #define __STDC__ 1
// CSKY: #define __UINT16_C_SUFFIX__
// CSKY: #define __UINT16_MAX__ 65535
// CSKY: #define __UINT16_TYPE__ unsigned short
// CSKY: #define __UINT32_C_SUFFIX__ U
// CSKY: #define __UINT32_MAX__ 4294967295U
// CSKY: #define __UINT32_TYPE__ unsigned int
// CSKY: #define __UINT64_C_SUFFIX__ ULL
// CSKY: #define __UINT64_MAX__ 18446744073709551615ULL
// CSKY: #define __UINT64_TYPE__ long long unsigned int
// CSKY: #define __UINT8_C_SUFFIX__
// CSKY: #define __UINT8_MAX__ 255
// CSKY: #define __UINT8_TYPE__ unsigned char
// CSKY: #define __UINTMAX_C_SUFFIX__ ULL
// CSKY: #define __UINTMAX_MAX__ 18446744073709551615ULL
// CSKY: #define __UINTMAX_TYPE__ long long unsigned int
// CSKY: #define __UINTMAX_WIDTH__ 64
// CSKY: #define __UINTPTR_MAX__ 4294967295U
// CSKY: #define __UINTPTR_TYPE__ unsigned int
// CSKY: #define __UINTPTR_WIDTH__ 32
// TODO: C-SKY GCC defines UINT_FAST16 to be unsigned int
// CSKY: #define __UINT_FAST16_MAX__ 65535
// CSKY: #define __UINT_FAST16_TYPE__ unsigned short
// CSKY: #define __UINT_FAST32_MAX__ 4294967295U
// CSKY: #define __UINT_FAST32_TYPE__ unsigned int
// CSKY: #define __UINT_FAST64_MAX__ 18446744073709551615ULL
// CSKY: #define __UINT_FAST64_TYPE__ long long unsigned int
// TODO: C-SKY GCC defines UINT_FAST8 to be unsigned int
// CSKY: #define __UINT_FAST8_MAX__ 255
// CSKY: #define __UINT_FAST8_TYPE__ unsigned char
// CSKY: #define __UINT_LEAST16_MAX__ 65535
// CSKY: #define __UINT_LEAST16_TYPE__ unsigned short
// CSKY: #define __UINT_LEAST32_MAX__ 4294967295U
// CSKY: #define __UINT_LEAST32_TYPE__ unsigned int
// CSKY: #define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// CSKY: #define __UINT_LEAST64_TYPE__ long long unsigned int
// CSKY: #define __UINT_LEAST8_MAX__ 255
// CSKY: #define __UINT_LEAST8_TYPE__ unsigned char
// CSKY: #define __USER_LABEL_PREFIX__
// CSKY: #define __WCHAR_MAX__ 2147483647
// CSKY: #define __WCHAR_TYPE__ int
// CSKY: #define __WCHAR_WIDTH__ 32
// CSKY: #define __WINT_TYPE__ unsigned int
// CSKY: #define __WINT_UNSIGNED__ 1
// CSKY: #define __WINT_WIDTH__ 32
// CSKY: #define __ck810__ 1
// CSKY: #define __ckcore__ 2
// CSKY: #define __cskyLE__ 1
// CSKY: #define __csky__ 2
// CSKY: #define __cskyabi__ 2
// CSKY: #define __cskyle__ 1
// CSKY-LINUX: #define __gnu_linux__ 1
// CSKY-LINUX: #define __linux 1
// CSKY-LINUX: #define __linux__ 1
// CSKY-LINUX: #define __unix 1
// CSKY-LINUX: #define __unix__ 1
// CSKY-LINUX: #define linux 1
// CSKY-LINUX: #define unix 1
