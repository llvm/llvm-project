// RUN: %clang_cc1 -E -dM -x assembler-with-cpp < /dev/null | FileCheck -match-full-lines -check-prefix ASM %s
//
// ASM:#define __ASSEMBLER__ 1
//
//
// RUN: %clang_cc1 -fblocks -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix BLOCKS %s
//
// BLOCKS:#define __BLOCKS__ 1
// BLOCKS:#define __block __attribute__((__blocks__(byref)))
//
//
// RUN: %clang_cc1 -x c++ -fgnuc-version=4.2.1 -std=c++2b -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix CXX2B %s
//
// CXX2B:#define __GNUG__ 4
// CXX2B:#define __GXX_EXPERIMENTAL_CXX0X__ 1
// CXX2B:#define __GXX_RTTI 1
// CXX2B:#define __GXX_WEAK__ 1
// CXX2B:#define __cplusplus 202101L
// CXX2B:#define __private_extern__ extern
//
// RUN: %clang_cc1 -x c++ -fgnuc-version=4.2.1 -std=c++20 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix CXX2A %s
// RUN: %clang_cc1 -x c++ -fgnuc-version=4.2.1 -std=c++2a -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix CXX2A %s
//
// CXX2A:#define __GNUG__ 4
// CXX2A:#define __GXX_EXPERIMENTAL_CXX0X__ 1
// CXX2A:#define __GXX_RTTI 1
// CXX2A:#define __GXX_WEAK__ 1
// CXX2A:#define __cplusplus 202002L
// CXX2A:#define __private_extern__ extern
//
// RUN: %clang_cc1 -x c++ -fgnuc-version=4.2.1 -std=c++17 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix CXX1Z %s
// RUN: %clang_cc1 -x c++ -fgnuc-version=4.2.1 -std=c++1z -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix CXX1Z %s
//
// CXX1Z:#define __GNUG__ 4
// CXX1Z:#define __GXX_EXPERIMENTAL_CXX0X__ 1
// CXX1Z:#define __GXX_RTTI 1
// CXX1Z:#define __GXX_WEAK__ 1
// CXX1Z:#define __cplusplus 201703L
// CXX1Z:#define __private_extern__ extern
//
//
// RUN: %clang_cc1 -x c++ -fgnuc-version=4.2.1 -std=c++14 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix CXX1Y %s
// RUN: %clang_cc1 -x c++ -fgnuc-version=4.2.1 -std=c++1y -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix CXX1Y %s
//
// CXX1Y:#define __GNUG__ 4
// CXX1Y:#define __GXX_EXPERIMENTAL_CXX0X__ 1
// CXX1Y:#define __GXX_RTTI 1
// CXX1Y:#define __GXX_WEAK__ 1
// CXX1Y:#define __cplusplus 201402L
// CXX1Y:#define __private_extern__ extern
//
//
// RUN: %clang_cc1 -x c++ -fgnuc-version=4.2.1 -std=c++11 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix CXX11 %s
//
// CXX11:#define __GNUG__ 4
// CXX11:#define __GXX_EXPERIMENTAL_CXX0X__ 1
// CXX11:#define __GXX_RTTI 1
// CXX11:#define __GXX_WEAK__ 1
// CXX11:#define __cplusplus 201103L
// CXX11:#define __private_extern__ extern
//
//
// RUN: %clang_cc1 -x c++ -fgnuc-version=4.2.1 -std=c++98 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix CXX98 %s
//
// CXX98:#define __GNUG__ 4
// CXX98:#define __GXX_RTTI 1
// CXX98:#define __GXX_WEAK__ 1
// CXX98:#define __cplusplus 199711L
// CXX98:#define __private_extern__ extern
//
//
// RUN: %clang_cc1 -fdeprecated-macro -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix DEPRECATED %s
//
// DEPRECATED:#define __DEPRECATED 1
//
//
// RUN: %clang_cc1 -std=c99 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix C99 %s
//
// C99:#define __STDC_VERSION__ 199901L
// C99:#define __STRICT_ANSI__ 1
// C99-NOT: __GXX_EXPERIMENTAL_CXX0X__
// C99-NOT: __GXX_RTTI
// C99-NOT: __GXX_WEAK__
// C99-NOT: __cplusplus
//
//
// RUN: %clang_cc1 -std=c11 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix C11 %s
// RUN: %clang_cc1 -std=c1x -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix C11 %s
// RUN: %clang_cc1 -std=iso9899:2011 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix C11 %s
// RUN: %clang_cc1 -std=iso9899:201x -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix C11 %s
//
// C11:#define __STDC_UTF_16__ 1
// C11:#define __STDC_UTF_32__ 1
// C11:#define __STDC_VERSION__ 201112L
// C11:#define __STRICT_ANSI__ 1
// C11-NOT: __GXX_EXPERIMENTAL_CXX0X__
// C11-NOT: __GXX_RTTI
// C11-NOT: __GXX_WEAK__
// C11-NOT: __cplusplus
//
//
// RUN: %clang_cc1 -fgnuc-version=4.2.1 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix COMMON %s
//
// COMMON:#define __CONSTANT_CFSTRINGS__ 1
// COMMON:#define __FINITE_MATH_ONLY__ 0
// COMMON:#define __GNUC_MINOR__ {{.*}}
// COMMON:#define __GNUC_PATCHLEVEL__ {{.*}}
// COMMON:#define __GNUC_STDC_INLINE__ 1
// COMMON:#define __GNUC__ {{.*}}
// COMMON:#define __GXX_ABI_VERSION {{.*}}
// COMMON:#define __ORDER_BIG_ENDIAN__ 4321
// COMMON:#define __ORDER_LITTLE_ENDIAN__ 1234
// COMMON:#define __ORDER_PDP_ENDIAN__ 3412
// COMMON:#define __STDC_HOSTED__ 1
// COMMON:#define __STDC__ 1
// COMMON:#define __VERSION__ {{.*}}
// COMMON:#define __clang__ 1
// COMMON:#define __clang_major__ {{[0-9]+}}
// COMMON:#define __clang_minor__ {{[0-9]+}}
// COMMON:#define __clang_patchlevel__ {{[0-9]+}}
// COMMON:#define __clang_version__ {{.*}}
// COMMON:#define __llvm__ 1
//
// RUN: %clang_cc1 -E -dM -triple=x86_64-pc-win32 < /dev/null | FileCheck -match-full-lines -check-prefix C-DEFAULT %s
// RUN: %clang_cc1 -E -dM -triple=x86_64-pc-linux-gnu < /dev/null | FileCheck -match-full-lines -check-prefix C-DEFAULT %s
// RUN: %clang_cc1 -E -dM -triple=x86_64-apple-darwin < /dev/null | FileCheck -match-full-lines -check-prefix C-DEFAULT %s
// RUN: %clang_cc1 -E -dM -triple=armv7a-apple-darwin < /dev/null | FileCheck -match-full-lines -check-prefix C-DEFAULT %s
//
// C-DEFAULT:#define __STDC_VERSION__ 201710L
//
// RUN: %clang_cc1 -ffreestanding -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix FREESTANDING %s
// FREESTANDING:#define __STDC_HOSTED__ 0
//
// RUN: %clang_cc1 -x c++ -fgnuc-version=4.2.1 -std=gnu++2b -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix GXX2B %s
//
// GXX2B:#define __GNUG__ 4
// GXX2B:#define __GXX_WEAK__ 1
// GXX2B:#define __cplusplus 202101L
// GXX2B:#define __private_extern__ extern
//
// RUN: %clang_cc1 -x c++ -fgnuc-version=4.2.1 -std=gnu++20 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix GXX2A %s
// RUN: %clang_cc1 -x c++ -fgnuc-version=4.2.1 -std=gnu++2a -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix GXX2A %s
//
// GXX2A:#define __GNUG__ 4
// GXX2A:#define __GXX_WEAK__ 1
// GXX2A:#define __cplusplus 202002L
// GXX2A:#define __private_extern__ extern
//
//
// RUN: %clang_cc1 -x c++ -fgnuc-version=4.2.1 -std=gnu++17 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix GXX1Z %s
// RUN: %clang_cc1 -x c++ -fgnuc-version=4.2.1 -std=gnu++1z -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix GXX1Z %s
//
// GXX1Z:#define __GNUG__ 4
// GXX1Z:#define __GXX_WEAK__ 1
// GXX1Z:#define __cplusplus 201703L
// GXX1Z:#define __private_extern__ extern
//
//
// RUN: %clang_cc1 -x c++ -fgnuc-version=4.2.1 -std=gnu++14 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix GXX1Y %s
// RUN: %clang_cc1 -x c++ -fgnuc-version=4.2.1 -std=gnu++1y -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix GXX1Y %s
//
// GXX1Y:#define __GNUG__ 4
// GXX1Y:#define __GXX_WEAK__ 1
// GXX1Y:#define __cplusplus 201402L
// GXX1Y:#define __private_extern__ extern
//
//
// RUN: %clang_cc1 -x c++ -fgnuc-version=4.2.1 -std=gnu++11 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix GXX11 %s
//
// GXX11:#define __GNUG__ 4
// GXX11:#define __GXX_WEAK__ 1
// GXX11:#define __cplusplus 201103L
// GXX11:#define __private_extern__ extern
//
//
// RUN: %clang_cc1 -x c++ -fgnuc-version=4.2.1 -std=gnu++98 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix GXX98 %s
//
// GXX98:#define __GNUG__ 4
// GXX98:#define __GXX_WEAK__ 1
// GXX98:#define __cplusplus 199711L
// GXX98:#define __private_extern__ extern
//
//
// RUN: %clang_cc1 -std=iso9899:199409 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix C94 %s
//
// C94:#define __STDC_VERSION__ 199409L
//
//
// RUN: %clang_cc1 -fms-extensions -triple i686-pc-win32 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix MSEXT %s
//
// MSEXT-NOT:#define __STDC__
// MSEXT:#define _INTEGRAL_MAX_BITS 64
// MSEXT-NOT:#define _NATIVE_WCHAR_T_DEFINED 1
// MSEXT-NOT:#define _WCHAR_T_DEFINED 1
//
//
// RUN: %clang_cc1 -x c++ -fms-extensions -triple i686-pc-win32 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix MSEXT-CXX %s
//
// MSEXT-CXX:#define _NATIVE_WCHAR_T_DEFINED 1
// MSEXT-CXX:#define _WCHAR_T_DEFINED 1
// MSEXT-CXX:#define __BOOL_DEFINED 1
//
//
// RUN: %clang_cc1 -x c++ -fno-wchar -fms-extensions -triple i686-pc-win32 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix MSEXT-CXX-NOWCHAR %s
//
// MSEXT-CXX-NOWCHAR-NOT:#define _NATIVE_WCHAR_T_DEFINED 1
// MSEXT-CXX-NOWCHAR-NOT:#define _WCHAR_T_DEFINED 1
// MSEXT-CXX-NOWCHAR:#define __BOOL_DEFINED 1
//
//
// RUN: %clang_cc1 -x objective-c -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix OBJC %s
// RUN: %clang_cc1 -x objective-c++ -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix OBJC %s
//
// OBJC:#define OBJC_NEW_PROPERTIES 1
// OBJC:#define __NEXT_RUNTIME__ 1
// OBJC:#define __OBJC__ 1
//
//
// RUN: %clang_cc1 -x objective-c -fobjc-gc -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix OBJCGC %s
//
// OBJCGC:#define __OBJC_GC__ 1
//
//
// RUN: %clang_cc1 -x objective-c -fobjc-exceptions -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix NONFRAGILE %s
//
// NONFRAGILE:#define OBJC_ZEROCOST_EXCEPTIONS 1
// NONFRAGILE:#define __OBJC2__ 1
//
//
// RUN: %clang_cc1 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix O0 %s
//
// O0:#define __NO_INLINE__ 1
// O0-NOT:#define __OPTIMIZE_SIZE__
// O0-NOT:#define __OPTIMIZE__
//
//
// RUN: %clang_cc1 -fno-inline -O3 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix NO_INLINE %s
//
// NO_INLINE:#define __NO_INLINE__ 1
// NO_INLINE-NOT:#define __OPTIMIZE_SIZE__
// NO_INLINE:#define __OPTIMIZE__ 1
//
//
// RUN: %clang_cc1 -O1 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix O1 %s
//
// O1-NOT:#define __OPTIMIZE_SIZE__
// O1:#define __OPTIMIZE__ 1
//
//
// RUN: %clang_cc1 -Og -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix Og %s
//
// Og-NOT:#define __OPTIMIZE_SIZE__
// Og:#define __OPTIMIZE__ 1
//
//
// RUN: %clang_cc1 -Os -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix Os %s
//
// Os:#define __OPTIMIZE_SIZE__ 1
// Os:#define __OPTIMIZE__ 1
//
//
// RUN: %clang_cc1 -Oz -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix Oz %s
//
// Oz:#define __OPTIMIZE_SIZE__ 1
// Oz:#define __OPTIMIZE__ 1
//
//
// RUN: %clang_cc1 -fpascal-strings -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix PASCAL %s
//
// PASCAL:#define __PASCAL_STRINGS__ 1
//
//
// RUN: %clang_cc1 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix SCHAR %s
//
// SCHAR:#define __STDC__ 1
// SCHAR-NOT:#define __UNSIGNED_CHAR__
// SCHAR:#define __clang__ 1
//
// RUN: %clang_cc1 -E -dM -fwchar-type=short -fno-signed-wchar < /dev/null | FileCheck -match-full-lines -check-prefix SHORTWCHAR %s
// wchar_t is u16 for targeting Win32.
// RUN: %clang_cc1 -E -dM -fwchar-type=short -fno-signed-wchar -triple=x86_64-w64-mingw32 < /dev/null | FileCheck -match-full-lines -check-prefix SHORTWCHAR %s
// RUN: %clang_cc1 -dM -fwchar-type=short -fno-signed-wchar -triple=x86_64-unknown-windows-cygnus -E /dev/null | FileCheck -match-full-lines -check-prefix SHORTWCHAR %s
//
// SHORTWCHAR: #define __SIZEOF_WCHAR_T__ 2
// SHORTWCHAR: #define __WCHAR_MAX__ 65535
// SHORTWCHAR: #define __WCHAR_TYPE__ unsigned short
// SHORTWCHAR: #define __WCHAR_WIDTH__ 16
//
// RUN: %clang_cc1 -E -dM -fwchar-type=int -triple=i686-unknown-unknown < /dev/null | FileCheck -match-full-lines -check-prefix SHORTWCHAR2 %s
// RUN: %clang_cc1 -E -dM -fwchar-type=int -triple=x86_64-unknown-unknown < /dev/null | FileCheck -match-full-lines -check-prefix SHORTWCHAR2 %s
//
// SHORTWCHAR2: #define __SIZEOF_WCHAR_T__ 4
// SHORTWCHAR2: #define __WCHAR_WIDTH__ 32
// Other definitions vary from platform to platform

//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=msp430-none-none < /dev/null | FileCheck -match-full-lines -check-prefix MSP430 %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=msp430-none-none < /dev/null | FileCheck -match-full-lines -check-prefix MSP430 -check-prefix MSP430-CXX %s
//
// MSP430:#define MSP430 1
// MSP430-NOT:#define _LP64
// MSP430:#define __BIGGEST_ALIGNMENT__ 2
// MSP430:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// MSP430:#define __CHAR16_TYPE__ unsigned short
// MSP430:#define __CHAR32_TYPE__ unsigned int
// MSP430:#define __CHAR_BIT__ 8
// MSP430:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// MSP430:#define __DBL_DIG__ 15
// MSP430:#define __DBL_EPSILON__ 2.2204460492503131e-16
// MSP430:#define __DBL_HAS_DENORM__ 1
// MSP430:#define __DBL_HAS_INFINITY__ 1
// MSP430:#define __DBL_HAS_QUIET_NAN__ 1
// MSP430:#define __DBL_MANT_DIG__ 53
// MSP430:#define __DBL_MAX_10_EXP__ 308
// MSP430:#define __DBL_MAX_EXP__ 1024
// MSP430:#define __DBL_MAX__ 1.7976931348623157e+308
// MSP430:#define __DBL_MIN_10_EXP__ (-307)
// MSP430:#define __DBL_MIN_EXP__ (-1021)
// MSP430:#define __DBL_MIN__ 2.2250738585072014e-308
// MSP430:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// MSP430:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// MSP430:#define __FLT_DIG__ 6
// MSP430:#define __FLT_EPSILON__ 1.19209290e-7F
// MSP430:#define __FLT_EVAL_METHOD__ 0
// MSP430:#define __FLT_HAS_DENORM__ 1
// MSP430:#define __FLT_HAS_INFINITY__ 1
// MSP430:#define __FLT_HAS_QUIET_NAN__ 1
// MSP430:#define __FLT_MANT_DIG__ 24
// MSP430:#define __FLT_MAX_10_EXP__ 38
// MSP430:#define __FLT_MAX_EXP__ 128
// MSP430:#define __FLT_MAX__ 3.40282347e+38F
// MSP430:#define __FLT_MIN_10_EXP__ (-37)
// MSP430:#define __FLT_MIN_EXP__ (-125)
// MSP430:#define __FLT_MIN__ 1.17549435e-38F
// MSP430:#define __FLT_RADIX__ 2
// MSP430:#define __INT16_C_SUFFIX__
// MSP430:#define __INT16_FMTd__ "hd"
// MSP430:#define __INT16_FMTi__ "hi"
// MSP430:#define __INT16_MAX__ 32767
// MSP430:#define __INT16_TYPE__ short
// MSP430:#define __INT32_C_SUFFIX__ L
// MSP430:#define __INT32_FMTd__ "ld"
// MSP430:#define __INT32_FMTi__ "li"
// MSP430:#define __INT32_MAX__ 2147483647L
// MSP430:#define __INT32_TYPE__ long int
// MSP430:#define __INT64_C_SUFFIX__ LL
// MSP430:#define __INT64_FMTd__ "lld"
// MSP430:#define __INT64_FMTi__ "lli"
// MSP430:#define __INT64_MAX__ 9223372036854775807LL
// MSP430:#define __INT64_TYPE__ long long int
// MSP430:#define __INT8_C_SUFFIX__
// MSP430:#define __INT8_FMTd__ "hhd"
// MSP430:#define __INT8_FMTi__ "hhi"
// MSP430:#define __INT8_MAX__ 127
// MSP430:#define __INT8_TYPE__ signed char
// MSP430:#define __INTMAX_C_SUFFIX__ LL
// MSP430:#define __INTMAX_FMTd__ "lld"
// MSP430:#define __INTMAX_FMTi__ "lli"
// MSP430:#define __INTMAX_MAX__ 9223372036854775807LL
// MSP430:#define __INTMAX_TYPE__ long long int
// MSP430:#define __INTMAX_WIDTH__ 64
// MSP430:#define __INTPTR_FMTd__ "d"
// MSP430:#define __INTPTR_FMTi__ "i"
// MSP430:#define __INTPTR_MAX__ 32767
// MSP430:#define __INTPTR_TYPE__ int
// MSP430:#define __INTPTR_WIDTH__ 16
// MSP430:#define __INT_FAST16_FMTd__ "hd"
// MSP430:#define __INT_FAST16_FMTi__ "hi"
// MSP430:#define __INT_FAST16_MAX__ 32767
// MSP430:#define __INT_FAST16_TYPE__ short
// MSP430:#define __INT_FAST32_FMTd__ "ld"
// MSP430:#define __INT_FAST32_FMTi__ "li"
// MSP430:#define __INT_FAST32_MAX__ 2147483647L
// MSP430:#define __INT_FAST32_TYPE__ long int
// MSP430:#define __INT_FAST64_FMTd__ "lld"
// MSP430:#define __INT_FAST64_FMTi__ "lli"
// MSP430:#define __INT_FAST64_MAX__ 9223372036854775807LL
// MSP430:#define __INT_FAST64_TYPE__ long long int
// MSP430:#define __INT_FAST8_FMTd__ "hhd"
// MSP430:#define __INT_FAST8_FMTi__ "hhi"
// MSP430:#define __INT_FAST8_MAX__ 127
// MSP430:#define __INT_FAST8_TYPE__ signed char
// MSP430:#define __INT_LEAST16_FMTd__ "hd"
// MSP430:#define __INT_LEAST16_FMTi__ "hi"
// MSP430:#define __INT_LEAST16_MAX__ 32767
// MSP430:#define __INT_LEAST16_TYPE__ short
// MSP430:#define __INT_LEAST32_FMTd__ "ld"
// MSP430:#define __INT_LEAST32_FMTi__ "li"
// MSP430:#define __INT_LEAST32_MAX__ 2147483647L
// MSP430:#define __INT_LEAST32_TYPE__ long int
// MSP430:#define __INT_LEAST64_FMTd__ "lld"
// MSP430:#define __INT_LEAST64_FMTi__ "lli"
// MSP430:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// MSP430:#define __INT_LEAST64_TYPE__ long long int
// MSP430:#define __INT_LEAST8_FMTd__ "hhd"
// MSP430:#define __INT_LEAST8_FMTi__ "hhi"
// MSP430:#define __INT_LEAST8_MAX__ 127
// MSP430:#define __INT_LEAST8_TYPE__ signed char
// MSP430:#define __INT_MAX__ 32767
// MSP430:#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324L
// MSP430:#define __LDBL_DIG__ 15
// MSP430:#define __LDBL_EPSILON__ 2.2204460492503131e-16L
// MSP430:#define __LDBL_HAS_DENORM__ 1
// MSP430:#define __LDBL_HAS_INFINITY__ 1
// MSP430:#define __LDBL_HAS_QUIET_NAN__ 1
// MSP430:#define __LDBL_MANT_DIG__ 53
// MSP430:#define __LDBL_MAX_10_EXP__ 308
// MSP430:#define __LDBL_MAX_EXP__ 1024
// MSP430:#define __LDBL_MAX__ 1.7976931348623157e+308L
// MSP430:#define __LDBL_MIN_10_EXP__ (-307)
// MSP430:#define __LDBL_MIN_EXP__ (-1021)
// MSP430:#define __LDBL_MIN__ 2.2250738585072014e-308L
// MSP430:#define __LITTLE_ENDIAN__ 1
// MSP430:#define __LONG_LONG_MAX__ 9223372036854775807LL
// MSP430:#define __LONG_MAX__ 2147483647L
// MSP430-NOT:#define __LP64__
// MSP430:#define __MSP430__ 1
// MSP430:#define __POINTER_WIDTH__ 16
// MSP430:#define __PTRDIFF_TYPE__ int
// MSP430:#define __PTRDIFF_WIDTH__ 16
// MSP430:#define __SCHAR_MAX__ 127
// MSP430:#define __SHRT_MAX__ 32767
// MSP430:#define __SIG_ATOMIC_MAX__ 2147483647L
// MSP430:#define __SIG_ATOMIC_WIDTH__ 32
// MSP430:#define __SIZEOF_DOUBLE__ 8
// MSP430:#define __SIZEOF_FLOAT__ 4
// MSP430:#define __SIZEOF_INT__ 2
// MSP430:#define __SIZEOF_LONG_DOUBLE__ 8
// MSP430:#define __SIZEOF_LONG_LONG__ 8
// MSP430:#define __SIZEOF_LONG__ 4
// MSP430:#define __SIZEOF_POINTER__ 2
// MSP430:#define __SIZEOF_PTRDIFF_T__ 2
// MSP430:#define __SIZEOF_SHORT__ 2
// MSP430:#define __SIZEOF_SIZE_T__ 2
// MSP430:#define __SIZEOF_WCHAR_T__ 2
// MSP430:#define __SIZEOF_WINT_T__ 2
// MSP430:#define __SIZE_MAX__ 65535U
// MSP430:#define __SIZE_TYPE__ unsigned int
// MSP430:#define __SIZE_WIDTH__ 16
// MSP430-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 2U
// MSP430:#define __UINT16_C_SUFFIX__ U
// MSP430:#define __UINT16_MAX__ 65535U
// MSP430:#define __UINT16_TYPE__ unsigned short
// MSP430:#define __UINT32_C_SUFFIX__ UL
// MSP430:#define __UINT32_MAX__ 4294967295UL
// MSP430:#define __UINT32_TYPE__ long unsigned int
// MSP430:#define __UINT64_C_SUFFIX__ ULL
// MSP430:#define __UINT64_MAX__ 18446744073709551615ULL
// MSP430:#define __UINT64_TYPE__ long long unsigned int
// MSP430:#define __UINT8_C_SUFFIX__
// MSP430:#define __UINT8_MAX__ 255
// MSP430:#define __UINT8_TYPE__ unsigned char
// MSP430:#define __UINTMAX_C_SUFFIX__ ULL
// MSP430:#define __UINTMAX_MAX__ 18446744073709551615ULL
// MSP430:#define __UINTMAX_TYPE__ long long unsigned int
// MSP430:#define __UINTMAX_WIDTH__ 64
// MSP430:#define __UINTPTR_MAX__ 65535U
// MSP430:#define __UINTPTR_TYPE__ unsigned int
// MSP430:#define __UINTPTR_WIDTH__ 16
// MSP430:#define __UINT_FAST16_MAX__ 65535U
// MSP430:#define __UINT_FAST16_TYPE__ unsigned short
// MSP430:#define __UINT_FAST32_MAX__ 4294967295UL
// MSP430:#define __UINT_FAST32_TYPE__ long unsigned int
// MSP430:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// MSP430:#define __UINT_FAST64_TYPE__ long long unsigned int
// MSP430:#define __UINT_FAST8_MAX__ 255
// MSP430:#define __UINT_FAST8_TYPE__ unsigned char
// MSP430:#define __UINT_LEAST16_MAX__ 65535U
// MSP430:#define __UINT_LEAST16_TYPE__ unsigned short
// MSP430:#define __UINT_LEAST32_MAX__ 4294967295UL
// MSP430:#define __UINT_LEAST32_TYPE__ long unsigned int
// MSP430:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// MSP430:#define __UINT_LEAST64_TYPE__ long long unsigned int
// MSP430:#define __UINT_LEAST8_MAX__ 255
// MSP430:#define __UINT_LEAST8_TYPE__ unsigned char
// MSP430:#define __USER_LABEL_PREFIX__
// MSP430:#define __WCHAR_MAX__ 32767
// MSP430:#define __WCHAR_TYPE__ int
// MSP430:#define __WCHAR_WIDTH__ 16
// MSP430:#define __WINT_TYPE__ int
// MSP430:#define __WINT_WIDTH__ 16
// MSP430:#define __clang__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=nvptx-none-none < /dev/null | FileCheck -match-full-lines -check-prefix NVPTX32 %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=nvptx-none-none < /dev/null | FileCheck -match-full-lines -check-prefix NVPTX32 -check-prefix NVPTX32-CXX %s
//
// NVPTX32-NOT:#define _LP64
// NVPTX32:#define __BIGGEST_ALIGNMENT__ 8
// NVPTX32:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// NVPTX32:#define __CHAR16_TYPE__ unsigned short
// NVPTX32:#define __CHAR32_TYPE__ unsigned int
// NVPTX32:#define __CHAR_BIT__ 8
// NVPTX32:#define __CONSTANT_CFSTRINGS__ 1
// NVPTX32:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// NVPTX32:#define __DBL_DIG__ 15
// NVPTX32:#define __DBL_EPSILON__ 2.2204460492503131e-16
// NVPTX32:#define __DBL_HAS_DENORM__ 1
// NVPTX32:#define __DBL_HAS_INFINITY__ 1
// NVPTX32:#define __DBL_HAS_QUIET_NAN__ 1
// NVPTX32:#define __DBL_MANT_DIG__ 53
// NVPTX32:#define __DBL_MAX_10_EXP__ 308
// NVPTX32:#define __DBL_MAX_EXP__ 1024
// NVPTX32:#define __DBL_MAX__ 1.7976931348623157e+308
// NVPTX32:#define __DBL_MIN_10_EXP__ (-307)
// NVPTX32:#define __DBL_MIN_EXP__ (-1021)
// NVPTX32:#define __DBL_MIN__ 2.2250738585072014e-308
// NVPTX32:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// NVPTX32:#define __FINITE_MATH_ONLY__ 0
// NVPTX32:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// NVPTX32:#define __FLT_DIG__ 6
// NVPTX32:#define __FLT_EPSILON__ 1.19209290e-7F
// NVPTX32:#define __FLT_EVAL_METHOD__ 0
// NVPTX32:#define __FLT_HAS_DENORM__ 1
// NVPTX32:#define __FLT_HAS_INFINITY__ 1
// NVPTX32:#define __FLT_HAS_QUIET_NAN__ 1
// NVPTX32:#define __FLT_MANT_DIG__ 24
// NVPTX32:#define __FLT_MAX_10_EXP__ 38
// NVPTX32:#define __FLT_MAX_EXP__ 128
// NVPTX32:#define __FLT_MAX__ 3.40282347e+38F
// NVPTX32:#define __FLT_MIN_10_EXP__ (-37)
// NVPTX32:#define __FLT_MIN_EXP__ (-125)
// NVPTX32:#define __FLT_MIN__ 1.17549435e-38F
// NVPTX32:#define __FLT_RADIX__ 2
// NVPTX32:#define __INT16_C_SUFFIX__
// NVPTX32:#define __INT16_FMTd__ "hd"
// NVPTX32:#define __INT16_FMTi__ "hi"
// NVPTX32:#define __INT16_MAX__ 32767
// NVPTX32:#define __INT16_TYPE__ short
// NVPTX32:#define __INT32_C_SUFFIX__
// NVPTX32:#define __INT32_FMTd__ "d"
// NVPTX32:#define __INT32_FMTi__ "i"
// NVPTX32:#define __INT32_MAX__ 2147483647
// NVPTX32:#define __INT32_TYPE__ int
// NVPTX32:#define __INT64_C_SUFFIX__ LL
// NVPTX32:#define __INT64_FMTd__ "lld"
// NVPTX32:#define __INT64_FMTi__ "lli"
// NVPTX32:#define __INT64_MAX__ 9223372036854775807LL
// NVPTX32:#define __INT64_TYPE__ long long int
// NVPTX32:#define __INT8_C_SUFFIX__
// NVPTX32:#define __INT8_FMTd__ "hhd"
// NVPTX32:#define __INT8_FMTi__ "hhi"
// NVPTX32:#define __INT8_MAX__ 127
// NVPTX32:#define __INT8_TYPE__ signed char
// NVPTX32:#define __INTMAX_C_SUFFIX__ LL
// NVPTX32:#define __INTMAX_FMTd__ "lld"
// NVPTX32:#define __INTMAX_FMTi__ "lli"
// NVPTX32:#define __INTMAX_MAX__ 9223372036854775807LL
// NVPTX32:#define __INTMAX_TYPE__ long long int
// NVPTX32:#define __INTMAX_WIDTH__ 64
// NVPTX32:#define __INTPTR_FMTd__ "d"
// NVPTX32:#define __INTPTR_FMTi__ "i"
// NVPTX32:#define __INTPTR_MAX__ 2147483647
// NVPTX32:#define __INTPTR_TYPE__ int
// NVPTX32:#define __INTPTR_WIDTH__ 32
// NVPTX32:#define __INT_FAST16_FMTd__ "hd"
// NVPTX32:#define __INT_FAST16_FMTi__ "hi"
// NVPTX32:#define __INT_FAST16_MAX__ 32767
// NVPTX32:#define __INT_FAST16_TYPE__ short
// NVPTX32:#define __INT_FAST32_FMTd__ "d"
// NVPTX32:#define __INT_FAST32_FMTi__ "i"
// NVPTX32:#define __INT_FAST32_MAX__ 2147483647
// NVPTX32:#define __INT_FAST32_TYPE__ int
// NVPTX32:#define __INT_FAST64_FMTd__ "lld"
// NVPTX32:#define __INT_FAST64_FMTi__ "lli"
// NVPTX32:#define __INT_FAST64_MAX__ 9223372036854775807LL
// NVPTX32:#define __INT_FAST64_TYPE__ long long int
// NVPTX32:#define __INT_FAST8_FMTd__ "hhd"
// NVPTX32:#define __INT_FAST8_FMTi__ "hhi"
// NVPTX32:#define __INT_FAST8_MAX__ 127
// NVPTX32:#define __INT_FAST8_TYPE__ signed char
// NVPTX32:#define __INT_LEAST16_FMTd__ "hd"
// NVPTX32:#define __INT_LEAST16_FMTi__ "hi"
// NVPTX32:#define __INT_LEAST16_MAX__ 32767
// NVPTX32:#define __INT_LEAST16_TYPE__ short
// NVPTX32:#define __INT_LEAST32_FMTd__ "d"
// NVPTX32:#define __INT_LEAST32_FMTi__ "i"
// NVPTX32:#define __INT_LEAST32_MAX__ 2147483647
// NVPTX32:#define __INT_LEAST32_TYPE__ int
// NVPTX32:#define __INT_LEAST64_FMTd__ "lld"
// NVPTX32:#define __INT_LEAST64_FMTi__ "lli"
// NVPTX32:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// NVPTX32:#define __INT_LEAST64_TYPE__ long long int
// NVPTX32:#define __INT_LEAST8_FMTd__ "hhd"
// NVPTX32:#define __INT_LEAST8_FMTi__ "hhi"
// NVPTX32:#define __INT_LEAST8_MAX__ 127
// NVPTX32:#define __INT_LEAST8_TYPE__ signed char
// NVPTX32:#define __INT_MAX__ 2147483647
// NVPTX32:#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324L
// NVPTX32:#define __LDBL_DIG__ 15
// NVPTX32:#define __LDBL_EPSILON__ 2.2204460492503131e-16L
// NVPTX32:#define __LDBL_HAS_DENORM__ 1
// NVPTX32:#define __LDBL_HAS_INFINITY__ 1
// NVPTX32:#define __LDBL_HAS_QUIET_NAN__ 1
// NVPTX32:#define __LDBL_MANT_DIG__ 53
// NVPTX32:#define __LDBL_MAX_10_EXP__ 308
// NVPTX32:#define __LDBL_MAX_EXP__ 1024
// NVPTX32:#define __LDBL_MAX__ 1.7976931348623157e+308L
// NVPTX32:#define __LDBL_MIN_10_EXP__ (-307)
// NVPTX32:#define __LDBL_MIN_EXP__ (-1021)
// NVPTX32:#define __LDBL_MIN__ 2.2250738585072014e-308L
// NVPTX32:#define __LITTLE_ENDIAN__ 1
// NVPTX32:#define __LONG_LONG_MAX__ 9223372036854775807LL
// NVPTX32:#define __LONG_MAX__ 2147483647L
// NVPTX32-NOT:#define __LP64__
// NVPTX32:#define __NVPTX__ 1
// NVPTX32:#define __POINTER_WIDTH__ 32
// NVPTX32:#define __PRAGMA_REDEFINE_EXTNAME 1
// NVPTX32:#define __PTRDIFF_TYPE__ int
// NVPTX32:#define __PTRDIFF_WIDTH__ 32
// NVPTX32:#define __PTX__ 1
// NVPTX32:#define __SCHAR_MAX__ 127
// NVPTX32:#define __SHRT_MAX__ 32767
// NVPTX32:#define __SIG_ATOMIC_MAX__ 2147483647
// NVPTX32:#define __SIG_ATOMIC_WIDTH__ 32
// NVPTX32:#define __SIZEOF_DOUBLE__ 8
// NVPTX32:#define __SIZEOF_FLOAT__ 4
// NVPTX32:#define __SIZEOF_INT__ 4
// NVPTX32:#define __SIZEOF_LONG_DOUBLE__ 8
// NVPTX32:#define __SIZEOF_LONG_LONG__ 8
// NVPTX32:#define __SIZEOF_LONG__ 4
// NVPTX32:#define __SIZEOF_POINTER__ 4
// NVPTX32:#define __SIZEOF_PTRDIFF_T__ 4
// NVPTX32:#define __SIZEOF_SHORT__ 2
// NVPTX32:#define __SIZEOF_SIZE_T__ 4
// NVPTX32:#define __SIZEOF_WCHAR_T__ 4
// NVPTX32:#define __SIZEOF_WINT_T__ 4
// NVPTX32:#define __SIZE_MAX__ 4294967295U
// NVPTX32:#define __SIZE_TYPE__ unsigned int
// NVPTX32:#define __SIZE_WIDTH__ 32
// NVPTX32-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 8U
// NVPTX32:#define __UINT16_C_SUFFIX__
// NVPTX32:#define __UINT16_MAX__ 65535
// NVPTX32:#define __UINT16_TYPE__ unsigned short
// NVPTX32:#define __UINT32_C_SUFFIX__ U
// NVPTX32:#define __UINT32_MAX__ 4294967295U
// NVPTX32:#define __UINT32_TYPE__ unsigned int
// NVPTX32:#define __UINT64_C_SUFFIX__ ULL
// NVPTX32:#define __UINT64_MAX__ 18446744073709551615ULL
// NVPTX32:#define __UINT64_TYPE__ long long unsigned int
// NVPTX32:#define __UINT8_C_SUFFIX__
// NVPTX32:#define __UINT8_MAX__ 255
// NVPTX32:#define __UINT8_TYPE__ unsigned char
// NVPTX32:#define __UINTMAX_C_SUFFIX__ ULL
// NVPTX32:#define __UINTMAX_MAX__ 18446744073709551615ULL
// NVPTX32:#define __UINTMAX_TYPE__ long long unsigned int
// NVPTX32:#define __UINTMAX_WIDTH__ 64
// NVPTX32:#define __UINTPTR_MAX__ 4294967295U
// NVPTX32:#define __UINTPTR_TYPE__ unsigned int
// NVPTX32:#define __UINTPTR_WIDTH__ 32
// NVPTX32:#define __UINT_FAST16_MAX__ 65535
// NVPTX32:#define __UINT_FAST16_TYPE__ unsigned short
// NVPTX32:#define __UINT_FAST32_MAX__ 4294967295U
// NVPTX32:#define __UINT_FAST32_TYPE__ unsigned int
// NVPTX32:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// NVPTX32:#define __UINT_FAST64_TYPE__ long long unsigned int
// NVPTX32:#define __UINT_FAST8_MAX__ 255
// NVPTX32:#define __UINT_FAST8_TYPE__ unsigned char
// NVPTX32:#define __UINT_LEAST16_MAX__ 65535
// NVPTX32:#define __UINT_LEAST16_TYPE__ unsigned short
// NVPTX32:#define __UINT_LEAST32_MAX__ 4294967295U
// NVPTX32:#define __UINT_LEAST32_TYPE__ unsigned int
// NVPTX32:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// NVPTX32:#define __UINT_LEAST64_TYPE__ long long unsigned int
// NVPTX32:#define __UINT_LEAST8_MAX__ 255
// NVPTX32:#define __UINT_LEAST8_TYPE__ unsigned char
// NVPTX32:#define __USER_LABEL_PREFIX__
// NVPTX32:#define __WCHAR_MAX__ 2147483647
// NVPTX32:#define __WCHAR_TYPE__ int
// NVPTX32:#define __WCHAR_WIDTH__ 32
// NVPTX32:#define __WINT_TYPE__ int
// NVPTX32:#define __WINT_WIDTH__ 32
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=nvptx64-none-none < /dev/null | FileCheck -match-full-lines -check-prefix NVPTX64 %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=nvptx64-none-none < /dev/null | FileCheck -match-full-lines -check-prefix NVPTX64 -check-prefix NVPTX64-CXX %s
//
// NVPTX64:#define _LP64 1
// NVPTX64:#define __BIGGEST_ALIGNMENT__ 8
// NVPTX64:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// NVPTX64:#define __CHAR16_TYPE__ unsigned short
// NVPTX64:#define __CHAR32_TYPE__ unsigned int
// NVPTX64:#define __CHAR_BIT__ 8
// NVPTX64:#define __CONSTANT_CFSTRINGS__ 1
// NVPTX64:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// NVPTX64:#define __DBL_DIG__ 15
// NVPTX64:#define __DBL_EPSILON__ 2.2204460492503131e-16
// NVPTX64:#define __DBL_HAS_DENORM__ 1
// NVPTX64:#define __DBL_HAS_INFINITY__ 1
// NVPTX64:#define __DBL_HAS_QUIET_NAN__ 1
// NVPTX64:#define __DBL_MANT_DIG__ 53
// NVPTX64:#define __DBL_MAX_10_EXP__ 308
// NVPTX64:#define __DBL_MAX_EXP__ 1024
// NVPTX64:#define __DBL_MAX__ 1.7976931348623157e+308
// NVPTX64:#define __DBL_MIN_10_EXP__ (-307)
// NVPTX64:#define __DBL_MIN_EXP__ (-1021)
// NVPTX64:#define __DBL_MIN__ 2.2250738585072014e-308
// NVPTX64:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// NVPTX64:#define __FINITE_MATH_ONLY__ 0
// NVPTX64:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// NVPTX64:#define __FLT_DIG__ 6
// NVPTX64:#define __FLT_EPSILON__ 1.19209290e-7F
// NVPTX64:#define __FLT_EVAL_METHOD__ 0
// NVPTX64:#define __FLT_HAS_DENORM__ 1
// NVPTX64:#define __FLT_HAS_INFINITY__ 1
// NVPTX64:#define __FLT_HAS_QUIET_NAN__ 1
// NVPTX64:#define __FLT_MANT_DIG__ 24
// NVPTX64:#define __FLT_MAX_10_EXP__ 38
// NVPTX64:#define __FLT_MAX_EXP__ 128
// NVPTX64:#define __FLT_MAX__ 3.40282347e+38F
// NVPTX64:#define __FLT_MIN_10_EXP__ (-37)
// NVPTX64:#define __FLT_MIN_EXP__ (-125)
// NVPTX64:#define __FLT_MIN__ 1.17549435e-38F
// NVPTX64:#define __FLT_RADIX__ 2
// NVPTX64:#define __INT16_C_SUFFIX__
// NVPTX64:#define __INT16_FMTd__ "hd"
// NVPTX64:#define __INT16_FMTi__ "hi"
// NVPTX64:#define __INT16_MAX__ 32767
// NVPTX64:#define __INT16_TYPE__ short
// NVPTX64:#define __INT32_C_SUFFIX__
// NVPTX64:#define __INT32_FMTd__ "d"
// NVPTX64:#define __INT32_FMTi__ "i"
// NVPTX64:#define __INT32_MAX__ 2147483647
// NVPTX64:#define __INT32_TYPE__ int
// NVPTX64:#define __INT64_C_SUFFIX__ LL
// NVPTX64:#define __INT64_FMTd__ "lld"
// NVPTX64:#define __INT64_FMTi__ "lli"
// NVPTX64:#define __INT64_MAX__ 9223372036854775807LL
// NVPTX64:#define __INT64_TYPE__ long long int
// NVPTX64:#define __INT8_C_SUFFIX__
// NVPTX64:#define __INT8_FMTd__ "hhd"
// NVPTX64:#define __INT8_FMTi__ "hhi"
// NVPTX64:#define __INT8_MAX__ 127
// NVPTX64:#define __INT8_TYPE__ signed char
// NVPTX64:#define __INTMAX_C_SUFFIX__ LL
// NVPTX64:#define __INTMAX_FMTd__ "lld"
// NVPTX64:#define __INTMAX_FMTi__ "lli"
// NVPTX64:#define __INTMAX_MAX__ 9223372036854775807LL
// NVPTX64:#define __INTMAX_TYPE__ long long int
// NVPTX64:#define __INTMAX_WIDTH__ 64
// NVPTX64:#define __INTPTR_FMTd__ "ld"
// NVPTX64:#define __INTPTR_FMTi__ "li"
// NVPTX64:#define __INTPTR_MAX__ 9223372036854775807L
// NVPTX64:#define __INTPTR_TYPE__ long int
// NVPTX64:#define __INTPTR_WIDTH__ 64
// NVPTX64:#define __INT_FAST16_FMTd__ "hd"
// NVPTX64:#define __INT_FAST16_FMTi__ "hi"
// NVPTX64:#define __INT_FAST16_MAX__ 32767
// NVPTX64:#define __INT_FAST16_TYPE__ short
// NVPTX64:#define __INT_FAST32_FMTd__ "d"
// NVPTX64:#define __INT_FAST32_FMTi__ "i"
// NVPTX64:#define __INT_FAST32_MAX__ 2147483647
// NVPTX64:#define __INT_FAST32_TYPE__ int
// NVPTX64:#define __INT_FAST64_FMTd__ "ld"
// NVPTX64:#define __INT_FAST64_FMTi__ "li"
// NVPTX64:#define __INT_FAST64_MAX__ 9223372036854775807L
// NVPTX64:#define __INT_FAST64_TYPE__ long int
// NVPTX64:#define __INT_FAST8_FMTd__ "hhd"
// NVPTX64:#define __INT_FAST8_FMTi__ "hhi"
// NVPTX64:#define __INT_FAST8_MAX__ 127
// NVPTX64:#define __INT_FAST8_TYPE__ signed char
// NVPTX64:#define __INT_LEAST16_FMTd__ "hd"
// NVPTX64:#define __INT_LEAST16_FMTi__ "hi"
// NVPTX64:#define __INT_LEAST16_MAX__ 32767
// NVPTX64:#define __INT_LEAST16_TYPE__ short
// NVPTX64:#define __INT_LEAST32_FMTd__ "d"
// NVPTX64:#define __INT_LEAST32_FMTi__ "i"
// NVPTX64:#define __INT_LEAST32_MAX__ 2147483647
// NVPTX64:#define __INT_LEAST32_TYPE__ int
// NVPTX64:#define __INT_LEAST64_FMTd__ "ld"
// NVPTX64:#define __INT_LEAST64_FMTi__ "li"
// NVPTX64:#define __INT_LEAST64_MAX__ 9223372036854775807L
// NVPTX64:#define __INT_LEAST64_TYPE__ long int
// NVPTX64:#define __INT_LEAST8_FMTd__ "hhd"
// NVPTX64:#define __INT_LEAST8_FMTi__ "hhi"
// NVPTX64:#define __INT_LEAST8_MAX__ 127
// NVPTX64:#define __INT_LEAST8_TYPE__ signed char
// NVPTX64:#define __INT_MAX__ 2147483647
// NVPTX64:#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324L
// NVPTX64:#define __LDBL_DIG__ 15
// NVPTX64:#define __LDBL_EPSILON__ 2.2204460492503131e-16L
// NVPTX64:#define __LDBL_HAS_DENORM__ 1
// NVPTX64:#define __LDBL_HAS_INFINITY__ 1
// NVPTX64:#define __LDBL_HAS_QUIET_NAN__ 1
// NVPTX64:#define __LDBL_MANT_DIG__ 53
// NVPTX64:#define __LDBL_MAX_10_EXP__ 308
// NVPTX64:#define __LDBL_MAX_EXP__ 1024
// NVPTX64:#define __LDBL_MAX__ 1.7976931348623157e+308L
// NVPTX64:#define __LDBL_MIN_10_EXP__ (-307)
// NVPTX64:#define __LDBL_MIN_EXP__ (-1021)
// NVPTX64:#define __LDBL_MIN__ 2.2250738585072014e-308L
// NVPTX64:#define __LITTLE_ENDIAN__ 1
// NVPTX64:#define __LONG_LONG_MAX__ 9223372036854775807LL
// NVPTX64:#define __LONG_MAX__ 9223372036854775807L
// NVPTX64:#define __LP64__ 1
// NVPTX64:#define __NVPTX__ 1
// NVPTX64:#define __POINTER_WIDTH__ 64
// NVPTX64:#define __PRAGMA_REDEFINE_EXTNAME 1
// NVPTX64:#define __PTRDIFF_TYPE__ long int
// NVPTX64:#define __PTRDIFF_WIDTH__ 64
// NVPTX64:#define __PTX__ 1
// NVPTX64:#define __SCHAR_MAX__ 127
// NVPTX64:#define __SHRT_MAX__ 32767
// NVPTX64:#define __SIG_ATOMIC_MAX__ 2147483647
// NVPTX64:#define __SIG_ATOMIC_WIDTH__ 32
// NVPTX64:#define __SIZEOF_DOUBLE__ 8
// NVPTX64:#define __SIZEOF_FLOAT__ 4
// NVPTX64:#define __SIZEOF_INT__ 4
// NVPTX64:#define __SIZEOF_LONG_DOUBLE__ 8
// NVPTX64:#define __SIZEOF_LONG_LONG__ 8
// NVPTX64:#define __SIZEOF_LONG__ 8
// NVPTX64:#define __SIZEOF_POINTER__ 8
// NVPTX64:#define __SIZEOF_PTRDIFF_T__ 8
// NVPTX64:#define __SIZEOF_SHORT__ 2
// NVPTX64:#define __SIZEOF_SIZE_T__ 8
// NVPTX64:#define __SIZEOF_WCHAR_T__ 4
// NVPTX64:#define __SIZEOF_WINT_T__ 4
// NVPTX64:#define __SIZE_MAX__ 18446744073709551615UL
// NVPTX64:#define __SIZE_TYPE__ long unsigned int
// NVPTX64:#define __SIZE_WIDTH__ 64
// NVPTX64-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 8UL
// NVPTX64:#define __UINT16_C_SUFFIX__
// NVPTX64:#define __UINT16_MAX__ 65535
// NVPTX64:#define __UINT16_TYPE__ unsigned short
// NVPTX64:#define __UINT32_C_SUFFIX__ U
// NVPTX64:#define __UINT32_MAX__ 4294967295U
// NVPTX64:#define __UINT32_TYPE__ unsigned int
// NVPTX64:#define __UINT64_C_SUFFIX__ ULL
// NVPTX64:#define __UINT64_MAX__ 18446744073709551615ULL
// NVPTX64:#define __UINT64_TYPE__ long long unsigned int
// NVPTX64:#define __UINT8_C_SUFFIX__
// NVPTX64:#define __UINT8_MAX__ 255
// NVPTX64:#define __UINT8_TYPE__ unsigned char
// NVPTX64:#define __UINTMAX_C_SUFFIX__ ULL
// NVPTX64:#define __UINTMAX_MAX__ 18446744073709551615ULL
// NVPTX64:#define __UINTMAX_TYPE__ long long unsigned int
// NVPTX64:#define __UINTMAX_WIDTH__ 64
// NVPTX64:#define __UINTPTR_MAX__ 18446744073709551615UL
// NVPTX64:#define __UINTPTR_TYPE__ long unsigned int
// NVPTX64:#define __UINTPTR_WIDTH__ 64
// NVPTX64:#define __UINT_FAST16_MAX__ 65535
// NVPTX64:#define __UINT_FAST16_TYPE__ unsigned short
// NVPTX64:#define __UINT_FAST32_MAX__ 4294967295U
// NVPTX64:#define __UINT_FAST32_TYPE__ unsigned int
// NVPTX64:#define __UINT_FAST64_MAX__ 18446744073709551615UL
// NVPTX64:#define __UINT_FAST64_TYPE__ long unsigned int
// NVPTX64:#define __UINT_FAST8_MAX__ 255
// NVPTX64:#define __UINT_FAST8_TYPE__ unsigned char
// NVPTX64:#define __UINT_LEAST16_MAX__ 65535
// NVPTX64:#define __UINT_LEAST16_TYPE__ unsigned short
// NVPTX64:#define __UINT_LEAST32_MAX__ 4294967295U
// NVPTX64:#define __UINT_LEAST32_TYPE__ unsigned int
// NVPTX64:#define __UINT_LEAST64_MAX__ 18446744073709551615UL
// NVPTX64:#define __UINT_LEAST64_TYPE__ long unsigned int
// NVPTX64:#define __UINT_LEAST8_MAX__ 255
// NVPTX64:#define __UINT_LEAST8_TYPE__ unsigned char
// NVPTX64:#define __USER_LABEL_PREFIX__
// NVPTX64:#define __WCHAR_MAX__ 2147483647
// NVPTX64:#define __WCHAR_TYPE__ int
// NVPTX64:#define __WCHAR_WIDTH__ 32
// NVPTX64:#define __WINT_TYPE__ int
// NVPTX64:#define __WINT_WIDTH__ 32
//

// RUN: %clang_cc1 -x cl -E -dM -ffreestanding -triple=amdgcn < /dev/null | FileCheck -match-full-lines -check-prefix AMDGCN --check-prefix AMDGPU %s
// RUN: %clang_cc1 -x cl -E -dM -ffreestanding -triple=r600 -target-cpu caicos < /dev/null | FileCheck -match-full-lines --check-prefix AMDGPU %s
//
// AMDGPU:#define __ENDIAN_LITTLE__ 1
// AMDGPU:#define cl_khr_byte_addressable_store 1
// AMDGCN:#define cl_khr_fp64 1
// AMDGPU:#define cl_khr_global_int32_base_atomics 1
// AMDGPU:#define cl_khr_global_int32_extended_atomics 1
// AMDGPU:#define cl_khr_local_int32_base_atomics 1
// AMDGPU:#define cl_khr_local_int32_extended_atomics 1

// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=sparc-none-none < /dev/null | FileCheck -match-full-lines -check-prefix SPARC -check-prefix SPARC-DEFAULT %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=sparc-rtems-elf < /dev/null | FileCheck -match-full-lines -check-prefix SPARC -check-prefix SPARC-DEFAULT %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=sparc-none-netbsd < /dev/null | FileCheck -match-full-lines -check-prefix SPARC -check-prefix SPARC-NETOPENBSD %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=sparc-none-openbsd < /dev/null | FileCheck -match-full-lines -check-prefix SPARC -check-prefix SPARC-NETOPENBSD %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=sparc-none-none < /dev/null | FileCheck -match-full-lines -check-prefix SPARC -check-prefix SPARC-DEFAULT -check-prefix SPARC-DEFAULT-CXX %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=sparc-none-openbsd < /dev/null | FileCheck -match-full-lines -check-prefix SPARC -check-prefix SPARC-NETOPENBSD -check-prefix SPARC-NETOPENBSD-CXX %s
//
// SPARC-NOT:#define _LP64
// SPARC:#define __BIGGEST_ALIGNMENT__ 8
// SPARC:#define __BIG_ENDIAN__ 1
// SPARC:#define __BYTE_ORDER__ __ORDER_BIG_ENDIAN__
// SPARC:#define __CHAR16_TYPE__ unsigned short
// SPARC:#define __CHAR32_TYPE__ unsigned int
// SPARC:#define __CHAR_BIT__ 8
// SPARC:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// SPARC:#define __DBL_DIG__ 15
// SPARC:#define __DBL_EPSILON__ 2.2204460492503131e-16
// SPARC:#define __DBL_HAS_DENORM__ 1
// SPARC:#define __DBL_HAS_INFINITY__ 1
// SPARC:#define __DBL_HAS_QUIET_NAN__ 1
// SPARC:#define __DBL_MANT_DIG__ 53
// SPARC:#define __DBL_MAX_10_EXP__ 308
// SPARC:#define __DBL_MAX_EXP__ 1024
// SPARC:#define __DBL_MAX__ 1.7976931348623157e+308
// SPARC:#define __DBL_MIN_10_EXP__ (-307)
// SPARC:#define __DBL_MIN_EXP__ (-1021)
// SPARC:#define __DBL_MIN__ 2.2250738585072014e-308
// SPARC:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// SPARC:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// SPARC:#define __FLT_DIG__ 6
// SPARC:#define __FLT_EPSILON__ 1.19209290e-7F
// SPARC:#define __FLT_EVAL_METHOD__ 0
// SPARC:#define __FLT_HAS_DENORM__ 1
// SPARC:#define __FLT_HAS_INFINITY__ 1
// SPARC:#define __FLT_HAS_QUIET_NAN__ 1
// SPARC:#define __FLT_MANT_DIG__ 24
// SPARC:#define __FLT_MAX_10_EXP__ 38
// SPARC:#define __FLT_MAX_EXP__ 128
// SPARC:#define __FLT_MAX__ 3.40282347e+38F
// SPARC:#define __FLT_MIN_10_EXP__ (-37)
// SPARC:#define __FLT_MIN_EXP__ (-125)
// SPARC:#define __FLT_MIN__ 1.17549435e-38F
// SPARC:#define __FLT_RADIX__ 2
// SPARC:#define __GCC_ATOMIC_LLONG_LOCK_FREE 1
// SPARC:#define __INT16_C_SUFFIX__
// SPARC:#define __INT16_FMTd__ "hd"
// SPARC:#define __INT16_FMTi__ "hi"
// SPARC:#define __INT16_MAX__ 32767
// SPARC:#define __INT16_TYPE__ short
// SPARC:#define __INT32_C_SUFFIX__
// SPARC:#define __INT32_FMTd__ "d"
// SPARC:#define __INT32_FMTi__ "i"
// SPARC:#define __INT32_MAX__ 2147483647
// SPARC:#define __INT32_TYPE__ int
// SPARC:#define __INT64_C_SUFFIX__ LL
// SPARC:#define __INT64_FMTd__ "lld"
// SPARC:#define __INT64_FMTi__ "lli"
// SPARC:#define __INT64_MAX__ 9223372036854775807LL
// SPARC:#define __INT64_TYPE__ long long int
// SPARC:#define __INT8_C_SUFFIX__
// SPARC:#define __INT8_FMTd__ "hhd"
// SPARC:#define __INT8_FMTi__ "hhi"
// SPARC:#define __INT8_MAX__ 127
// SPARC:#define __INT8_TYPE__ signed char
// SPARC:#define __INTMAX_C_SUFFIX__ LL
// SPARC:#define __INTMAX_FMTd__ "lld"
// SPARC:#define __INTMAX_FMTi__ "lli"
// SPARC:#define __INTMAX_MAX__ 9223372036854775807LL
// SPARC:#define __INTMAX_TYPE__ long long int
// SPARC:#define __INTMAX_WIDTH__ 64
// SPARC-DEFAULT:#define __INTPTR_FMTd__ "d"
// SPARC-DEFAULT:#define __INTPTR_FMTi__ "i"
// SPARC-DEFAULT:#define __INTPTR_MAX__ 2147483647
// SPARC-DEFAULT:#define __INTPTR_TYPE__ int
// SPARC-NETOPENBSD:#define __INTPTR_FMTd__ "ld"
// SPARC-NETOPENBSD:#define __INTPTR_FMTi__ "li"
// SPARC-NETOPENBSD:#define __INTPTR_MAX__ 2147483647L
// SPARC-NETOPENBSD:#define __INTPTR_TYPE__ long int
// SPARC:#define __INTPTR_WIDTH__ 32
// SPARC:#define __INT_FAST16_FMTd__ "hd"
// SPARC:#define __INT_FAST16_FMTi__ "hi"
// SPARC:#define __INT_FAST16_MAX__ 32767
// SPARC:#define __INT_FAST16_TYPE__ short
// SPARC:#define __INT_FAST32_FMTd__ "d"
// SPARC:#define __INT_FAST32_FMTi__ "i"
// SPARC:#define __INT_FAST32_MAX__ 2147483647
// SPARC:#define __INT_FAST32_TYPE__ int
// SPARC:#define __INT_FAST64_FMTd__ "lld"
// SPARC:#define __INT_FAST64_FMTi__ "lli"
// SPARC:#define __INT_FAST64_MAX__ 9223372036854775807LL
// SPARC:#define __INT_FAST64_TYPE__ long long int
// SPARC:#define __INT_FAST8_FMTd__ "hhd"
// SPARC:#define __INT_FAST8_FMTi__ "hhi"
// SPARC:#define __INT_FAST8_MAX__ 127
// SPARC:#define __INT_FAST8_TYPE__ signed char
// SPARC:#define __INT_LEAST16_FMTd__ "hd"
// SPARC:#define __INT_LEAST16_FMTi__ "hi"
// SPARC:#define __INT_LEAST16_MAX__ 32767
// SPARC:#define __INT_LEAST16_TYPE__ short
// SPARC:#define __INT_LEAST32_FMTd__ "d"
// SPARC:#define __INT_LEAST32_FMTi__ "i"
// SPARC:#define __INT_LEAST32_MAX__ 2147483647
// SPARC:#define __INT_LEAST32_TYPE__ int
// SPARC:#define __INT_LEAST64_FMTd__ "lld"
// SPARC:#define __INT_LEAST64_FMTi__ "lli"
// SPARC:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// SPARC:#define __INT_LEAST64_TYPE__ long long int
// SPARC:#define __INT_LEAST8_FMTd__ "hhd"
// SPARC:#define __INT_LEAST8_FMTi__ "hhi"
// SPARC:#define __INT_LEAST8_MAX__ 127
// SPARC:#define __INT_LEAST8_TYPE__ signed char
// SPARC:#define __INT_MAX__ 2147483647
// SPARC:#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324L
// SPARC:#define __LDBL_DIG__ 15
// SPARC:#define __LDBL_EPSILON__ 2.2204460492503131e-16L
// SPARC:#define __LDBL_HAS_DENORM__ 1
// SPARC:#define __LDBL_HAS_INFINITY__ 1
// SPARC:#define __LDBL_HAS_QUIET_NAN__ 1
// SPARC:#define __LDBL_MANT_DIG__ 53
// SPARC:#define __LDBL_MAX_10_EXP__ 308
// SPARC:#define __LDBL_MAX_EXP__ 1024
// SPARC:#define __LDBL_MAX__ 1.7976931348623157e+308L
// SPARC:#define __LDBL_MIN_10_EXP__ (-307)
// SPARC:#define __LDBL_MIN_EXP__ (-1021)
// SPARC:#define __LDBL_MIN__ 2.2250738585072014e-308L
// SPARC:#define __LONG_LONG_MAX__ 9223372036854775807LL
// SPARC:#define __LONG_MAX__ 2147483647L
// SPARC-NOT:#define __LP64__
// SPARC:#define __POINTER_WIDTH__ 32
// SPARC-DEFAULT:#define __PTRDIFF_TYPE__ int
// SPARC-NETOPENBSD:#define __PTRDIFF_TYPE__ long int
// SPARC:#define __PTRDIFF_WIDTH__ 32
// SPARC:#define __REGISTER_PREFIX__
// SPARC:#define __SCHAR_MAX__ 127
// SPARC:#define __SHRT_MAX__ 32767
// SPARC:#define __SIG_ATOMIC_MAX__ 2147483647
// SPARC:#define __SIG_ATOMIC_WIDTH__ 32
// SPARC:#define __SIZEOF_DOUBLE__ 8
// SPARC:#define __SIZEOF_FLOAT__ 4
// SPARC:#define __SIZEOF_INT__ 4
// SPARC:#define __SIZEOF_LONG_DOUBLE__ 8
// SPARC:#define __SIZEOF_LONG_LONG__ 8
// SPARC:#define __SIZEOF_LONG__ 4
// SPARC:#define __SIZEOF_POINTER__ 4
// SPARC:#define __SIZEOF_PTRDIFF_T__ 4
// SPARC:#define __SIZEOF_SHORT__ 2
// SPARC:#define __SIZEOF_SIZE_T__ 4
// SPARC:#define __SIZEOF_WCHAR_T__ 4
// SPARC:#define __SIZEOF_WINT_T__ 4
// SPARC-DEFAULT:#define __SIZE_MAX__ 4294967295U
// SPARC-DEFAULT:#define __SIZE_TYPE__ unsigned int
// SPARC-NETOPENBSD:#define __SIZE_MAX__ 4294967295UL
// SPARC-NETOPENBSD:#define __SIZE_TYPE__ long unsigned int
// SPARC:#define __SIZE_WIDTH__ 32
// SPARC-DEFAULT-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 8U
// SPARC-NETOPENBSD-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 8UL
// SPARC:#define __UINT16_C_SUFFIX__
// SPARC:#define __UINT16_MAX__ 65535
// SPARC:#define __UINT16_TYPE__ unsigned short
// SPARC:#define __UINT32_C_SUFFIX__ U
// SPARC:#define __UINT32_MAX__ 4294967295U
// SPARC:#define __UINT32_TYPE__ unsigned int
// SPARC:#define __UINT64_C_SUFFIX__ ULL
// SPARC:#define __UINT64_MAX__ 18446744073709551615ULL
// SPARC:#define __UINT64_TYPE__ long long unsigned int
// SPARC:#define __UINT8_C_SUFFIX__
// SPARC:#define __UINT8_MAX__ 255
// SPARC:#define __UINT8_TYPE__ unsigned char
// SPARC:#define __UINTMAX_C_SUFFIX__ ULL
// SPARC:#define __UINTMAX_MAX__ 18446744073709551615ULL
// SPARC:#define __UINTMAX_TYPE__ long long unsigned int
// SPARC:#define __UINTMAX_WIDTH__ 64
// SPARC-DEFAULT:#define __UINTPTR_MAX__ 4294967295U
// SPARC-DEFAULT:#define __UINTPTR_TYPE__ unsigned int
// SPARC-NETOPENBSD:#define __UINTPTR_MAX__ 4294967295UL
// SPARC-NETOPENBSD:#define __UINTPTR_TYPE__ long unsigned int
// SPARC:#define __UINTPTR_WIDTH__ 32
// SPARC:#define __UINT_FAST16_MAX__ 65535
// SPARC:#define __UINT_FAST16_TYPE__ unsigned short
// SPARC:#define __UINT_FAST32_MAX__ 4294967295U
// SPARC:#define __UINT_FAST32_TYPE__ unsigned int
// SPARC:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// SPARC:#define __UINT_FAST64_TYPE__ long long unsigned int
// SPARC:#define __UINT_FAST8_MAX__ 255
// SPARC:#define __UINT_FAST8_TYPE__ unsigned char
// SPARC:#define __UINT_LEAST16_MAX__ 65535
// SPARC:#define __UINT_LEAST16_TYPE__ unsigned short
// SPARC:#define __UINT_LEAST32_MAX__ 4294967295U
// SPARC:#define __UINT_LEAST32_TYPE__ unsigned int
// SPARC:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// SPARC:#define __UINT_LEAST64_TYPE__ long long unsigned int
// SPARC:#define __UINT_LEAST8_MAX__ 255
// SPARC:#define __UINT_LEAST8_TYPE__ unsigned char
// SPARC:#define __USER_LABEL_PREFIX__
// SPARC:#define __VERSION__ "{{.*}}Clang{{.*}}
// SPARC:#define __WCHAR_MAX__ 2147483647
// SPARC:#define __WCHAR_TYPE__ int
// SPARC:#define __WCHAR_WIDTH__ 32
// SPARC:#define __WINT_TYPE__ int
// SPARC:#define __WINT_WIDTH__ 32
// SPARC:#define __sparc 1
// SPARC:#define __sparc__ 1
// SPARC:#define __sparcv8 1
// SPARC:#define sparc 1

// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=tce-none-none < /dev/null | FileCheck -match-full-lines -check-prefix TCE %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=tce-none-none < /dev/null | FileCheck -match-full-lines -check-prefix TCE -check-prefix TCE-CXX %s
//
// TCE-NOT:#define _LP64
// TCE:#define __BIGGEST_ALIGNMENT__ 4
// TCE:#define __BIG_ENDIAN__ 1
// TCE:#define __BYTE_ORDER__ __ORDER_BIG_ENDIAN__
// TCE:#define __CHAR16_TYPE__ unsigned short
// TCE:#define __CHAR32_TYPE__ unsigned int
// TCE:#define __CHAR_BIT__ 8
// TCE:#define __DBL_DENORM_MIN__ 1.40129846e-45
// TCE:#define __DBL_DIG__ 6
// TCE:#define __DBL_EPSILON__ 1.19209290e-7
// TCE:#define __DBL_HAS_DENORM__ 1
// TCE:#define __DBL_HAS_INFINITY__ 1
// TCE:#define __DBL_HAS_QUIET_NAN__ 1
// TCE:#define __DBL_MANT_DIG__ 24
// TCE:#define __DBL_MAX_10_EXP__ 38
// TCE:#define __DBL_MAX_EXP__ 128
// TCE:#define __DBL_MAX__ 3.40282347e+38
// TCE:#define __DBL_MIN_10_EXP__ (-37)
// TCE:#define __DBL_MIN_EXP__ (-125)
// TCE:#define __DBL_MIN__ 1.17549435e-38
// TCE:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// TCE:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// TCE:#define __FLT_DIG__ 6
// TCE:#define __FLT_EPSILON__ 1.19209290e-7F
// TCE:#define __FLT_EVAL_METHOD__ 0
// TCE:#define __FLT_HAS_DENORM__ 1
// TCE:#define __FLT_HAS_INFINITY__ 1
// TCE:#define __FLT_HAS_QUIET_NAN__ 1
// TCE:#define __FLT_MANT_DIG__ 24
// TCE:#define __FLT_MAX_10_EXP__ 38
// TCE:#define __FLT_MAX_EXP__ 128
// TCE:#define __FLT_MAX__ 3.40282347e+38F
// TCE:#define __FLT_MIN_10_EXP__ (-37)
// TCE:#define __FLT_MIN_EXP__ (-125)
// TCE:#define __FLT_MIN__ 1.17549435e-38F
// TCE:#define __FLT_RADIX__ 2
// TCE:#define __INT16_C_SUFFIX__
// TCE:#define __INT16_FMTd__ "hd"
// TCE:#define __INT16_FMTi__ "hi"
// TCE:#define __INT16_MAX__ 32767
// TCE:#define __INT16_TYPE__ short
// TCE:#define __INT32_C_SUFFIX__
// TCE:#define __INT32_FMTd__ "d"
// TCE:#define __INT32_FMTi__ "i"
// TCE:#define __INT32_MAX__ 2147483647
// TCE:#define __INT32_TYPE__ int
// TCE:#define __INT8_C_SUFFIX__
// TCE:#define __INT8_FMTd__ "hhd"
// TCE:#define __INT8_FMTi__ "hhi"
// TCE:#define __INT8_MAX__ 127
// TCE:#define __INT8_TYPE__ signed char
// TCE:#define __INTMAX_C_SUFFIX__ L
// TCE:#define __INTMAX_FMTd__ "ld"
// TCE:#define __INTMAX_FMTi__ "li"
// TCE:#define __INTMAX_MAX__ 2147483647L
// TCE:#define __INTMAX_TYPE__ long int
// TCE:#define __INTMAX_WIDTH__ 32
// TCE:#define __INTPTR_FMTd__ "d"
// TCE:#define __INTPTR_FMTi__ "i"
// TCE:#define __INTPTR_MAX__ 2147483647
// TCE:#define __INTPTR_TYPE__ int
// TCE:#define __INTPTR_WIDTH__ 32
// TCE:#define __INT_FAST16_FMTd__ "hd"
// TCE:#define __INT_FAST16_FMTi__ "hi"
// TCE:#define __INT_FAST16_MAX__ 32767
// TCE:#define __INT_FAST16_TYPE__ short
// TCE:#define __INT_FAST32_FMTd__ "d"
// TCE:#define __INT_FAST32_FMTi__ "i"
// TCE:#define __INT_FAST32_MAX__ 2147483647
// TCE:#define __INT_FAST32_TYPE__ int
// TCE:#define __INT_FAST8_FMTd__ "hhd"
// TCE:#define __INT_FAST8_FMTi__ "hhi"
// TCE:#define __INT_FAST8_MAX__ 127
// TCE:#define __INT_FAST8_TYPE__ signed char
// TCE:#define __INT_LEAST16_FMTd__ "hd"
// TCE:#define __INT_LEAST16_FMTi__ "hi"
// TCE:#define __INT_LEAST16_MAX__ 32767
// TCE:#define __INT_LEAST16_TYPE__ short
// TCE:#define __INT_LEAST32_FMTd__ "d"
// TCE:#define __INT_LEAST32_FMTi__ "i"
// TCE:#define __INT_LEAST32_MAX__ 2147483647
// TCE:#define __INT_LEAST32_TYPE__ int
// TCE:#define __INT_LEAST8_FMTd__ "hhd"
// TCE:#define __INT_LEAST8_FMTi__ "hhi"
// TCE:#define __INT_LEAST8_MAX__ 127
// TCE:#define __INT_LEAST8_TYPE__ signed char
// TCE:#define __INT_MAX__ 2147483647
// TCE:#define __LDBL_DENORM_MIN__ 1.40129846e-45L
// TCE:#define __LDBL_DIG__ 6
// TCE:#define __LDBL_EPSILON__ 1.19209290e-7L
// TCE:#define __LDBL_HAS_DENORM__ 1
// TCE:#define __LDBL_HAS_INFINITY__ 1
// TCE:#define __LDBL_HAS_QUIET_NAN__ 1
// TCE:#define __LDBL_MANT_DIG__ 24
// TCE:#define __LDBL_MAX_10_EXP__ 38
// TCE:#define __LDBL_MAX_EXP__ 128
// TCE:#define __LDBL_MAX__ 3.40282347e+38L
// TCE:#define __LDBL_MIN_10_EXP__ (-37)
// TCE:#define __LDBL_MIN_EXP__ (-125)
// TCE:#define __LDBL_MIN__ 1.17549435e-38L
// TCE:#define __LONG_LONG_MAX__ 2147483647LL
// TCE:#define __LONG_MAX__ 2147483647L
// TCE-NOT:#define __LP64__
// TCE:#define __POINTER_WIDTH__ 32
// TCE:#define __PTRDIFF_TYPE__ int
// TCE:#define __PTRDIFF_WIDTH__ 32
// TCE:#define __SCHAR_MAX__ 127
// TCE:#define __SHRT_MAX__ 32767
// TCE:#define __SIG_ATOMIC_MAX__ 2147483647
// TCE:#define __SIG_ATOMIC_WIDTH__ 32
// TCE:#define __SIZEOF_DOUBLE__ 4
// TCE:#define __SIZEOF_FLOAT__ 4
// TCE:#define __SIZEOF_INT__ 4
// TCE:#define __SIZEOF_LONG_DOUBLE__ 4
// TCE:#define __SIZEOF_LONG_LONG__ 4
// TCE:#define __SIZEOF_LONG__ 4
// TCE:#define __SIZEOF_POINTER__ 4
// TCE:#define __SIZEOF_PTRDIFF_T__ 4
// TCE:#define __SIZEOF_SHORT__ 2
// TCE:#define __SIZEOF_SIZE_T__ 4
// TCE:#define __SIZEOF_WCHAR_T__ 4
// TCE:#define __SIZEOF_WINT_T__ 4
// TCE:#define __SIZE_MAX__ 4294967295U
// TCE:#define __SIZE_TYPE__ unsigned int
// TCE:#define __SIZE_WIDTH__ 32
// TCE-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 4U
// TCE:#define __TCE_V1__ 1
// TCE:#define __TCE__ 1
// TCE:#define __UINT16_C_SUFFIX__
// TCE:#define __UINT16_MAX__ 65535
// TCE:#define __UINT16_TYPE__ unsigned short
// TCE:#define __UINT32_C_SUFFIX__ U
// TCE:#define __UINT32_MAX__ 4294967295U
// TCE:#define __UINT32_TYPE__ unsigned int
// TCE:#define __UINT8_C_SUFFIX__
// TCE:#define __UINT8_MAX__ 255
// TCE:#define __UINT8_TYPE__ unsigned char
// TCE:#define __UINTMAX_C_SUFFIX__ UL
// TCE:#define __UINTMAX_MAX__ 4294967295UL
// TCE:#define __UINTMAX_TYPE__ long unsigned int
// TCE:#define __UINTMAX_WIDTH__ 32
// TCE:#define __UINTPTR_MAX__ 4294967295U
// TCE:#define __UINTPTR_TYPE__ unsigned int
// TCE:#define __UINTPTR_WIDTH__ 32
// TCE:#define __UINT_FAST16_MAX__ 65535
// TCE:#define __UINT_FAST16_TYPE__ unsigned short
// TCE:#define __UINT_FAST32_MAX__ 4294967295U
// TCE:#define __UINT_FAST32_TYPE__ unsigned int
// TCE:#define __UINT_FAST8_MAX__ 255
// TCE:#define __UINT_FAST8_TYPE__ unsigned char
// TCE:#define __UINT_LEAST16_MAX__ 65535
// TCE:#define __UINT_LEAST16_TYPE__ unsigned short
// TCE:#define __UINT_LEAST32_MAX__ 4294967295U
// TCE:#define __UINT_LEAST32_TYPE__ unsigned int
// TCE:#define __UINT_LEAST8_MAX__ 255
// TCE:#define __UINT_LEAST8_TYPE__ unsigned char
// TCE:#define __USER_LABEL_PREFIX__
// TCE:#define __WCHAR_MAX__ 2147483647
// TCE:#define __WCHAR_TYPE__ int
// TCE:#define __WCHAR_WIDTH__ 32
// TCE:#define __WINT_TYPE__ int
// TCE:#define __WINT_WIDTH__ 32
// TCE:#define __tce 1
// TCE:#define __tce__ 1
// TCE:#define tce 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=x86_64-scei-ps4 < /dev/null | FileCheck -match-full-lines -check-prefix PS4 %s
//
// PS4:#define _LP64 1
// PS4:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// PS4:#define __CHAR16_TYPE__ unsigned short
// PS4:#define __CHAR32_TYPE__ unsigned int
// PS4:#define __CHAR_BIT__ 8
// PS4:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// PS4:#define __DBL_DIG__ 15
// PS4:#define __DBL_EPSILON__ 2.2204460492503131e-16
// PS4:#define __DBL_HAS_DENORM__ 1
// PS4:#define __DBL_HAS_INFINITY__ 1
// PS4:#define __DBL_HAS_QUIET_NAN__ 1
// PS4:#define __DBL_MANT_DIG__ 53
// PS4:#define __DBL_MAX_10_EXP__ 308
// PS4:#define __DBL_MAX_EXP__ 1024
// PS4:#define __DBL_MAX__ 1.7976931348623157e+308
// PS4:#define __DBL_MIN_10_EXP__ (-307)
// PS4:#define __DBL_MIN_EXP__ (-1021)
// PS4:#define __DBL_MIN__ 2.2250738585072014e-308
// PS4:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// PS4:#define __ELF__ 1
// PS4:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// PS4:#define __FLT_DIG__ 6
// PS4:#define __FLT_EPSILON__ 1.19209290e-7F
// PS4:#define __FLT_EVAL_METHOD__ 0
// PS4:#define __FLT_HAS_DENORM__ 1
// PS4:#define __FLT_HAS_INFINITY__ 1
// PS4:#define __FLT_HAS_QUIET_NAN__ 1
// PS4:#define __FLT_MANT_DIG__ 24
// PS4:#define __FLT_MAX_10_EXP__ 38
// PS4:#define __FLT_MAX_EXP__ 128
// PS4:#define __FLT_MAX__ 3.40282347e+38F
// PS4:#define __FLT_MIN_10_EXP__ (-37)
// PS4:#define __FLT_MIN_EXP__ (-125)
// PS4:#define __FLT_MIN__ 1.17549435e-38F
// PS4:#define __FLT_RADIX__ 2
// PS4:#define __FreeBSD__ 9
// PS4:#define __FreeBSD_cc_version 900001
// PS4:#define __INT16_TYPE__ short
// PS4:#define __INT32_TYPE__ int
// PS4:#define __INT64_C_SUFFIX__ L
// PS4:#define __INT64_TYPE__ long int
// PS4:#define __INT8_TYPE__ signed char
// PS4:#define __INTMAX_MAX__ 9223372036854775807L
// PS4:#define __INTMAX_TYPE__ long int
// PS4:#define __INTMAX_WIDTH__ 64
// PS4:#define __INTPTR_TYPE__ long int
// PS4:#define __INTPTR_WIDTH__ 64
// PS4:#define __INT_MAX__ 2147483647
// PS4:#define __KPRINTF_ATTRIBUTE__ 1
// PS4:#define __LDBL_DENORM_MIN__ 3.64519953188247460253e-4951L
// PS4:#define __LDBL_DIG__ 18
// PS4:#define __LDBL_EPSILON__ 1.08420217248550443401e-19L
// PS4:#define __LDBL_HAS_DENORM__ 1
// PS4:#define __LDBL_HAS_INFINITY__ 1
// PS4:#define __LDBL_HAS_QUIET_NAN__ 1
// PS4:#define __LDBL_MANT_DIG__ 64
// PS4:#define __LDBL_MAX_10_EXP__ 4932
// PS4:#define __LDBL_MAX_EXP__ 16384
// PS4:#define __LDBL_MAX__ 1.18973149535723176502e+4932L
// PS4:#define __LDBL_MIN_10_EXP__ (-4931)
// PS4:#define __LDBL_MIN_EXP__ (-16381)
// PS4:#define __LDBL_MIN__ 3.36210314311209350626e-4932L
// PS4:#define __LITTLE_ENDIAN__ 1
// PS4:#define __LONG_LONG_MAX__ 9223372036854775807LL
// PS4:#define __LONG_MAX__ 9223372036854775807L
// PS4:#define __LP64__ 1
// PS4:#define __MMX__ 1
// PS4:#define __NO_MATH_INLINES 1
// PS4:#define __ORBIS__ 1
// PS4:#define __POINTER_WIDTH__ 64
// PS4:#define __PTRDIFF_MAX__ 9223372036854775807L
// PS4:#define __PTRDIFF_TYPE__ long int
// PS4:#define __PTRDIFF_WIDTH__ 64
// PS4:#define __REGISTER_PREFIX__
// PS4:#define __SCE__ 1
// PS4:#define __SCHAR_MAX__ 127
// PS4:#define __SHRT_MAX__ 32767
// PS4:#define __SIG_ATOMIC_MAX__ 2147483647
// PS4:#define __SIG_ATOMIC_WIDTH__ 32
// PS4:#define __SIZEOF_DOUBLE__ 8
// PS4:#define __SIZEOF_FLOAT__ 4
// PS4:#define __SIZEOF_INT__ 4
// PS4:#define __SIZEOF_LONG_DOUBLE__ 16
// PS4:#define __SIZEOF_LONG_LONG__ 8
// PS4:#define __SIZEOF_LONG__ 8
// PS4:#define __SIZEOF_POINTER__ 8
// PS4:#define __SIZEOF_PTRDIFF_T__ 8
// PS4:#define __SIZEOF_SHORT__ 2
// PS4:#define __SIZEOF_SIZE_T__ 8
// PS4:#define __SIZEOF_WCHAR_T__ 2
// PS4:#define __SIZEOF_WINT_T__ 4
// PS4:#define __SIZE_TYPE__ long unsigned int
// PS4:#define __SIZE_WIDTH__ 64
// PS4:#define __SSE2_MATH__ 1
// PS4:#define __SSE2__ 1
// PS4:#define __SSE_MATH__ 1
// PS4:#define __SSE__ 1
// PS4:#define __STDC_VERSION__ 199901L
// PS4:#define __UINTMAX_TYPE__ long unsigned int
// PS4:#define __USER_LABEL_PREFIX__
// PS4:#define __WCHAR_MAX__ 65535
// PS4:#define __WCHAR_TYPE__ unsigned short
// PS4:#define __WCHAR_UNSIGNED__ 1
// PS4:#define __WCHAR_WIDTH__ 16
// PS4:#define __WINT_TYPE__ int
// PS4:#define __WINT_WIDTH__ 32
// PS4:#define __amd64 1
// PS4:#define __amd64__ 1
// PS4:#define __unix 1
// PS4:#define __unix__ 1
// PS4:#define __x86_64 1
// PS4:#define __x86_64__ 1
// PS4:#define unix 1
//
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=x86_64-scei-ps4 < /dev/null | FileCheck -match-full-lines -check-prefix PS4-CXX %s
// PS4-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 32UL
//
// RUN: %clang_cc1 -E -dM -triple=x86_64-pc-mingw32 < /dev/null | FileCheck -match-full-lines -check-prefix X86-64-DECLSPEC %s
// RUN: %clang_cc1 -E -dM -fms-extensions -triple=x86_64-unknown-mingw32 < /dev/null | FileCheck -match-full-lines -check-prefix X86-64-DECLSPEC %s
// X86-64-DECLSPEC: #define __declspec{{.*}}
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=sparc64-none-none < /dev/null | FileCheck -match-full-lines -check-prefix SPARCV9 %s
// SPARCV9:#define __BIGGEST_ALIGNMENT__ 16
// SPARCV9:#define __INT64_TYPE__ long int
// SPARCV9:#define __INTMAX_C_SUFFIX__ L
// SPARCV9:#define __INTMAX_TYPE__ long int
// SPARCV9:#define __INTPTR_TYPE__ long int
// SPARCV9:#define __LONG_MAX__ 9223372036854775807L
// SPARCV9:#define __LP64__ 1
// SPARCV9:#define __SIZEOF_LONG__ 8
// SPARCV9:#define __SIZEOF_POINTER__ 8
// SPARCV9:#define __UINTPTR_TYPE__ long unsigned int
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=sparc64-none-openbsd < /dev/null | FileCheck -match-full-lines -check-prefix SPARC64-OBSD %s
// SPARC64-OBSD:#define __INT64_TYPE__ long long int
// SPARC64-OBSD:#define __INTMAX_C_SUFFIX__ LL
// SPARC64-OBSD:#define __INTMAX_TYPE__ long long int
// SPARC64-OBSD:#define __UINTMAX_C_SUFFIX__ ULL
// SPARC64-OBSD:#define __UINTMAX_TYPE__ long long unsigned int
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=x86_64-pc-kfreebsd-gnu < /dev/null | FileCheck -match-full-lines -check-prefix KFREEBSD-DEFINE %s
// KFREEBSD-DEFINE:#define __FreeBSD_kernel__ 1
// KFREEBSD-DEFINE:#define __GLIBC__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=i686-pc-kfreebsd-gnu < /dev/null | FileCheck -match-full-lines -check-prefix KFREEBSDI686-DEFINE %s
// KFREEBSDI686-DEFINE:#define __FreeBSD_kernel__ 1
// KFREEBSDI686-DEFINE:#define __GLIBC__ 1
//
// RUN: %clang_cc1 -x c++ -triple i686-pc-linux-gnu -fobjc-runtime=gcc -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix GNUSOURCE %s
// RUN: %clang_cc1 -x c++ -triple sparc-rtems-elf -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix GNUSOURCE %s
// GNUSOURCE:#define _GNU_SOURCE 1
//
// Check that the GNUstep Objective-C ABI defines exist and are clamped at the
// highest supported version.
// RUN: %clang_cc1 -x objective-c -triple i386-unknown-freebsd -fobjc-runtime=gnustep-1.9 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix GNUSTEP1 %s
// GNUSTEP1:#define __OBJC_GNUSTEP_RUNTIME_ABI__ 18
// RUN: %clang_cc1 -x objective-c -triple i386-unknown-freebsd -fobjc-runtime=gnustep-2.5 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix GNUSTEP2 %s
// GNUSTEP2:#define __OBJC_GNUSTEP_RUNTIME_ABI__ 20
//
// RUN: %clang_cc1 -x c++ -fgnuc-version=4.2.1 -std=c++98 -fno-rtti -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix NORTTI %s
// NORTTI: #define __GXX_ABI_VERSION {{.*}}
// NORTTI-NOT:#define __GXX_RTTI
// NORTTI:#define __STDC__ 1
//
// RUN: %clang_cc1 -triple arm-linux-androideabi -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix ANDROID %s
// ANDROID-NOT:#define __ANDROID_API__
// ANDROID-NOT:#define __ANDROID_MIN_SDK_VERSION__
// ANDROID:#define __ANDROID__ 1
// ANDROID-NOT:#define __gnu_linux__
//
// RUN: %clang_cc1 -x c++ -triple i686-linux-android -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix I386-ANDROID-CXX %s
// I386-ANDROID-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 8U
//
// RUN: %clang_cc1 -x c++ -triple x86_64-linux-android -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix X86_64-ANDROID-CXX %s
// X86_64-ANDROID-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 16UL
//
// RUN: %clang_cc1 -triple arm-linux-androideabi20 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix ANDROID20 %s
// ANDROID20:#define __ANDROID_API__ __ANDROID_MIN_SDK_VERSION__
// ANDROID20:#define __ANDROID_MIN_SDK_VERSION__ 20
// ANDROID20:#define __ANDROID__ 1
// ANDROID-NOT:#define __gnu_linux__
//
// RUN: %clang_cc1 -triple lanai-unknown-unknown -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix LANAI %s
// LANAI: #define __lanai__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=amd64-unknown-openbsd6.1 < /dev/null | FileCheck -match-full-lines -check-prefix OPENBSD %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=aarch64-unknown-openbsd6.1 < /dev/null | FileCheck -match-full-lines -check-prefix OPENBSD %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=arm-unknown-openbsd6.1-gnueabi < /dev/null | FileCheck -match-full-lines -check-prefix OPENBSD %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=i386-unknown-openbsd6.1 < /dev/null | FileCheck -match-full-lines -check-prefix OPENBSD %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-unknown-openbsd6.1 < /dev/null | FileCheck -match-full-lines -check-prefix OPENBSD %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-unknown-openbsd6.1 < /dev/null | FileCheck -match-full-lines -check-prefix OPENBSD %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64le-unknown-openbsd6.1 < /dev/null | FileCheck -match-full-lines -check-prefix OPENBSD %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips64-unknown-openbsd6.1 < /dev/null | FileCheck -match-full-lines -check-prefix OPENBSD %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips64el-unknown-openbsd6.1 < /dev/null | FileCheck -match-full-lines -check-prefix OPENBSD %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=sparc64-unknown-openbsd6.1 < /dev/null | FileCheck -match-full-lines -check-prefix OPENBSD %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=riscv64-unknown-openbsd6.1 < /dev/null | FileCheck -match-full-lines -check-prefix OPENBSD %s
// OPENBSD:#define __ELF__ 1
// OPENBSD:#define __INT16_TYPE__ short
// OPENBSD:#define __INT32_TYPE__ int
// OPENBSD:#define __INT64_TYPE__ long long int
// OPENBSD:#define __INT8_TYPE__ signed char
// OPENBSD:#define __INTMAX_TYPE__ long long int
// OPENBSD:#define __INTPTR_TYPE__ long int
// OPENBSD:#define __OpenBSD__ 1
// OPENBSD:#define __PTRDIFF_TYPE__ long int
// OPENBSD:#define __SIZE_TYPE__ long unsigned int
// OPENBSD:#define __UINT16_TYPE__ unsigned short
// OPENBSD:#define __UINT32_TYPE__ unsigned int
// OPENBSD:#define __UINT64_TYPE__ long long unsigned int
// OPENBSD:#define __UINT8_TYPE__ unsigned char
// OPENBSD:#define __UINTMAX_TYPE__ long long unsigned int
// OPENBSD:#define __UINTPTR_TYPE__ long unsigned int
// OPENBSD:#define __WCHAR_TYPE__ int
// OPENBSD:#define __WINT_TYPE__ int
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=xcore-none-none < /dev/null | FileCheck -match-full-lines -check-prefix XCORE %s
// XCORE:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// XCORE:#define __LITTLE_ENDIAN__ 1
// XCORE:#define __XS1B__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=wasm32-unknown-unknown \
// RUN:   < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefixes=WEBASSEMBLY,WEBASSEMBLY32 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=wasm64-unknown-unknown \
// RUN:   < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefixes=WEBASSEMBLY,WEBASSEMBLY64 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=wasm32-emscripten \
// RUN:   < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefixes=WEBASSEMBLY,WEBASSEMBLY32,EMSCRIPTEN %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=wasm32-emscripten -pthread -target-feature +atomics \
// RUN:   < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefixes=WEBASSEMBLY,WEBASSEMBLY32,EMSCRIPTEN,EMSCRIPTEN-THREADS %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=wasm64-emscripten \
// RUN:   < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefixes=WEBASSEMBLY,WEBASSEMBLY64,EMSCRIPTEN %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=wasm32-wasi \
// RUN:   < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefixes=WEBASSEMBLY,WEBASSEMBLY32,WEBASSEMBLY-WASI %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=wasm64-wasi \
// RUN:   < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefixes=WEBASSEMBLY,WEBASSEMBLY64,WEBASSEMBLY-WASI %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=wasm32-unknown-unknown -x c++ \
// RUN:   < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefixes=WEBASSEMBLY-CXX %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=wasm32-unknown-unknown -x c++ -pthread -target-feature +atomics \
// RUN:   < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefixes=WEBASSEMBLY-CXX-ATOMICS %s
//
// WEBASSEMBLY32:#define _ILP32 1
// WEBASSEMBLY32-NOT:#define _LP64
// WEBASSEMBLY64-NOT:#define _ILP32
// WEBASSEMBLY64:#define _LP64 1
// EMSCRIPTEN-THREADS:#define _REENTRANT 1
// WEBASSEMBLY-NEXT:#define __ATOMIC_ACQUIRE 2
// WEBASSEMBLY-NEXT:#define __ATOMIC_ACQ_REL 4
// WEBASSEMBLY-NEXT:#define __ATOMIC_CONSUME 1
// WEBASSEMBLY-NEXT:#define __ATOMIC_RELAXED 0
// WEBASSEMBLY-NEXT:#define __ATOMIC_RELEASE 3
// WEBASSEMBLY-NEXT:#define __ATOMIC_SEQ_CST 5
// WEBASSEMBLY-NEXT:#define __BIGGEST_ALIGNMENT__ 16
// WEBASSEMBLY-NEXT:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// WEBASSEMBLY-NEXT:#define __CHAR16_TYPE__ unsigned short
// WEBASSEMBLY-NEXT:#define __CHAR32_TYPE__ unsigned int
// WEBASSEMBLY-NEXT:#define __CHAR_BIT__ 8
// WEBASSEMBLY-NOT:#define __CHAR_UNSIGNED__
// WEBASSEMBLY-NEXT:#define __CLANG_ATOMIC_BOOL_LOCK_FREE 2
// WEBASSEMBLY-NEXT:#define __CLANG_ATOMIC_CHAR16_T_LOCK_FREE 2
// WEBASSEMBLY-NEXT:#define __CLANG_ATOMIC_CHAR32_T_LOCK_FREE 2
// WEBASSEMBLY-NEXT:#define __CLANG_ATOMIC_CHAR_LOCK_FREE 2
// WEBASSEMBLY-NEXT:#define __CLANG_ATOMIC_INT_LOCK_FREE 2
// WEBASSEMBLY-NEXT:#define __CLANG_ATOMIC_LLONG_LOCK_FREE 2
// WEBASSEMBLY-NEXT:#define __CLANG_ATOMIC_LONG_LOCK_FREE 2
// WEBASSEMBLY-NEXT:#define __CLANG_ATOMIC_POINTER_LOCK_FREE 2
// WEBASSEMBLY-NEXT:#define __CLANG_ATOMIC_SHORT_LOCK_FREE 2
// WEBASSEMBLY-NEXT:#define __CLANG_ATOMIC_WCHAR_T_LOCK_FREE 2
// WEBASSEMBLY-NEXT:#define __CONSTANT_CFSTRINGS__ 1
// WEBASSEMBLY-NEXT:#define __DBL_DECIMAL_DIG__ 17
// WEBASSEMBLY-NEXT:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// WEBASSEMBLY-NEXT:#define __DBL_DIG__ 15
// WEBASSEMBLY-NEXT:#define __DBL_EPSILON__ 2.2204460492503131e-16
// WEBASSEMBLY-NEXT:#define __DBL_HAS_DENORM__ 1
// WEBASSEMBLY-NEXT:#define __DBL_HAS_INFINITY__ 1
// WEBASSEMBLY-NEXT:#define __DBL_HAS_QUIET_NAN__ 1
// WEBASSEMBLY-NEXT:#define __DBL_MANT_DIG__ 53
// WEBASSEMBLY-NEXT:#define __DBL_MAX_10_EXP__ 308
// WEBASSEMBLY-NEXT:#define __DBL_MAX_EXP__ 1024
// WEBASSEMBLY-NEXT:#define __DBL_MAX__ 1.7976931348623157e+308
// WEBASSEMBLY-NEXT:#define __DBL_MIN_10_EXP__ (-307)
// WEBASSEMBLY-NEXT:#define __DBL_MIN_EXP__ (-1021)
// WEBASSEMBLY-NEXT:#define __DBL_MIN__ 2.2250738585072014e-308
// WEBASSEMBLY-NEXT:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// WEBASSEMBLY-NOT:#define __ELF__
// EMSCRIPTEN-THREADS-NEXT:#define __EMSCRIPTEN_PTHREADS__ 1
// EMSCRIPTEN-NEXT:#define __EMSCRIPTEN__ 1
// WEBASSEMBLY-NEXT:#define __FINITE_MATH_ONLY__ 0
// WEBASSEMBLY-NEXT:#define __FLOAT128__ 1
// WEBASSEMBLY-NOT:#define __FLT16_DECIMAL_DIG__
// WEBASSEMBLY-NOT:#define __FLT16_DENORM_MIN__
// WEBASSEMBLY-NOT:#define __FLT16_DIG__
// WEBASSEMBLY-NOT:#define __FLT16_EPSILON__
// WEBASSEMBLY-NOT:#define __FLT16_HAS_DENORM__
// WEBASSEMBLY-NOT:#define __FLT16_HAS_INFINITY__
// WEBASSEMBLY-NOT:#define __FLT16_HAS_QUIET_NAN__
// WEBASSEMBLY-NOT:#define __FLT16_MANT_DIG__
// WEBASSEMBLY-NOT:#define __FLT16_MAX_10_EXP__
// WEBASSEMBLY-NOT:#define __FLT16_MAX_EXP__
// WEBASSEMBLY-NOT:#define __FLT16_MAX__
// WEBASSEMBLY-NOT:#define __FLT16_MIN_10_EXP__
// WEBASSEMBLY-NOT:#define __FLT16_MIN_EXP__
// WEBASSEMBLY-NOT:#define __FLT16_MIN__
// WEBASSEMBLY-NEXT:#define __FLT_DECIMAL_DIG__ 9
// WEBASSEMBLY-NEXT:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// WEBASSEMBLY-NEXT:#define __FLT_DIG__ 6
// WEBASSEMBLY-NEXT:#define __FLT_EPSILON__ 1.19209290e-7F
// WEBASSEMBLY-NEXT:#define __FLT_EVAL_METHOD__ 0
// WEBASSEMBLY-NEXT:#define __FLT_HAS_DENORM__ 1
// WEBASSEMBLY-NEXT:#define __FLT_HAS_INFINITY__ 1
// WEBASSEMBLY-NEXT:#define __FLT_HAS_QUIET_NAN__ 1
// WEBASSEMBLY-NEXT:#define __FLT_MANT_DIG__ 24
// WEBASSEMBLY-NEXT:#define __FLT_MAX_10_EXP__ 38
// WEBASSEMBLY-NEXT:#define __FLT_MAX_EXP__ 128
// WEBASSEMBLY-NEXT:#define __FLT_MAX__ 3.40282347e+38F
// WEBASSEMBLY-NEXT:#define __FLT_MIN_10_EXP__ (-37)
// WEBASSEMBLY-NEXT:#define __FLT_MIN_EXP__ (-125)
// WEBASSEMBLY-NEXT:#define __FLT_MIN__ 1.17549435e-38F
// WEBASSEMBLY-NEXT:#define __FLT_RADIX__ 2
// WEBASSEMBLY-NEXT:#define __GCC_ATOMIC_BOOL_LOCK_FREE 2
// WEBASSEMBLY-NEXT:#define __GCC_ATOMIC_CHAR16_T_LOCK_FREE 2
// WEBASSEMBLY-NEXT:#define __GCC_ATOMIC_CHAR32_T_LOCK_FREE 2
// WEBASSEMBLY-NEXT:#define __GCC_ATOMIC_CHAR_LOCK_FREE 2
// WEBASSEMBLY-NEXT:#define __GCC_ATOMIC_INT_LOCK_FREE 2
// WEBASSEMBLY-NEXT:#define __GCC_ATOMIC_LLONG_LOCK_FREE 2
// WEBASSEMBLY-NEXT:#define __GCC_ATOMIC_LONG_LOCK_FREE 2
// WEBASSEMBLY-NEXT:#define __GCC_ATOMIC_POINTER_LOCK_FREE 2
// WEBASSEMBLY-NEXT:#define __GCC_ATOMIC_SHORT_LOCK_FREE 2
// WEBASSEMBLY-NEXT:#define __GCC_ATOMIC_TEST_AND_SET_TRUEVAL 1
// WEBASSEMBLY-NEXT:#define __GCC_ATOMIC_WCHAR_T_LOCK_FREE 2
// WEBASSEMBLY-NEXT:#define __GNUC_MINOR__ {{.*}}
// WEBASSEMBLY-NEXT:#define __GNUC_PATCHLEVEL__ {{.*}}
// WEBASSEMBLY-NEXT:#define __GNUC_STDC_INLINE__ 1
// WEBASSEMBLY-NEXT:#define __GNUC__ {{.*}}
// WEBASSEMBLY-NEXT:#define __GXX_ABI_VERSION 1002
// WEBASSEMBLY32-NEXT:#define __ILP32__ 1
// WEBASSEMBLY64-NOT:#define __ILP32__
// WEBASSEMBLY-NEXT:#define __INT16_C_SUFFIX__
// WEBASSEMBLY-NEXT:#define __INT16_FMTd__ "hd"
// WEBASSEMBLY-NEXT:#define __INT16_FMTi__ "hi"
// WEBASSEMBLY-NEXT:#define __INT16_MAX__ 32767
// WEBASSEMBLY-NEXT:#define __INT16_TYPE__ short
// WEBASSEMBLY-NEXT:#define __INT32_C_SUFFIX__
// WEBASSEMBLY-NEXT:#define __INT32_FMTd__ "d"
// WEBASSEMBLY-NEXT:#define __INT32_FMTi__ "i"
// WEBASSEMBLY-NEXT:#define __INT32_MAX__ 2147483647
// WEBASSEMBLY-NEXT:#define __INT32_TYPE__ int
// WEBASSEMBLY-NEXT:#define __INT64_C_SUFFIX__ LL
// WEBASSEMBLY-NEXT:#define __INT64_FMTd__ "lld"
// WEBASSEMBLY-NEXT:#define __INT64_FMTi__ "lli"
// WEBASSEMBLY-NEXT:#define __INT64_MAX__ 9223372036854775807LL
// WEBASSEMBLY-NEXT:#define __INT64_TYPE__ long long int
// WEBASSEMBLY-NEXT:#define __INT8_C_SUFFIX__
// WEBASSEMBLY-NEXT:#define __INT8_FMTd__ "hhd"
// WEBASSEMBLY-NEXT:#define __INT8_FMTi__ "hhi"
// WEBASSEMBLY-NEXT:#define __INT8_MAX__ 127
// WEBASSEMBLY-NEXT:#define __INT8_TYPE__ signed char
// WEBASSEMBLY-NEXT:#define __INTMAX_C_SUFFIX__ LL
// WEBASSEMBLY-NEXT:#define __INTMAX_FMTd__ "lld"
// WEBASSEMBLY-NEXT:#define __INTMAX_FMTi__ "lli"
// WEBASSEMBLY-NEXT:#define __INTMAX_MAX__ 9223372036854775807LL
// WEBASSEMBLY-NEXT:#define __INTMAX_TYPE__ long long int
// WEBASSEMBLY-NEXT:#define __INTMAX_WIDTH__ 64
// WEBASSEMBLY-NEXT:#define __INTPTR_FMTd__ "ld"
// WEBASSEMBLY-NEXT:#define __INTPTR_FMTi__ "li"
// WEBASSEMBLY32-NEXT:#define __INTPTR_MAX__ 2147483647L
// WEBASSEMBLY64-NEXT:#define __INTPTR_MAX__ 9223372036854775807L
// WEBASSEMBLY-NEXT:#define __INTPTR_TYPE__ long int
// WEBASSEMBLY32-NEXT:#define __INTPTR_WIDTH__ 32
// WEBASSEMBLY64-NEXT:#define __INTPTR_WIDTH__ 64
// WEBASSEMBLY-NEXT:#define __INT_FAST16_FMTd__ "hd"
// WEBASSEMBLY-NEXT:#define __INT_FAST16_FMTi__ "hi"
// WEBASSEMBLY-NEXT:#define __INT_FAST16_MAX__ 32767
// WEBASSEMBLY-NEXT:#define __INT_FAST16_TYPE__ short
// WEBASSEMBLY-NEXT:#define __INT_FAST32_FMTd__ "d"
// WEBASSEMBLY-NEXT:#define __INT_FAST32_FMTi__ "i"
// WEBASSEMBLY-NEXT:#define __INT_FAST32_MAX__ 2147483647
// WEBASSEMBLY-NEXT:#define __INT_FAST32_TYPE__ int
// WEBASSEMBLY-NEXT:#define __INT_FAST64_FMTd__ "lld"
// WEBASSEMBLY-NEXT:#define __INT_FAST64_FMTi__ "lli"
// WEBASSEMBLY-NEXT:#define __INT_FAST64_MAX__ 9223372036854775807LL
// WEBASSEMBLY-NEXT:#define __INT_FAST64_TYPE__ long long int
// WEBASSEMBLY-NEXT:#define __INT_FAST8_FMTd__ "hhd"
// WEBASSEMBLY-NEXT:#define __INT_FAST8_FMTi__ "hhi"
// WEBASSEMBLY-NEXT:#define __INT_FAST8_MAX__ 127
// WEBASSEMBLY-NEXT:#define __INT_FAST8_TYPE__ signed char
// WEBASSEMBLY-NEXT:#define __INT_LEAST16_FMTd__ "hd"
// WEBASSEMBLY-NEXT:#define __INT_LEAST16_FMTi__ "hi"
// WEBASSEMBLY-NEXT:#define __INT_LEAST16_MAX__ 32767
// WEBASSEMBLY-NEXT:#define __INT_LEAST16_TYPE__ short
// WEBASSEMBLY-NEXT:#define __INT_LEAST32_FMTd__ "d"
// WEBASSEMBLY-NEXT:#define __INT_LEAST32_FMTi__ "i"
// WEBASSEMBLY-NEXT:#define __INT_LEAST32_MAX__ 2147483647
// WEBASSEMBLY-NEXT:#define __INT_LEAST32_TYPE__ int
// WEBASSEMBLY-NEXT:#define __INT_LEAST64_FMTd__ "lld"
// WEBASSEMBLY-NEXT:#define __INT_LEAST64_FMTi__ "lli"
// WEBASSEMBLY-NEXT:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// WEBASSEMBLY-NEXT:#define __INT_LEAST64_TYPE__ long long int
// WEBASSEMBLY-NEXT:#define __INT_LEAST8_FMTd__ "hhd"
// WEBASSEMBLY-NEXT:#define __INT_LEAST8_FMTi__ "hhi"
// WEBASSEMBLY-NEXT:#define __INT_LEAST8_MAX__ 127
// WEBASSEMBLY-NEXT:#define __INT_LEAST8_TYPE__ signed char
// WEBASSEMBLY-NEXT:#define __INT_MAX__ 2147483647
// WEBASSEMBLY-NEXT:#define __LDBL_DECIMAL_DIG__ 36
// WEBASSEMBLY-NEXT:#define __LDBL_DENORM_MIN__ 6.47517511943802511092443895822764655e-4966L
// WEBASSEMBLY-NEXT:#define __LDBL_DIG__ 33
// WEBASSEMBLY-NEXT:#define __LDBL_EPSILON__ 1.92592994438723585305597794258492732e-34L
// WEBASSEMBLY-NEXT:#define __LDBL_HAS_DENORM__ 1
// WEBASSEMBLY-NEXT:#define __LDBL_HAS_INFINITY__ 1
// WEBASSEMBLY-NEXT:#define __LDBL_HAS_QUIET_NAN__ 1
// WEBASSEMBLY-NEXT:#define __LDBL_MANT_DIG__ 113
// WEBASSEMBLY-NEXT:#define __LDBL_MAX_10_EXP__ 4932
// WEBASSEMBLY-NEXT:#define __LDBL_MAX_EXP__ 16384
// WEBASSEMBLY-NEXT:#define __LDBL_MAX__ 1.18973149535723176508575932662800702e+4932L
// WEBASSEMBLY-NEXT:#define __LDBL_MIN_10_EXP__ (-4931)
// WEBASSEMBLY-NEXT:#define __LDBL_MIN_EXP__ (-16381)
// WEBASSEMBLY-NEXT:#define __LDBL_MIN__ 3.36210314311209350626267781732175260e-4932L
// WEBASSEMBLY-NEXT:#define __LITTLE_ENDIAN__ 1
// WEBASSEMBLY-NEXT:#define __LONG_LONG_MAX__ 9223372036854775807LL
// WEBASSEMBLY32-NEXT:#define __LONG_MAX__ 2147483647L
// WEBASSEMBLY32-NOT:#define __LP64__
// WEBASSEMBLY64-NEXT:#define __LONG_MAX__ 9223372036854775807L
// WEBASSEMBLY64-NEXT:#define __LP64__ 1
// WEBASSEMBLY-NEXT:#define __NO_INLINE__ 1
// WEBASSEMBLY-NEXT:#define __OBJC_BOOL_IS_BOOL 0
// WEBASSEMBLY-NEXT:#define __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES 3
// WEBASSEMBLY-NEXT:#define __OPENCL_MEMORY_SCOPE_DEVICE 2
// WEBASSEMBLY-NEXT:#define __OPENCL_MEMORY_SCOPE_SUB_GROUP 4
// WEBASSEMBLY-NEXT:#define __OPENCL_MEMORY_SCOPE_WORK_GROUP 1
// WEBASSEMBLY-NEXT:#define __OPENCL_MEMORY_SCOPE_WORK_ITEM 0
// WEBASSEMBLY-NEXT:#define __ORDER_BIG_ENDIAN__ 4321
// WEBASSEMBLY-NEXT:#define __ORDER_LITTLE_ENDIAN__ 1234
// WEBASSEMBLY-NEXT:#define __ORDER_PDP_ENDIAN__ 3412
// WEBASSEMBLY32-NEXT:#define __POINTER_WIDTH__ 32
// WEBASSEMBLY64-NEXT:#define __POINTER_WIDTH__ 64
// WEBASSEMBLY-NEXT:#define __PRAGMA_REDEFINE_EXTNAME 1
// WEBASSEMBLY-NEXT:#define __PTRDIFF_FMTd__ "ld"
// WEBASSEMBLY-NEXT:#define __PTRDIFF_FMTi__ "li"
// WEBASSEMBLY32-NEXT:#define __PTRDIFF_MAX__ 2147483647L
// WEBASSEMBLY64-NEXT:#define __PTRDIFF_MAX__ 9223372036854775807L
// WEBASSEMBLY-NEXT:#define __PTRDIFF_TYPE__ long int
// WEBASSEMBLY32-NEXT:#define __PTRDIFF_WIDTH__ 32
// WEBASSEMBLY64-NEXT:#define __PTRDIFF_WIDTH__ 64
// WEBASSEMBLY-NOT:#define __REGISTER_PREFIX__
// WEBASSEMBLY-NEXT:#define __SCHAR_MAX__ 127
// WEBASSEMBLY-NEXT:#define __SHRT_MAX__ 32767
// WEBASSEMBLY32-NEXT:#define __SIG_ATOMIC_MAX__ 2147483647L
// WEBASSEMBLY32-NEXT:#define __SIG_ATOMIC_WIDTH__ 32
// WEBASSEMBLY64-NEXT:#define __SIG_ATOMIC_MAX__ 9223372036854775807L
// WEBASSEMBLY64-NEXT:#define __SIG_ATOMIC_WIDTH__ 64
// WEBASSEMBLY-NEXT:#define __SIZEOF_DOUBLE__ 8
// WEBASSEMBLY-NEXT:#define __SIZEOF_FLOAT__ 4
// WEBASSEMBLY-NEXT:#define __SIZEOF_INT128__ 16
// WEBASSEMBLY-NEXT:#define __SIZEOF_INT__ 4
// WEBASSEMBLY-NEXT:#define __SIZEOF_LONG_DOUBLE__ 16
// WEBASSEMBLY-NEXT:#define __SIZEOF_LONG_LONG__ 8
// WEBASSEMBLY32-NEXT:#define __SIZEOF_LONG__ 4
// WEBASSEMBLY32-NEXT:#define __SIZEOF_POINTER__ 4
// WEBASSEMBLY32-NEXT:#define __SIZEOF_PTRDIFF_T__ 4
// WEBASSEMBLY64-NEXT:#define __SIZEOF_LONG__ 8
// WEBASSEMBLY64-NEXT:#define __SIZEOF_POINTER__ 8
// WEBASSEMBLY64-NEXT:#define __SIZEOF_PTRDIFF_T__ 8
// WEBASSEMBLY-NEXT:#define __SIZEOF_SHORT__ 2
// WEBASSEMBLY32-NEXT:#define __SIZEOF_SIZE_T__ 4
// WEBASSEMBLY64-NEXT:#define __SIZEOF_SIZE_T__ 8
// WEBASSEMBLY-NEXT:#define __SIZEOF_WCHAR_T__ 4
// WEBASSEMBLY-NEXT:#define __SIZEOF_WINT_T__ 4
// WEBASSEMBLY-NEXT:#define __SIZE_FMTX__ "lX"
// WEBASSEMBLY-NEXT:#define __SIZE_FMTo__ "lo"
// WEBASSEMBLY-NEXT:#define __SIZE_FMTu__ "lu"
// WEBASSEMBLY-NEXT:#define __SIZE_FMTx__ "lx"
// WEBASSEMBLY32-NEXT:#define __SIZE_MAX__ 4294967295UL
// WEBASSEMBLY64-NEXT:#define __SIZE_MAX__ 18446744073709551615UL
// WEBASSEMBLY-NEXT:#define __SIZE_TYPE__ long unsigned int
// WEBASSEMBLY32-NEXT:#define __SIZE_WIDTH__ 32
// WEBASSEMBLY64-NEXT:#define __SIZE_WIDTH__ 64
// WEBASSEMBLY-NEXT:#define __STDC_HOSTED__ 0
// WEBASSEMBLY-NOT:#define __STDC_MB_MIGHT_NEQ_WC__
// WEBASSEMBLY-NOT:#define __STDC_NO_ATOMICS__
// WEBASSEMBLY-NOT:#define __STDC_NO_COMPLEX__
// WEBASSEMBLY-NOT:#define __STDC_NO_VLA__
// WEBASSEMBLY-NOT:#define __STDC_NO_THREADS__
// WEBASSEMBLY-NEXT:#define __STDC_UTF_16__ 1
// WEBASSEMBLY-NEXT:#define __STDC_UTF_32__ 1
// WEBASSEMBLY-NEXT:#define __STDC_VERSION__ 201710L
// WEBASSEMBLY-NEXT:#define __STDC__ 1
// WEBASSEMBLY-NEXT:#define __UINT16_C_SUFFIX__
// WEBASSEMBLY-NEXT:#define __UINT16_FMTX__ "hX"
// WEBASSEMBLY-NEXT:#define __UINT16_FMTo__ "ho"
// WEBASSEMBLY-NEXT:#define __UINT16_FMTu__ "hu"
// WEBASSEMBLY-NEXT:#define __UINT16_FMTx__ "hx"
// WEBASSEMBLY-NEXT:#define __UINT16_MAX__ 65535
// WEBASSEMBLY-NEXT:#define __UINT16_TYPE__ unsigned short
// WEBASSEMBLY-NEXT:#define __UINT32_C_SUFFIX__ U
// WEBASSEMBLY-NEXT:#define __UINT32_FMTX__ "X"
// WEBASSEMBLY-NEXT:#define __UINT32_FMTo__ "o"
// WEBASSEMBLY-NEXT:#define __UINT32_FMTu__ "u"
// WEBASSEMBLY-NEXT:#define __UINT32_FMTx__ "x"
// WEBASSEMBLY-NEXT:#define __UINT32_MAX__ 4294967295U
// WEBASSEMBLY-NEXT:#define __UINT32_TYPE__ unsigned int
// WEBASSEMBLY-NEXT:#define __UINT64_C_SUFFIX__ ULL
// WEBASSEMBLY-NEXT:#define __UINT64_FMTX__ "llX"
// WEBASSEMBLY-NEXT:#define __UINT64_FMTo__ "llo"
// WEBASSEMBLY-NEXT:#define __UINT64_FMTu__ "llu"
// WEBASSEMBLY-NEXT:#define __UINT64_FMTx__ "llx"
// WEBASSEMBLY-NEXT:#define __UINT64_MAX__ 18446744073709551615ULL
// WEBASSEMBLY-NEXT:#define __UINT64_TYPE__ long long unsigned int
// WEBASSEMBLY-NEXT:#define __UINT8_C_SUFFIX__
// WEBASSEMBLY-NEXT:#define __UINT8_FMTX__ "hhX"
// WEBASSEMBLY-NEXT:#define __UINT8_FMTo__ "hho"
// WEBASSEMBLY-NEXT:#define __UINT8_FMTu__ "hhu"
// WEBASSEMBLY-NEXT:#define __UINT8_FMTx__ "hhx"
// WEBASSEMBLY-NEXT:#define __UINT8_MAX__ 255
// WEBASSEMBLY-NEXT:#define __UINT8_TYPE__ unsigned char
// WEBASSEMBLY-NEXT:#define __UINTMAX_C_SUFFIX__ ULL
// WEBASSEMBLY-NEXT:#define __UINTMAX_FMTX__ "llX"
// WEBASSEMBLY-NEXT:#define __UINTMAX_FMTo__ "llo"
// WEBASSEMBLY-NEXT:#define __UINTMAX_FMTu__ "llu"
// WEBASSEMBLY-NEXT:#define __UINTMAX_FMTx__ "llx"
// WEBASSEMBLY-NEXT:#define __UINTMAX_MAX__ 18446744073709551615ULL
// WEBASSEMBLY-NEXT:#define __UINTMAX_TYPE__ long long unsigned int
// WEBASSEMBLY-NEXT:#define __UINTMAX_WIDTH__ 64
// WEBASSEMBLY-NEXT:#define __UINTPTR_FMTX__ "lX"
// WEBASSEMBLY-NEXT:#define __UINTPTR_FMTo__ "lo"
// WEBASSEMBLY-NEXT:#define __UINTPTR_FMTu__ "lu"
// WEBASSEMBLY-NEXT:#define __UINTPTR_FMTx__ "lx"
// WEBASSEMBLY32-NEXT:#define __UINTPTR_MAX__ 4294967295UL
// WEBASSEMBLY64-NEXT:#define __UINTPTR_MAX__ 18446744073709551615UL
// WEBASSEMBLY-NEXT:#define __UINTPTR_TYPE__ long unsigned int
// WEBASSEMBLY32-NEXT:#define __UINTPTR_WIDTH__ 32
// WEBASSEMBLY64-NEXT:#define __UINTPTR_WIDTH__ 64
// WEBASSEMBLY-NEXT:#define __UINT_FAST16_FMTX__ "hX"
// WEBASSEMBLY-NEXT:#define __UINT_FAST16_FMTo__ "ho"
// WEBASSEMBLY-NEXT:#define __UINT_FAST16_FMTu__ "hu"
// WEBASSEMBLY-NEXT:#define __UINT_FAST16_FMTx__ "hx"
// WEBASSEMBLY-NEXT:#define __UINT_FAST16_MAX__ 65535
// WEBASSEMBLY-NEXT:#define __UINT_FAST16_TYPE__ unsigned short
// WEBASSEMBLY-NEXT:#define __UINT_FAST32_FMTX__ "X"
// WEBASSEMBLY-NEXT:#define __UINT_FAST32_FMTo__ "o"
// WEBASSEMBLY-NEXT:#define __UINT_FAST32_FMTu__ "u"
// WEBASSEMBLY-NEXT:#define __UINT_FAST32_FMTx__ "x"
// WEBASSEMBLY-NEXT:#define __UINT_FAST32_MAX__ 4294967295U
// WEBASSEMBLY-NEXT:#define __UINT_FAST32_TYPE__ unsigned int
// WEBASSEMBLY-NEXT:#define __UINT_FAST64_FMTX__ "llX"
// WEBASSEMBLY-NEXT:#define __UINT_FAST64_FMTo__ "llo"
// WEBASSEMBLY-NEXT:#define __UINT_FAST64_FMTu__ "llu"
// WEBASSEMBLY-NEXT:#define __UINT_FAST64_FMTx__ "llx"
// WEBASSEMBLY-NEXT:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// WEBASSEMBLY-NEXT:#define __UINT_FAST64_TYPE__ long long unsigned int
// WEBASSEMBLY-NEXT:#define __UINT_FAST8_FMTX__ "hhX"
// WEBASSEMBLY-NEXT:#define __UINT_FAST8_FMTo__ "hho"
// WEBASSEMBLY-NEXT:#define __UINT_FAST8_FMTu__ "hhu"
// WEBASSEMBLY-NEXT:#define __UINT_FAST8_FMTx__ "hhx"
// WEBASSEMBLY-NEXT:#define __UINT_FAST8_MAX__ 255
// WEBASSEMBLY-NEXT:#define __UINT_FAST8_TYPE__ unsigned char
// WEBASSEMBLY-NEXT:#define __UINT_LEAST16_FMTX__ "hX"
// WEBASSEMBLY-NEXT:#define __UINT_LEAST16_FMTo__ "ho"
// WEBASSEMBLY-NEXT:#define __UINT_LEAST16_FMTu__ "hu"
// WEBASSEMBLY-NEXT:#define __UINT_LEAST16_FMTx__ "hx"
// WEBASSEMBLY-NEXT:#define __UINT_LEAST16_MAX__ 65535
// WEBASSEMBLY-NEXT:#define __UINT_LEAST16_TYPE__ unsigned short
// WEBASSEMBLY-NEXT:#define __UINT_LEAST32_FMTX__ "X"
// WEBASSEMBLY-NEXT:#define __UINT_LEAST32_FMTo__ "o"
// WEBASSEMBLY-NEXT:#define __UINT_LEAST32_FMTu__ "u"
// WEBASSEMBLY-NEXT:#define __UINT_LEAST32_FMTx__ "x"
// WEBASSEMBLY-NEXT:#define __UINT_LEAST32_MAX__ 4294967295U
// WEBASSEMBLY-NEXT:#define __UINT_LEAST32_TYPE__ unsigned int
// WEBASSEMBLY-NEXT:#define __UINT_LEAST64_FMTX__ "llX"
// WEBASSEMBLY-NEXT:#define __UINT_LEAST64_FMTo__ "llo"
// WEBASSEMBLY-NEXT:#define __UINT_LEAST64_FMTu__ "llu"
// WEBASSEMBLY-NEXT:#define __UINT_LEAST64_FMTx__ "llx"
// WEBASSEMBLY-NEXT:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// WEBASSEMBLY-NEXT:#define __UINT_LEAST64_TYPE__ long long unsigned int
// WEBASSEMBLY-NEXT:#define __UINT_LEAST8_FMTX__ "hhX"
// WEBASSEMBLY-NEXT:#define __UINT_LEAST8_FMTo__ "hho"
// WEBASSEMBLY-NEXT:#define __UINT_LEAST8_FMTu__ "hhu"
// WEBASSEMBLY-NEXT:#define __UINT_LEAST8_FMTx__ "hhx"
// WEBASSEMBLY-NEXT:#define __UINT_LEAST8_MAX__ 255
// WEBASSEMBLY-NEXT:#define __UINT_LEAST8_TYPE__ unsigned char
// WEBASSEMBLY-NEXT:#define __USER_LABEL_PREFIX__
// WEBASSEMBLY-NEXT:#define __VERSION__ "{{.*}}"
// WEBASSEMBLY-NEXT:#define __WCHAR_MAX__ 2147483647
// WEBASSEMBLY-NEXT:#define __WCHAR_TYPE__ int
// WEBASSEMBLY-NOT:#define __WCHAR_UNSIGNED__
// WEBASSEMBLY-NEXT:#define __WCHAR_WIDTH__ 32
// WEBASSEMBLY-NEXT:#define __WINT_MAX__ 2147483647
// WEBASSEMBLY-NEXT:#define __WINT_TYPE__ int
// WEBASSEMBLY-NOT:#define __WINT_UNSIGNED__
// WEBASSEMBLY-NEXT:#define __WINT_WIDTH__ 32
// WEBASSEMBLY-NEXT:#define __clang__ 1
// WEBASSEMBLY-NEXT:#define __clang_major__ {{.*}}
// WEBASSEMBLY-NEXT:#define __clang_minor__ {{.*}}
// WEBASSEMBLY-NEXT:#define __clang_patchlevel__ {{.*}}
// WEBASSEMBLY-NEXT:#define __clang_version__ "{{.*}}"
// WEBASSEMBLY-NEXT:#define __llvm__ 1
// WEBASSEMBLY-NOT:#define __unix
// WEBASSEMBLY-NOT:#define __unix__
// WEBASSEMBLY-WASI-NEXT:#define __wasi__ 1
// WEBASSEMBLY-NOT:#define __wasm_simd128__
// WEBASSEMBLY-NOT:#define __wasm_simd256__
// WEBASSEMBLY-NOT:#define __wasm_simd512__
// WEBASSEMBLY-NEXT:#define __wasm 1
// WEBASSEMBLY32-NEXT:#define __wasm32 1
// WEBASSEMBLY64-NOT:#define __wasm32
// WEBASSEMBLY32-NEXT:#define __wasm32__ 1
// WEBASSEMBLY64-NOT:#define __wasm32__
// WEBASSEMBLY32-NOT:#define __wasm64__
// WEBASSEMBLY32-NOT:#define __wasm64
// WEBASSEMBLY64-NEXT:#define __wasm64 1
// WEBASSEMBLY64-NEXT:#define __wasm64__ 1
// WEBASSEMBLY-NEXT:#define __wasm__ 1
// WEBASSEMBLY-CXX-NOT:_REENTRANT
// WEBASSEMBLY-CXX-NOT:__STDCPP_THREADS__
// WEBASSEMBLY-CXX-ATOMICS:#define _REENTRANT 1
// WEBASSEMBLY-CXX-ATOMICS:#define __STDCPP_THREADS__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple i686-windows-cygnus < /dev/null | FileCheck -match-full-lines -check-prefix CYGWIN-X32 %s
// CYGWIN-X32: #define __USER_LABEL_PREFIX__ _

// RUN: %clang_cc1 -E -dM -ffreestanding -triple x86_64-windows-cygnus < /dev/null | FileCheck -match-full-lines -check-prefix CYGWIN-X64 %s
// CYGWIN-X64: #define __USER_LABEL_PREFIX__

// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=avr \
// RUN:   < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=AVR %s
//
// AVR:#define __ATOMIC_ACQUIRE 2
// AVR:#define __ATOMIC_ACQ_REL 4
// AVR:#define __ATOMIC_CONSUME 1
// AVR:#define __ATOMIC_RELAXED 0
// AVR:#define __ATOMIC_RELEASE 3
// AVR:#define __ATOMIC_SEQ_CST 5
// AVR:#define __AVR__ 1
// AVR:#define __BIGGEST_ALIGNMENT__ 1
// AVR:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// AVR:#define __CHAR16_TYPE__ unsigned int
// AVR:#define __CHAR32_TYPE__ long unsigned int
// AVR:#define __CHAR_BIT__ 8
// AVR:#define __DBL_DECIMAL_DIG__ 9
// AVR:#define __DBL_DENORM_MIN__ 1.40129846e-45
// AVR:#define __DBL_DIG__ 6
// AVR:#define __DBL_EPSILON__ 1.19209290e-7
// AVR:#define __DBL_HAS_DENORM__ 1
// AVR:#define __DBL_HAS_INFINITY__ 1
// AVR:#define __DBL_HAS_QUIET_NAN__ 1
// AVR:#define __DBL_MANT_DIG__ 24
// AVR:#define __DBL_MAX_10_EXP__ 38
// AVR:#define __DBL_MAX_EXP__ 128
// AVR:#define __DBL_MAX__ 3.40282347e+38
// AVR:#define __DBL_MIN_10_EXP__ (-37)
// AVR:#define __DBL_MIN_EXP__ (-125)
// AVR:#define __DBL_MIN__ 1.17549435e-38
// AVR:#define __FINITE_MATH_ONLY__ 0
// AVR:#define __FLT_DECIMAL_DIG__ 9
// AVR:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// AVR:#define __FLT_DIG__ 6
// AVR:#define __FLT_EPSILON__ 1.19209290e-7F
// AVR:#define __FLT_EVAL_METHOD__ 0
// AVR:#define __FLT_HAS_DENORM__ 1
// AVR:#define __FLT_HAS_INFINITY__ 1
// AVR:#define __FLT_HAS_QUIET_NAN__ 1
// AVR:#define __FLT_MANT_DIG__ 24
// AVR:#define __FLT_MAX_10_EXP__ 38
// AVR:#define __FLT_MAX_EXP__ 128
// AVR:#define __FLT_MAX__ 3.40282347e+38F
// AVR:#define __FLT_MIN_10_EXP__ (-37)
// AVR:#define __FLT_MIN_EXP__ (-125)
// AVR:#define __FLT_MIN__ 1.17549435e-38F
// AVR:#define __FLT_RADIX__ 2
// AVR:#define __GCC_ATOMIC_BOOL_LOCK_FREE 1
// AVR:#define __GCC_ATOMIC_CHAR16_T_LOCK_FREE 1
// AVR:#define __GCC_ATOMIC_CHAR32_T_LOCK_FREE 1
// AVR:#define __GCC_ATOMIC_CHAR_LOCK_FREE 1
// AVR:#define __GCC_ATOMIC_INT_LOCK_FREE 1
// AVR:#define __GCC_ATOMIC_LLONG_LOCK_FREE 1
// AVR:#define __GCC_ATOMIC_LONG_LOCK_FREE 1
// AVR:#define __GCC_ATOMIC_POINTER_LOCK_FREE 1
// AVR:#define __GCC_ATOMIC_SHORT_LOCK_FREE 1
// AVR:#define __GCC_ATOMIC_TEST_AND_SET_TRUEVAL 1
// AVR:#define __GCC_ATOMIC_WCHAR_T_LOCK_FREE 1
// AVR:#define __GXX_ABI_VERSION 1002
// AVR:#define __INT16_C_SUFFIX__
// AVR:#define __INT16_MAX__ 32767
// AVR:#define __INT16_TYPE__ short
// AVR:#define __INT32_C_SUFFIX__ L
// AVR:#define __INT32_MAX__ 2147483647L
// AVR:#define __INT32_TYPE__ long int
// AVR:#define __INT64_C_SUFFIX__ LL
// AVR:#define __INT64_MAX__ 9223372036854775807LL
// AVR:#define __INT64_TYPE__ long long int
// AVR:#define __INT8_C_SUFFIX__
// AVR:#define __INT8_MAX__ 127
// AVR:#define __INT8_TYPE__ signed char
// AVR:#define __INTMAX_C_SUFFIX__ LL
// AVR:#define __INTMAX_MAX__ 9223372036854775807LL
// AVR:#define __INTMAX_TYPE__ long long int
// AVR:#define __INTPTR_MAX__ 32767
// AVR:#define __INTPTR_TYPE__ int
// AVR:#define __INT_FAST16_MAX__ 32767
// AVR:#define __INT_FAST16_TYPE__ int
// AVR:#define __INT_FAST32_MAX__ 2147483647L
// AVR:#define __INT_FAST32_TYPE__ long int
// AVR:#define __INT_FAST64_MAX__ 9223372036854775807LL
// AVR:#define __INT_FAST64_TYPE__ long long int
// AVR:#define __INT_FAST8_MAX__ 127
// AVR:#define __INT_FAST8_TYPE__ signed char
// AVR:#define __INT_LEAST16_MAX__ 32767
// AVR:#define __INT_LEAST16_TYPE__ int
// AVR:#define __INT_LEAST32_MAX__ 2147483647L
// AVR:#define __INT_LEAST32_TYPE__ long int
// AVR:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// AVR:#define __INT_LEAST64_TYPE__ long long int
// AVR:#define __INT_LEAST8_MAX__ 127
// AVR:#define __INT_LEAST8_TYPE__ signed char
// AVR:#define __INT_MAX__ 32767
// AVR:#define __LDBL_DECIMAL_DIG__ 9
// AVR:#define __LDBL_DENORM_MIN__ 1.40129846e-45L
// AVR:#define __LDBL_DIG__ 6
// AVR:#define __LDBL_EPSILON__ 1.19209290e-7L
// AVR:#define __LDBL_HAS_DENORM__ 1
// AVR:#define __LDBL_HAS_INFINITY__ 1
// AVR:#define __LDBL_HAS_QUIET_NAN__ 1
// AVR:#define __LDBL_MANT_DIG__ 24
// AVR:#define __LDBL_MAX_10_EXP__ 38
// AVR:#define __LDBL_MAX_EXP__ 128
// AVR:#define __LDBL_MAX__ 3.40282347e+38L
// AVR:#define __LDBL_MIN_10_EXP__ (-37)
// AVR:#define __LDBL_MIN_EXP__ (-125)
// AVR:#define __LDBL_MIN__ 1.17549435e-38L
// AVR:#define __LONG_LONG_MAX__ 9223372036854775807LL
// AVR:#define __LONG_MAX__ 2147483647L
// AVR:#define __NO_INLINE__ 1
// AVR:#define __ORDER_BIG_ENDIAN__ 4321
// AVR:#define __ORDER_LITTLE_ENDIAN__ 1234
// AVR:#define __ORDER_PDP_ENDIAN__ 3412
// AVR:#define __PRAGMA_REDEFINE_EXTNAME 1
// AVR:#define __PTRDIFF_MAX__ 32767
// AVR:#define __PTRDIFF_TYPE__ int
// AVR:#define __SCHAR_MAX__ 127
// AVR:#define __SHRT_MAX__ 32767
// AVR:#define __SIG_ATOMIC_MAX__ 127
// AVR:#define __SIG_ATOMIC_WIDTH__ 8
// AVR:#define __SIZEOF_DOUBLE__ 4
// AVR:#define __SIZEOF_FLOAT__ 4
// AVR:#define __SIZEOF_INT__ 2
// AVR:#define __SIZEOF_LONG_DOUBLE__ 4
// AVR:#define __SIZEOF_LONG_LONG__ 8
// AVR:#define __SIZEOF_LONG__ 4
// AVR:#define __SIZEOF_POINTER__ 2
// AVR:#define __SIZEOF_PTRDIFF_T__ 2
// AVR:#define __SIZEOF_SHORT__ 2
// AVR:#define __SIZEOF_SIZE_T__ 2
// AVR:#define __SIZEOF_WCHAR_T__ 2
// AVR:#define __SIZEOF_WINT_T__ 2
// AVR:#define __SIZE_MAX__ 65535U
// AVR:#define __SIZE_TYPE__ unsigned int
// AVR:#define __STDC__ 1
// AVR:#define __UINT16_MAX__ 65535U
// AVR:#define __UINT16_TYPE__ unsigned short
// AVR:#define __UINT32_C_SUFFIX__ UL
// AVR:#define __UINT32_MAX__ 4294967295UL
// AVR:#define __UINT32_TYPE__ long unsigned int
// AVR:#define __UINT64_C_SUFFIX__ ULL
// AVR:#define __UINT64_MAX__ 18446744073709551615ULL
// AVR:#define __UINT64_TYPE__ long long unsigned int
// AVR:#define __UINT8_C_SUFFIX__
// AVR:#define __UINT8_MAX__ 255
// AVR:#define __UINT8_TYPE__ unsigned char
// AVR:#define __UINTMAX_C_SUFFIX__ ULL
// AVR:#define __UINTMAX_MAX__ 18446744073709551615ULL
// AVR:#define __UINTMAX_TYPE__ long long unsigned int
// AVR:#define __UINTPTR_MAX__ 65535U
// AVR:#define __UINTPTR_TYPE__ unsigned int
// AVR:#define __UINT_FAST16_MAX__ 65535U
// AVR:#define __UINT_FAST16_TYPE__ unsigned int
// AVR:#define __UINT_FAST32_MAX__ 4294967295UL
// AVR:#define __UINT_FAST32_TYPE__ long unsigned int
// AVR:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// AVR:#define __UINT_FAST64_TYPE__ long long unsigned int
// AVR:#define __UINT_FAST8_MAX__ 255
// AVR:#define __UINT_FAST8_TYPE__ unsigned char
// AVR:#define __UINT_LEAST16_MAX__ 65535U
// AVR:#define __UINT_LEAST16_TYPE__ unsigned int
// AVR:#define __UINT_LEAST32_MAX__ 4294967295UL
// AVR:#define __UINT_LEAST32_TYPE__ long unsigned int
// AVR:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// AVR:#define __UINT_LEAST64_TYPE__ long long unsigned int
// AVR:#define __UINT_LEAST8_MAX__ 255
// AVR:#define __UINT_LEAST8_TYPE__ unsigned char
// AVR:#define __USER_LABEL_PREFIX__
// AVR:#define __WCHAR_MAX__ 32767
// AVR:#define __WCHAR_TYPE__ int
// AVR:#define __WINT_TYPE__ int


// RUN: %clang_cc1 -E -dM -ffreestanding \
// RUN:    -triple i686-windows-msvc -fms-compatibility -x c++ < /dev/null \
// RUN:  | FileCheck -match-full-lines -check-prefix MSVC-X32 %s

// RUN: %clang_cc1 -E -dM -ffreestanding \
// RUN:    -triple x86_64-windows-msvc -fms-compatibility -x c++ < /dev/null \
// RUN:  | FileCheck -match-full-lines -check-prefix MSVC-X64 %s

// MSVC-X32:#define __CLANG_ATOMIC_BOOL_LOCK_FREE 2
// MSVC-X32-NEXT:#define __CLANG_ATOMIC_CHAR16_T_LOCK_FREE 2
// MSVC-X32-NEXT:#define __CLANG_ATOMIC_CHAR32_T_LOCK_FREE 2
// MSVC-X32-NEXT:#define __CLANG_ATOMIC_CHAR_LOCK_FREE 2
// MSVC-X32-NEXT:#define __CLANG_ATOMIC_INT_LOCK_FREE 2
// MSVC-X32-NEXT:#define __CLANG_ATOMIC_LLONG_LOCK_FREE 2
// MSVC-X32-NEXT:#define __CLANG_ATOMIC_LONG_LOCK_FREE 2
// MSVC-X32-NEXT:#define __CLANG_ATOMIC_POINTER_LOCK_FREE 2
// MSVC-X32-NEXT:#define __CLANG_ATOMIC_SHORT_LOCK_FREE 2
// MSVC-X32-NEXT:#define __CLANG_ATOMIC_WCHAR_T_LOCK_FREE 2
// MSVC-X32-NOT:#define __GCC_ATOMIC{{.*}}
// MSVC-X32:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 8U

// MSVC-X64:#define __CLANG_ATOMIC_BOOL_LOCK_FREE 2
// MSVC-X64-NEXT:#define __CLANG_ATOMIC_CHAR16_T_LOCK_FREE 2
// MSVC-X64-NEXT:#define __CLANG_ATOMIC_CHAR32_T_LOCK_FREE 2
// MSVC-X64-NEXT:#define __CLANG_ATOMIC_CHAR_LOCK_FREE 2
// MSVC-X64-NEXT:#define __CLANG_ATOMIC_INT_LOCK_FREE 2
// MSVC-X64-NEXT:#define __CLANG_ATOMIC_LLONG_LOCK_FREE 2
// MSVC-X64-NEXT:#define __CLANG_ATOMIC_LONG_LOCK_FREE 2
// MSVC-X64-NEXT:#define __CLANG_ATOMIC_POINTER_LOCK_FREE 2
// MSVC-X64-NEXT:#define __CLANG_ATOMIC_SHORT_LOCK_FREE 2
// MSVC-X64-NEXT:#define __CLANG_ATOMIC_WCHAR_T_LOCK_FREE 2
// MSVC-X64-NOT:#define __GCC_ATOMIC{{.*}}
// MSVC-X64:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 16ULL

// RUN: %clang_cc1 -E -dM -ffreestanding                \
// RUN:  -fgnuc-version=4.2.1  -triple=aarch64-apple-ios9 < /dev/null        \
// RUN: | FileCheck -check-prefix=DARWIN %s
// RUN: %clang_cc1 -E -dM -ffreestanding                \
// RUN:   -fgnuc-version=4.2.1 -triple=aarch64-apple-macosx10.12 < /dev/null \
// RUN: | FileCheck -check-prefix=DARWIN %s

// DARWIN-NOT: OBJC_NEW_PROPERTIES
// DARWIN:#define __STDC_NO_THREADS__ 1

// RUN: %clang_cc1 -triple i386-apple-macosx -ffreestanding -dM -E /dev/null -o - | FileCheck -match-full-lines -check-prefix MACOS-32 %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx -ffreestanding -dM -E /dev/null -o - | FileCheck -match-full-lines -check-prefix MACOS-64 %s

// MACOS-32: #define __INTPTR_TYPE__ long int
// MACOS-32: #define __PTRDIFF_TYPE__ int
// MACOS-32: #define __SIZE_TYPE__ long unsigned int

// MACOS-64: #define __INTPTR_TYPE__ long int
// MACOS-64: #define __PTRDIFF_TYPE__ long int
// MACOS-64: #define __SIZE_TYPE__ long unsigned int

// RUN: %clang_cc1 -triple i386-apple-ios-simulator -ffreestanding -dM -E /dev/null -o - | FileCheck -match-full-lines -check-prefix IOS-32 %s
// RUN: %clang_cc1 -triple armv7-apple-ios -ffreestanding -dM -E /dev/null -o - | FileCheck -match-full-lines -check-prefix IOS-32 %s
// RUN: %clang_cc1 -triple x86_64-apple-ios-simulator -ffreestanding -dM -E /dev/null -o - | FileCheck -match-full-lines -check-prefix IOS-64 %s
// RUN: %clang_cc1 -triple arm64-apple-ios -ffreestanding -dM -E /dev/null -o - | FileCheck -match-full-lines -check-prefix IOS-64 %s

// IOS-32: #define __INTPTR_TYPE__ long int
// IOS-32: #define __PTRDIFF_TYPE__ int
// IOS-32: #define __SIZE_TYPE__ long unsigned int

// IOS-64: #define __INTPTR_TYPE__ long int
// IOS-64: #define __PTRDIFF_TYPE__ long int
// IOS-64: #define __SIZE_TYPE__ long unsigned int

// RUN: %clang_cc1 -triple i386-apple-tvos-simulator -ffreestanding -dM -E /dev/null -o - | FileCheck -match-full-lines -check-prefix TVOS-32 %s
// RUN: %clang_cc1 -triple armv7-apple-tvos -ffreestanding -dM -E /dev/null -o - | FileCheck -match-full-lines -check-prefix TVOS-32 %s
// RUN: %clang_cc1 -triple x86_64-apple-tvos-simulator -ffreestanding -dM -E /dev/null -o - | FileCheck -match-full-lines -check-prefix TVOS-64 %s
// RUN: %clang_cc1 -triple arm64-apple-tvos -ffreestanding -dM -E /dev/null -o - | FileCheck -match-full-lines -check-prefix TVOS-64 %s

// TVOS-32: #define __INTPTR_TYPE__ long int
// TVOS-32: #define __PTRDIFF_TYPE__ int
// TVOS-32: #define __SIZE_TYPE__ long unsigned int

// TVOS-64: #define __INTPTR_TYPE__ long int
// TVOS-64: #define __PTRDIFF_TYPE__ long int
// TVOS-64: #define __SIZE_TYPE__ long unsigned int

// RUN: %clang_cc1 -triple i386-apple-watchos-simulator -ffreestanding -dM -E /dev/null -o - | FileCheck -match-full-lines -check-prefix WATCHOS-32 %s
// RUN: %clang_cc1 -triple armv7k-apple-watchos -ffreestanding -dM -E /dev/null -o - | FileCheck -match-full-lines -check-prefix WATCHOS-64 %s
// RUN: %clang_cc1 -triple x86_64-apple-watchos-simulator -ffreestanding -dM -E /dev/null -o - | FileCheck -match-full-lines -check-prefix WATCHOS-64 %s
// RUN: %clang_cc1 -triple arm64-apple-watchos -ffreestanding -dM -E /dev/null -o - | FileCheck -match-full-lines -check-prefix WATCHOS-64 %s

// WATCHOS-32: #define __INTPTR_TYPE__ long int
// WATCHOS-32: #define __PTRDIFF_TYPE__ int
// WATCHOS-32: #define __SIZE_TYPE__ long unsigned int

// WATCHOS-64: #define __INTPTR_TYPE__ long int
// WATCHOS-64: #define __PTRDIFF_TYPE__ long int
// WATCHOS-64: #define __SIZE_TYPE__ long unsigned int

// RUN: %clang_cc1 -triple armv7-apple-none-macho -ffreestanding -dM -E /dev/null -o - | FileCheck -match-full-lines -check-prefix ARM-DARWIN-BAREMETAL-32 %s
// RUN: %clang_cc1 -triple arm64-apple-none-macho -ffreestanding -dM -E /dev/null -o - | FileCheck -match-full-lines -check-prefix ARM-DARWIN-BAREMETAL-64 %s

// ARM-DARWIN-BAREMETAL-32: #define __INTPTR_TYPE__ long int
// ARM-DARWIN-BAREMETAL-32: #define __PTRDIFF_TYPE__ int
// ARM-DARWIN-BAREMETAL-32: #define __SIZE_TYPE__ long unsigned int

// ARM-DARWIN-BAREMETAL-64: #define __INTPTR_TYPE__ long int
// ARM-DARWIN-BAREMETAL-64: #define __PTRDIFF_TYPE__ long int
// ARM-DARWIN-BAREMETAL-64: #define __SIZE_TYPE__ long unsigned int

// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=riscv32 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=RISCV32 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=riscv32-unknown-linux < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefixes=RISCV32,RISCV32-LINUX %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=riscv32 \
// RUN: -fforce-enable-int128 < /dev/null | FileCheck -match-full-lines \
// RUN: -check-prefixes=RISCV32,RISCV32-INT128 %s
// RISCV32: #define _ILP32 1
// RISCV32: #define __ATOMIC_ACQUIRE 2
// RISCV32: #define __ATOMIC_ACQ_REL 4
// RISCV32: #define __ATOMIC_CONSUME 1
// RISCV32: #define __ATOMIC_RELAXED 0
// RISCV32: #define __ATOMIC_RELEASE 3
// RISCV32: #define __ATOMIC_SEQ_CST 5
// RISCV32: #define __BIGGEST_ALIGNMENT__ 16
// RISCV32: #define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// RISCV32: #define __CHAR16_TYPE__ unsigned short
// RISCV32: #define __CHAR32_TYPE__ unsigned int
// RISCV32: #define __CHAR_BIT__ 8
// RISCV32: #define __DBL_DECIMAL_DIG__ 17
// RISCV32: #define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// RISCV32: #define __DBL_DIG__ 15
// RISCV32: #define __DBL_EPSILON__ 2.2204460492503131e-16
// RISCV32: #define __DBL_HAS_DENORM__ 1
// RISCV32: #define __DBL_HAS_INFINITY__ 1
// RISCV32: #define __DBL_HAS_QUIET_NAN__ 1
// RISCV32: #define __DBL_MANT_DIG__ 53
// RISCV32: #define __DBL_MAX_10_EXP__ 308
// RISCV32: #define __DBL_MAX_EXP__ 1024
// RISCV32: #define __DBL_MAX__ 1.7976931348623157e+308
// RISCV32: #define __DBL_MIN_10_EXP__ (-307)
// RISCV32: #define __DBL_MIN_EXP__ (-1021)
// RISCV32: #define __DBL_MIN__ 2.2250738585072014e-308
// RISCV32: #define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// RISCV32: #define __ELF__ 1
// RISCV32: #define __FINITE_MATH_ONLY__ 0
// RISCV32: #define __FLT_DECIMAL_DIG__ 9
// RISCV32: #define __FLT_DENORM_MIN__ 1.40129846e-45F
// RISCV32: #define __FLT_DIG__ 6
// RISCV32: #define __FLT_EPSILON__ 1.19209290e-7F
// RISCV32: #define __FLT_EVAL_METHOD__ 0
// RISCV32: #define __FLT_HAS_DENORM__ 1
// RISCV32: #define __FLT_HAS_INFINITY__ 1
// RISCV32: #define __FLT_HAS_QUIET_NAN__ 1
// RISCV32: #define __FLT_MANT_DIG__ 24
// RISCV32: #define __FLT_MAX_10_EXP__ 38
// RISCV32: #define __FLT_MAX_EXP__ 128
// RISCV32: #define __FLT_MAX__ 3.40282347e+38F
// RISCV32: #define __FLT_MIN_10_EXP__ (-37)
// RISCV32: #define __FLT_MIN_EXP__ (-125)
// RISCV32: #define __FLT_MIN__ 1.17549435e-38F
// RISCV32: #define __FLT_RADIX__ 2
// RISCV32: #define __GCC_ATOMIC_BOOL_LOCK_FREE 1
// RISCV32: #define __GCC_ATOMIC_CHAR16_T_LOCK_FREE 1
// RISCV32: #define __GCC_ATOMIC_CHAR32_T_LOCK_FREE 1
// RISCV32: #define __GCC_ATOMIC_CHAR_LOCK_FREE 1
// RISCV32: #define __GCC_ATOMIC_INT_LOCK_FREE 1
// RISCV32: #define __GCC_ATOMIC_LLONG_LOCK_FREE 1
// RISCV32: #define __GCC_ATOMIC_LONG_LOCK_FREE 1
// RISCV32: #define __GCC_ATOMIC_POINTER_LOCK_FREE 1
// RISCV32: #define __GCC_ATOMIC_SHORT_LOCK_FREE 1
// RISCV32: #define __GCC_ATOMIC_TEST_AND_SET_TRUEVAL 1
// RISCV32: #define __GCC_ATOMIC_WCHAR_T_LOCK_FREE 1
// RISCV32: #define __GNUC_MINOR__ {{.*}}
// RISCV32: #define __GNUC_PATCHLEVEL__ {{.*}}
// RISCV32: #define __GNUC_STDC_INLINE__ 1
// RISCV32: #define __GNUC__ {{.*}}
// RISCV32: #define __GXX_ABI_VERSION {{.*}}
// RISCV32: #define __ILP32__ 1
// RISCV32: #define __INT16_C_SUFFIX__
// RISCV32: #define __INT16_MAX__ 32767
// RISCV32: #define __INT16_TYPE__ short
// RISCV32: #define __INT32_C_SUFFIX__
// RISCV32: #define __INT32_MAX__ 2147483647
// RISCV32: #define __INT32_TYPE__ int
// RISCV32: #define __INT64_C_SUFFIX__ LL
// RISCV32: #define __INT64_MAX__ 9223372036854775807LL
// RISCV32: #define __INT64_TYPE__ long long int
// RISCV32: #define __INT8_C_SUFFIX__
// RISCV32: #define __INT8_MAX__ 127
// RISCV32: #define __INT8_TYPE__ signed char
// RISCV32: #define __INTMAX_C_SUFFIX__ LL
// RISCV32: #define __INTMAX_MAX__ 9223372036854775807LL
// RISCV32: #define __INTMAX_TYPE__ long long int
// RISCV32: #define __INTMAX_WIDTH__ 64
// RISCV32: #define __INTPTR_MAX__ 2147483647
// RISCV32: #define __INTPTR_TYPE__ int
// RISCV32: #define __INTPTR_WIDTH__ 32
// TODO: RISC-V GCC defines INT_FAST16 as int
// RISCV32: #define __INT_FAST16_MAX__ 32767
// RISCV32: #define __INT_FAST16_TYPE__ short
// RISCV32: #define __INT_FAST32_MAX__ 2147483647
// RISCV32: #define __INT_FAST32_TYPE__ int
// RISCV32: #define __INT_FAST64_MAX__ 9223372036854775807LL
// RISCV32: #define __INT_FAST64_TYPE__ long long int
// TODO: RISC-V GCC defines INT_FAST8 as int
// RISCV32: #define __INT_FAST8_MAX__ 127
// RISCV32: #define __INT_FAST8_TYPE__ signed char
// RISCV32: #define __INT_LEAST16_MAX__ 32767
// RISCV32: #define __INT_LEAST16_TYPE__ short
// RISCV32: #define __INT_LEAST32_MAX__ 2147483647
// RISCV32: #define __INT_LEAST32_TYPE__ int
// RISCV32: #define __INT_LEAST64_MAX__ 9223372036854775807LL
// RISCV32: #define __INT_LEAST64_TYPE__ long long int
// RISCV32: #define __INT_LEAST8_MAX__ 127
// RISCV32: #define __INT_LEAST8_TYPE__ signed char
// RISCV32: #define __INT_MAX__ 2147483647
// RISCV32: #define __LDBL_DECIMAL_DIG__ 36
// RISCV32: #define __LDBL_DENORM_MIN__ 6.47517511943802511092443895822764655e-4966L
// RISCV32: #define __LDBL_DIG__ 33
// RISCV32: #define __LDBL_EPSILON__ 1.92592994438723585305597794258492732e-34L
// RISCV32: #define __LDBL_HAS_DENORM__ 1
// RISCV32: #define __LDBL_HAS_INFINITY__ 1
// RISCV32: #define __LDBL_HAS_QUIET_NAN__ 1
// RISCV32: #define __LDBL_MANT_DIG__ 113
// RISCV32: #define __LDBL_MAX_10_EXP__ 4932
// RISCV32: #define __LDBL_MAX_EXP__ 16384
// RISCV32: #define __LDBL_MAX__ 1.18973149535723176508575932662800702e+4932L
// RISCV32: #define __LDBL_MIN_10_EXP__ (-4931)
// RISCV32: #define __LDBL_MIN_EXP__ (-16381)
// RISCV32: #define __LDBL_MIN__ 3.36210314311209350626267781732175260e-4932L
// RISCV32: #define __LITTLE_ENDIAN__ 1
// RISCV32: #define __LONG_LONG_MAX__ 9223372036854775807LL
// RISCV32: #define __LONG_MAX__ 2147483647L
// RISCV32: #define __NO_INLINE__ 1
// RISCV32: #define __POINTER_WIDTH__ 32
// RISCV32: #define __PRAGMA_REDEFINE_EXTNAME 1
// RISCV32: #define __PTRDIFF_MAX__ 2147483647
// RISCV32: #define __PTRDIFF_TYPE__ int
// RISCV32: #define __PTRDIFF_WIDTH__ 32
// RISCV32: #define __SCHAR_MAX__ 127
// RISCV32: #define __SHRT_MAX__ 32767
// RISCV32: #define __SIG_ATOMIC_MAX__ 2147483647
// RISCV32: #define __SIG_ATOMIC_WIDTH__ 32
// RISCV32: #define __SIZEOF_DOUBLE__ 8
// RISCV32: #define __SIZEOF_FLOAT__ 4
// RISCV32-INT128: #define __SIZEOF_INT128__ 16
// RISCV32: #define __SIZEOF_INT__ 4
// RISCV32: #define __SIZEOF_LONG_DOUBLE__ 16
// RISCV32: #define __SIZEOF_LONG_LONG__ 8
// RISCV32: #define __SIZEOF_LONG__ 4
// RISCV32: #define __SIZEOF_POINTER__ 4
// RISCV32: #define __SIZEOF_PTRDIFF_T__ 4
// RISCV32: #define __SIZEOF_SHORT__ 2
// RISCV32: #define __SIZEOF_SIZE_T__ 4
// RISCV32: #define __SIZEOF_WCHAR_T__ 4
// RISCV32: #define __SIZEOF_WINT_T__ 4
// RISCV32: #define __SIZE_MAX__ 4294967295U
// RISCV32: #define __SIZE_TYPE__ unsigned int
// RISCV32: #define __SIZE_WIDTH__ 32
// RISCV32: #define __STDC_HOSTED__ 0
// RISCV32: #define __STDC_UTF_16__ 1
// RISCV32: #define __STDC_UTF_32__ 1
// RISCV32: #define __STDC_VERSION__ 201710L
// RISCV32: #define __STDC__ 1
// RISCV32: #define __UINT16_C_SUFFIX__
// RISCV32: #define __UINT16_MAX__ 65535
// RISCV32: #define __UINT16_TYPE__ unsigned short
// RISCV32: #define __UINT32_C_SUFFIX__ U
// RISCV32: #define __UINT32_MAX__ 4294967295U
// RISCV32: #define __UINT32_TYPE__ unsigned int
// RISCV32: #define __UINT64_C_SUFFIX__ ULL
// RISCV32: #define __UINT64_MAX__ 18446744073709551615ULL
// RISCV32: #define __UINT64_TYPE__ long long unsigned int
// RISCV32: #define __UINT8_C_SUFFIX__
// RISCV32: #define __UINT8_MAX__ 255
// RISCV32: #define __UINT8_TYPE__ unsigned char
// RISCV32: #define __UINTMAX_C_SUFFIX__ ULL
// RISCV32: #define __UINTMAX_MAX__ 18446744073709551615ULL
// RISCV32: #define __UINTMAX_TYPE__ long long unsigned int
// RISCV32: #define __UINTMAX_WIDTH__ 64
// RISCV32: #define __UINTPTR_MAX__ 4294967295U
// RISCV32: #define __UINTPTR_TYPE__ unsigned int
// RISCV32: #define __UINTPTR_WIDTH__ 32
// TODO: RISC-V GCC defines UINT_FAST16 to be unsigned int
// RISCV32: #define __UINT_FAST16_MAX__ 65535
// RISCV32: #define __UINT_FAST16_TYPE__ unsigned short
// RISCV32: #define __UINT_FAST32_MAX__ 4294967295U
// RISCV32: #define __UINT_FAST32_TYPE__ unsigned int
// RISCV32: #define __UINT_FAST64_MAX__ 18446744073709551615ULL
// RISCV32: #define __UINT_FAST64_TYPE__ long long unsigned int
// TODO: RISC-V GCC defines UINT_FAST8 to be unsigned int
// RISCV32: #define __UINT_FAST8_MAX__ 255
// RISCV32: #define __UINT_FAST8_TYPE__ unsigned char
// RISCV32: #define __UINT_LEAST16_MAX__ 65535
// RISCV32: #define __UINT_LEAST16_TYPE__ unsigned short
// RISCV32: #define __UINT_LEAST32_MAX__ 4294967295U
// RISCV32: #define __UINT_LEAST32_TYPE__ unsigned int
// RISCV32: #define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// RISCV32: #define __UINT_LEAST64_TYPE__ long long unsigned int
// RISCV32: #define __UINT_LEAST8_MAX__ 255
// RISCV32: #define __UINT_LEAST8_TYPE__ unsigned char
// RISCV32: #define __USER_LABEL_PREFIX__
// RISCV32: #define __WCHAR_MAX__ 2147483647
// RISCV32: #define __WCHAR_TYPE__ int
// RISCV32: #define __WCHAR_WIDTH__ 32
// RISCV32: #define __WINT_TYPE__ unsigned int
// RISCV32: #define __WINT_UNSIGNED__ 1
// RISCV32: #define __WINT_WIDTH__ 32
// RISCV32-LINUX: #define __gnu_linux__ 1
// RISCV32-LINUX: #define __linux 1
// RISCV32-LINUX: #define __linux__ 1
// RISCV32: #define __riscv 1
// RISCV32: #define __riscv_cmodel_medlow 1
// RISCV32: #define __riscv_float_abi_soft 1
// RISCV32: #define __riscv_xlen 32
// RISCV32-LINUX: #define __unix 1
// RISCV32-LINUX: #define __unix__ 1
// RISCV32-LINUX: #define linux 1
// RISCV32-LINUX: #define unix 1

// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=riscv64 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=RISCV64 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -fgnuc-version=4.2.1 -triple=riscv64-unknown-linux < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefixes=RISCV64,RISCV64-LINUX %s
// RISCV64: #define _LP64 1
// RISCV64: #define __ATOMIC_ACQUIRE 2
// RISCV64: #define __ATOMIC_ACQ_REL 4
// RISCV64: #define __ATOMIC_CONSUME 1
// RISCV64: #define __ATOMIC_RELAXED 0
// RISCV64: #define __ATOMIC_RELEASE 3
// RISCV64: #define __ATOMIC_SEQ_CST 5
// RISCV64: #define __BIGGEST_ALIGNMENT__ 16
// RISCV64: #define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// RISCV64: #define __CHAR16_TYPE__ unsigned short
// RISCV64: #define __CHAR32_TYPE__ unsigned int
// RISCV64: #define __CHAR_BIT__ 8
// RISCV64: #define __DBL_DECIMAL_DIG__ 17
// RISCV64: #define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// RISCV64: #define __DBL_DIG__ 15
// RISCV64: #define __DBL_EPSILON__ 2.2204460492503131e-16
// RISCV64: #define __DBL_HAS_DENORM__ 1
// RISCV64: #define __DBL_HAS_INFINITY__ 1
// RISCV64: #define __DBL_HAS_QUIET_NAN__ 1
// RISCV64: #define __DBL_MANT_DIG__ 53
// RISCV64: #define __DBL_MAX_10_EXP__ 308
// RISCV64: #define __DBL_MAX_EXP__ 1024
// RISCV64: #define __DBL_MAX__ 1.7976931348623157e+308
// RISCV64: #define __DBL_MIN_10_EXP__ (-307)
// RISCV64: #define __DBL_MIN_EXP__ (-1021)
// RISCV64: #define __DBL_MIN__ 2.2250738585072014e-308
// RISCV64: #define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// RISCV64: #define __ELF__ 1
// RISCV64: #define __FINITE_MATH_ONLY__ 0
// RISCV64: #define __FLT_DECIMAL_DIG__ 9
// RISCV64: #define __FLT_DENORM_MIN__ 1.40129846e-45F
// RISCV64: #define __FLT_DIG__ 6
// RISCV64: #define __FLT_EPSILON__ 1.19209290e-7F
// RISCV64: #define __FLT_EVAL_METHOD__ 0
// RISCV64: #define __FLT_HAS_DENORM__ 1
// RISCV64: #define __FLT_HAS_INFINITY__ 1
// RISCV64: #define __FLT_HAS_QUIET_NAN__ 1
// RISCV64: #define __FLT_MANT_DIG__ 24
// RISCV64: #define __FLT_MAX_10_EXP__ 38
// RISCV64: #define __FLT_MAX_EXP__ 128
// RISCV64: #define __FLT_MAX__ 3.40282347e+38F
// RISCV64: #define __FLT_MIN_10_EXP__ (-37)
// RISCV64: #define __FLT_MIN_EXP__ (-125)
// RISCV64: #define __FLT_MIN__ 1.17549435e-38F
// RISCV64: #define __FLT_RADIX__ 2
// RISCV64: #define __GCC_ATOMIC_BOOL_LOCK_FREE 1
// RISCV64: #define __GCC_ATOMIC_CHAR16_T_LOCK_FREE 1
// RISCV64: #define __GCC_ATOMIC_CHAR32_T_LOCK_FREE 1
// RISCV64: #define __GCC_ATOMIC_CHAR_LOCK_FREE 1
// RISCV64: #define __GCC_ATOMIC_INT_LOCK_FREE 1
// RISCV64: #define __GCC_ATOMIC_LLONG_LOCK_FREE 1
// RISCV64: #define __GCC_ATOMIC_LONG_LOCK_FREE 1
// RISCV64: #define __GCC_ATOMIC_POINTER_LOCK_FREE 1
// RISCV64: #define __GCC_ATOMIC_SHORT_LOCK_FREE 1
// RISCV64: #define __GCC_ATOMIC_TEST_AND_SET_TRUEVAL 1
// RISCV64: #define __GCC_ATOMIC_WCHAR_T_LOCK_FREE 1
// RISCV64: #define __GNUC_MINOR__ {{.*}}
// RISCV64: #define __GNUC_PATCHLEVEL__ {{.*}}
// RISCV64: #define __GNUC_STDC_INLINE__ 1
// RISCV64: #define __GNUC__ {{.*}}
// RISCV64: #define __GXX_ABI_VERSION {{.*}}
// RISCV64: #define __INT16_C_SUFFIX__
// RISCV64: #define __INT16_MAX__ 32767
// RISCV64: #define __INT16_TYPE__ short
// RISCV64: #define __INT32_C_SUFFIX__
// RISCV64: #define __INT32_MAX__ 2147483647
// RISCV64: #define __INT32_TYPE__ int
// RISCV64: #define __INT64_C_SUFFIX__ L
// RISCV64: #define __INT64_MAX__ 9223372036854775807L
// RISCV64: #define __INT64_TYPE__ long int
// RISCV64: #define __INT8_C_SUFFIX__
// RISCV64: #define __INT8_MAX__ 127
// RISCV64: #define __INT8_TYPE__ signed char
// RISCV64: #define __INTMAX_C_SUFFIX__ L
// RISCV64: #define __INTMAX_MAX__ 9223372036854775807L
// RISCV64: #define __INTMAX_TYPE__ long int
// RISCV64: #define __INTMAX_WIDTH__ 64
// RISCV64: #define __INTPTR_MAX__ 9223372036854775807L
// RISCV64: #define __INTPTR_TYPE__ long int
// RISCV64: #define __INTPTR_WIDTH__ 64
// TODO: RISC-V GCC defines INT_FAST16 as int
// RISCV64: #define __INT_FAST16_MAX__ 32767
// RISCV64: #define __INT_FAST16_TYPE__ short
// RISCV64: #define __INT_FAST32_MAX__ 2147483647
// RISCV64: #define __INT_FAST32_TYPE__ int
// RISCV64: #define __INT_FAST64_MAX__ 9223372036854775807L
// RISCV64: #define __INT_FAST64_TYPE__ long int
// TODO: RISC-V GCC defines INT_FAST8 as int
// RISCV64: #define __INT_FAST8_MAX__ 127
// RISCV64: #define __INT_FAST8_TYPE__ signed char
// RISCV64: #define __INT_LEAST16_MAX__ 32767
// RISCV64: #define __INT_LEAST16_TYPE__ short
// RISCV64: #define __INT_LEAST32_MAX__ 2147483647
// RISCV64: #define __INT_LEAST32_TYPE__ int
// RISCV64: #define __INT_LEAST64_MAX__ 9223372036854775807L
// RISCV64: #define __INT_LEAST64_TYPE__ long int
// RISCV64: #define __INT_LEAST8_MAX__ 127
// RISCV64: #define __INT_LEAST8_TYPE__ signed char
// RISCV64: #define __INT_MAX__ 2147483647
// RISCV64: #define __LDBL_DECIMAL_DIG__ 36
// RISCV64: #define __LDBL_DENORM_MIN__ 6.47517511943802511092443895822764655e-4966L
// RISCV64: #define __LDBL_DIG__ 33
// RISCV64: #define __LDBL_EPSILON__ 1.92592994438723585305597794258492732e-34L
// RISCV64: #define __LDBL_HAS_DENORM__ 1
// RISCV64: #define __LDBL_HAS_INFINITY__ 1
// RISCV64: #define __LDBL_HAS_QUIET_NAN__ 1
// RISCV64: #define __LDBL_MANT_DIG__ 113
// RISCV64: #define __LDBL_MAX_10_EXP__ 4932
// RISCV64: #define __LDBL_MAX_EXP__ 16384
// RISCV64: #define __LDBL_MAX__ 1.18973149535723176508575932662800702e+4932L
// RISCV64: #define __LDBL_MIN_10_EXP__ (-4931)
// RISCV64: #define __LDBL_MIN_EXP__ (-16381)
// RISCV64: #define __LDBL_MIN__ 3.36210314311209350626267781732175260e-4932L
// RISCV64: #define __LITTLE_ENDIAN__ 1
// RISCV64: #define __LONG_LONG_MAX__ 9223372036854775807LL
// RISCV64: #define __LONG_MAX__ 9223372036854775807L
// RISCV64: #define __LP64__ 1
// RISCV64: #define __NO_INLINE__ 1
// RISCV64: #define __POINTER_WIDTH__ 64
// RISCV64: #define __PRAGMA_REDEFINE_EXTNAME 1
// RISCV64: #define __PTRDIFF_MAX__ 9223372036854775807L
// RISCV64: #define __PTRDIFF_TYPE__ long int
// RISCV64: #define __PTRDIFF_WIDTH__ 64
// RISCV64: #define __SCHAR_MAX__ 127
// RISCV64: #define __SHRT_MAX__ 32767
// RISCV64: #define __SIG_ATOMIC_MAX__ 2147483647
// RISCV64: #define __SIG_ATOMIC_WIDTH__ 32
// RISCV64: #define __SIZEOF_DOUBLE__ 8
// RISCV64: #define __SIZEOF_FLOAT__ 4
// RISCV64: #define __SIZEOF_INT__ 4
// RISCV64: #define __SIZEOF_LONG_DOUBLE__ 16
// RISCV64: #define __SIZEOF_LONG_LONG__ 8
// RISCV64: #define __SIZEOF_LONG__ 8
// RISCV64: #define __SIZEOF_POINTER__ 8
// RISCV64: #define __SIZEOF_PTRDIFF_T__ 8
// RISCV64: #define __SIZEOF_SHORT__ 2
// RISCV64: #define __SIZEOF_SIZE_T__ 8
// RISCV64: #define __SIZEOF_WCHAR_T__ 4
// RISCV64: #define __SIZEOF_WINT_T__ 4
// RISCV64: #define __SIZE_MAX__ 18446744073709551615UL
// RISCV64: #define __SIZE_TYPE__ long unsigned int
// RISCV64: #define __SIZE_WIDTH__ 64
// RISCV64: #define __STDC_HOSTED__ 0
// RISCV64: #define __STDC_UTF_16__ 1
// RISCV64: #define __STDC_UTF_32__ 1
// RISCV64: #define __STDC_VERSION__ 201710L
// RISCV64: #define __STDC__ 1
// RISCV64: #define __UINT16_C_SUFFIX__
// RISCV64: #define __UINT16_MAX__ 65535
// RISCV64: #define __UINT16_TYPE__ unsigned short
// RISCV64: #define __UINT32_C_SUFFIX__ U
// RISCV64: #define __UINT32_MAX__ 4294967295U
// RISCV64: #define __UINT32_TYPE__ unsigned int
// RISCV64: #define __UINT64_C_SUFFIX__ UL
// RISCV64: #define __UINT64_MAX__ 18446744073709551615UL
// RISCV64: #define __UINT64_TYPE__ long unsigned int
// RISCV64: #define __UINT8_C_SUFFIX__
// RISCV64: #define __UINT8_MAX__ 255
// RISCV64: #define __UINT8_TYPE__ unsigned char
// RISCV64: #define __UINTMAX_C_SUFFIX__ UL
// RISCV64: #define __UINTMAX_MAX__ 18446744073709551615UL
// RISCV64: #define __UINTMAX_TYPE__ long unsigned int
// RISCV64: #define __UINTMAX_WIDTH__ 64
// RISCV64: #define __UINTPTR_MAX__ 18446744073709551615UL
// RISCV64: #define __UINTPTR_TYPE__ long unsigned int
// RISCV64: #define __UINTPTR_WIDTH__ 64
// TODO: RISC-V GCC defines UINT_FAST16 to be unsigned int
// RISCV64: #define __UINT_FAST16_MAX__ 65535
// RISCV64: #define __UINT_FAST16_TYPE__ unsigned short
// RISCV64: #define __UINT_FAST32_MAX__ 4294967295U
// RISCV64: #define __UINT_FAST32_TYPE__ unsigned int
// RISCV64: #define __UINT_FAST64_MAX__ 18446744073709551615UL
// RISCV64: #define __UINT_FAST64_TYPE__ long unsigned int
// TODO: RISC-V GCC defines UINT_FAST8 to be unsigned int
// RISCV64: #define __UINT_FAST8_MAX__ 255
// RISCV64: #define __UINT_FAST8_TYPE__ unsigned char
// RISCV64: #define __UINT_LEAST16_MAX__ 65535
// RISCV64: #define __UINT_LEAST16_TYPE__ unsigned short
// RISCV64: #define __UINT_LEAST32_MAX__ 4294967295U
// RISCV64: #define __UINT_LEAST32_TYPE__ unsigned int
// RISCV64: #define __UINT_LEAST64_MAX__ 18446744073709551615UL
// RISCV64: #define __UINT_LEAST64_TYPE__ long unsigned int
// RISCV64: #define __UINT_LEAST8_MAX__ 255
// RISCV64: #define __UINT_LEAST8_TYPE__ unsigned char
// RISCV64: #define __USER_LABEL_PREFIX__
// RISCV64: #define __WCHAR_MAX__ 2147483647
// RISCV64: #define __WCHAR_TYPE__ int
// RISCV64: #define __WCHAR_WIDTH__ 32
// RISCV64: #define __WINT_TYPE__ unsigned int
// RISCV64: #define __WINT_UNSIGNED__ 1
// RISCV64: #define __WINT_WIDTH__ 32
// RISCV64-LINUX: #define __gnu_linux__ 1
// RISCV64-LINUX: #define __linux 1
// RISCV64-LINUX: #define __linux__ 1
// RISCV64: #define __riscv 1
// RISCV64: #define __riscv_cmodel_medlow 1
// RISCV64: #define __riscv_float_abi_soft 1
// RISCV64: #define __riscv_xlen 64
// RISCV64-LINUX: #define __unix 1
// RISCV64-LINUX: #define __unix__ 1
// RISCV64-LINUX: #define linux 1
// RISCV64-LINUX: #define unix 1
