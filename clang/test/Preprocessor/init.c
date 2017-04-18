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
// RUN: %clang_cc1 -x c++ -std=c++1z -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix CXX1Z %s
//
// CXX1Z:#define __GNUG__ {{.*}}
// CXX1Z:#define __GXX_EXPERIMENTAL_CXX0X__ 1
// CXX1Z:#define __GXX_RTTI 1
// CXX1Z:#define __GXX_WEAK__ 1
// CXX1Z:#define __cplusplus 201703L
// CXX1Z:#define __private_extern__ extern
//
//
// RUN: %clang_cc1 -x c++ -std=c++1y -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix CXX1Y %s
//
// CXX1Y:#define __GNUG__ {{.*}}
// CXX1Y:#define __GXX_EXPERIMENTAL_CXX0X__ 1
// CXX1Y:#define __GXX_RTTI 1
// CXX1Y:#define __GXX_WEAK__ 1
// CXX1Y:#define __cplusplus 201402L
// CXX1Y:#define __private_extern__ extern
//
//
// RUN: %clang_cc1 -x c++ -std=c++11 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix CXX11 %s
//
// CXX11:#define __GNUG__ {{.*}}
// CXX11:#define __GXX_EXPERIMENTAL_CXX0X__ 1
// CXX11:#define __GXX_RTTI 1
// CXX11:#define __GXX_WEAK__ 1
// CXX11:#define __cplusplus 201103L
// CXX11:#define __private_extern__ extern
//
// 
// RUN: %clang_cc1 -x c++ -std=c++98 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix CXX98 %s
// 
// CXX98:#define __GNUG__ {{.*}}
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
// RUN: %clang_cc1 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix COMMON %s
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
// C-DEFAULT:#define __STDC_VERSION__ 201112L
//
// RUN: %clang_cc1 -ffreestanding -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix FREESTANDING %s
// FREESTANDING:#define __STDC_HOSTED__ 0
//
//
// RUN: %clang_cc1 -x c++ -std=gnu++1z -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix GXX1Z %s
//
// GXX1Z:#define __GNUG__ {{.*}}
// GXX1Z:#define __GXX_WEAK__ 1
// GXX1Z:#define __cplusplus 201703L
// GXX1Z:#define __private_extern__ extern
//
//
// RUN: %clang_cc1 -x c++ -std=gnu++1y -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix GXX1Y %s
//
// GXX1Y:#define __GNUG__ {{.*}}
// GXX1Y:#define __GXX_WEAK__ 1
// GXX1Y:#define __cplusplus 201402L
// GXX1Y:#define __private_extern__ extern
//
//
// RUN: %clang_cc1 -x c++ -std=gnu++11 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix GXX11 %s
//
// GXX11:#define __GNUG__ {{.*}}
// GXX11:#define __GXX_WEAK__ 1
// GXX11:#define __cplusplus 201103L
// GXX11:#define __private_extern__ extern
//
//
// RUN: %clang_cc1 -x c++ -std=gnu++98 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix GXX98 %s
//
// GXX98:#define __GNUG__ {{.*}}
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
// Og	:#define __OPTIMIZE__ 1
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
// RUN: %clang_cc1 -E -dM -fshort-wchar < /dev/null | FileCheck -match-full-lines -check-prefix SHORTWCHAR %s
// wchar_t is u16 for targeting Win32.
// FIXME: Implement and check x86_64-cygwin.
// RUN: %clang_cc1 -E -dM -fno-short-wchar -triple=x86_64-w64-mingw32 < /dev/null | FileCheck -match-full-lines -check-prefix SHORTWCHAR %s
//
// SHORTWCHAR: #define __SIZEOF_WCHAR_T__ 2
// SHORTWCHAR: #define __WCHAR_MAX__ 65535
// SHORTWCHAR: #define __WCHAR_TYPE__ unsigned short
// SHORTWCHAR: #define __WCHAR_WIDTH__ 16
//
// RUN: %clang_cc1 -E -dM -fno-short-wchar -triple=i686-unknown-unknown < /dev/null | FileCheck -match-full-lines -check-prefix SHORTWCHAR2 %s
// RUN: %clang_cc1 -E -dM -fno-short-wchar -triple=x86_64-unknown-unknown < /dev/null | FileCheck -match-full-lines -check-prefix SHORTWCHAR2 %s
//
// SHORTWCHAR2: #define __SIZEOF_WCHAR_T__ 4
// SHORTWCHAR2: #define __WCHAR_WIDTH__ 32
// Other definitions vary from platform to platform

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=aarch64-none-none < /dev/null | FileCheck -match-full-lines -check-prefix AARCH64 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=arm64-none-none < /dev/null | FileCheck -match-full-lines -check-prefix AARCH64 %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=arm64-none-none < /dev/null | FileCheck -match-full-lines -check-prefix AARCH64 -check-prefix AARCH64-CXX %s
//
// AARCH64:#define _LP64 1
// AARCH64-NOT:#define __AARCH64EB__ 1
// AARCH64:#define __AARCH64EL__ 1
// AARCH64-NOT:#define __AARCH_BIG_ENDIAN 1
// AARCH64:#define __ARM_64BIT_STATE 1
// AARCH64:#define __ARM_ARCH 8
// AARCH64:#define __ARM_ARCH_ISA_A64 1
// AARCH64-NOT:#define __ARM_BIG_ENDIAN 1
// AARCH64:#define __BIGGEST_ALIGNMENT__ 16
// AARCH64:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// AARCH64:#define __CHAR16_TYPE__ unsigned short
// AARCH64:#define __CHAR32_TYPE__ unsigned int
// AARCH64:#define __CHAR_BIT__ 8
// AARCH64:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// AARCH64:#define __DBL_DIG__ 15
// AARCH64:#define __DBL_EPSILON__ 2.2204460492503131e-16
// AARCH64:#define __DBL_HAS_DENORM__ 1
// AARCH64:#define __DBL_HAS_INFINITY__ 1
// AARCH64:#define __DBL_HAS_QUIET_NAN__ 1
// AARCH64:#define __DBL_MANT_DIG__ 53
// AARCH64:#define __DBL_MAX_10_EXP__ 308
// AARCH64:#define __DBL_MAX_EXP__ 1024
// AARCH64:#define __DBL_MAX__ 1.7976931348623157e+308
// AARCH64:#define __DBL_MIN_10_EXP__ (-307)
// AARCH64:#define __DBL_MIN_EXP__ (-1021)
// AARCH64:#define __DBL_MIN__ 2.2250738585072014e-308
// AARCH64:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// AARCH64:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// AARCH64:#define __FLT_DIG__ 6
// AARCH64:#define __FLT_EPSILON__ 1.19209290e-7F
// AARCH64:#define __FLT_EVAL_METHOD__ 0
// AARCH64:#define __FLT_HAS_DENORM__ 1
// AARCH64:#define __FLT_HAS_INFINITY__ 1
// AARCH64:#define __FLT_HAS_QUIET_NAN__ 1
// AARCH64:#define __FLT_MANT_DIG__ 24
// AARCH64:#define __FLT_MAX_10_EXP__ 38
// AARCH64:#define __FLT_MAX_EXP__ 128
// AARCH64:#define __FLT_MAX__ 3.40282347e+38F
// AARCH64:#define __FLT_MIN_10_EXP__ (-37)
// AARCH64:#define __FLT_MIN_EXP__ (-125)
// AARCH64:#define __FLT_MIN__ 1.17549435e-38F
// AARCH64:#define __FLT_RADIX__ 2
// AARCH64:#define __INT16_C_SUFFIX__
// AARCH64:#define __INT16_FMTd__ "hd"
// AARCH64:#define __INT16_FMTi__ "hi"
// AARCH64:#define __INT16_MAX__ 32767
// AARCH64:#define __INT16_TYPE__ short
// AARCH64:#define __INT32_C_SUFFIX__
// AARCH64:#define __INT32_FMTd__ "d"
// AARCH64:#define __INT32_FMTi__ "i"
// AARCH64:#define __INT32_MAX__ 2147483647
// AARCH64:#define __INT32_TYPE__ int
// AARCH64:#define __INT64_C_SUFFIX__ L
// AARCH64:#define __INT64_FMTd__ "ld"
// AARCH64:#define __INT64_FMTi__ "li"
// AARCH64:#define __INT64_MAX__ 9223372036854775807L
// AARCH64:#define __INT64_TYPE__ long int
// AARCH64:#define __INT8_C_SUFFIX__
// AARCH64:#define __INT8_FMTd__ "hhd"
// AARCH64:#define __INT8_FMTi__ "hhi"
// AARCH64:#define __INT8_MAX__ 127
// AARCH64:#define __INT8_TYPE__ signed char
// AARCH64:#define __INTMAX_C_SUFFIX__ L
// AARCH64:#define __INTMAX_FMTd__ "ld"
// AARCH64:#define __INTMAX_FMTi__ "li"
// AARCH64:#define __INTMAX_MAX__ 9223372036854775807L
// AARCH64:#define __INTMAX_TYPE__ long int
// AARCH64:#define __INTMAX_WIDTH__ 64
// AARCH64:#define __INTPTR_FMTd__ "ld"
// AARCH64:#define __INTPTR_FMTi__ "li"
// AARCH64:#define __INTPTR_MAX__ 9223372036854775807L
// AARCH64:#define __INTPTR_TYPE__ long int
// AARCH64:#define __INTPTR_WIDTH__ 64
// AARCH64:#define __INT_FAST16_FMTd__ "hd"
// AARCH64:#define __INT_FAST16_FMTi__ "hi"
// AARCH64:#define __INT_FAST16_MAX__ 32767
// AARCH64:#define __INT_FAST16_TYPE__ short
// AARCH64:#define __INT_FAST32_FMTd__ "d"
// AARCH64:#define __INT_FAST32_FMTi__ "i"
// AARCH64:#define __INT_FAST32_MAX__ 2147483647
// AARCH64:#define __INT_FAST32_TYPE__ int
// AARCH64:#define __INT_FAST64_FMTd__ "ld"
// AARCH64:#define __INT_FAST64_FMTi__ "li"
// AARCH64:#define __INT_FAST64_MAX__ 9223372036854775807L
// AARCH64:#define __INT_FAST64_TYPE__ long int
// AARCH64:#define __INT_FAST8_FMTd__ "hhd"
// AARCH64:#define __INT_FAST8_FMTi__ "hhi"
// AARCH64:#define __INT_FAST8_MAX__ 127
// AARCH64:#define __INT_FAST8_TYPE__ signed char
// AARCH64:#define __INT_LEAST16_FMTd__ "hd"
// AARCH64:#define __INT_LEAST16_FMTi__ "hi"
// AARCH64:#define __INT_LEAST16_MAX__ 32767
// AARCH64:#define __INT_LEAST16_TYPE__ short
// AARCH64:#define __INT_LEAST32_FMTd__ "d"
// AARCH64:#define __INT_LEAST32_FMTi__ "i"
// AARCH64:#define __INT_LEAST32_MAX__ 2147483647
// AARCH64:#define __INT_LEAST32_TYPE__ int
// AARCH64:#define __INT_LEAST64_FMTd__ "ld"
// AARCH64:#define __INT_LEAST64_FMTi__ "li"
// AARCH64:#define __INT_LEAST64_MAX__ 9223372036854775807L
// AARCH64:#define __INT_LEAST64_TYPE__ long int
// AARCH64:#define __INT_LEAST8_FMTd__ "hhd"
// AARCH64:#define __INT_LEAST8_FMTi__ "hhi"
// AARCH64:#define __INT_LEAST8_MAX__ 127
// AARCH64:#define __INT_LEAST8_TYPE__ signed char
// AARCH64:#define __INT_MAX__ 2147483647
// AARCH64:#define __LDBL_DENORM_MIN__ 6.47517511943802511092443895822764655e-4966L
// AARCH64:#define __LDBL_DIG__ 33
// AARCH64:#define __LDBL_EPSILON__ 1.92592994438723585305597794258492732e-34L
// AARCH64:#define __LDBL_HAS_DENORM__ 1
// AARCH64:#define __LDBL_HAS_INFINITY__ 1
// AARCH64:#define __LDBL_HAS_QUIET_NAN__ 1
// AARCH64:#define __LDBL_MANT_DIG__ 113
// AARCH64:#define __LDBL_MAX_10_EXP__ 4932
// AARCH64:#define __LDBL_MAX_EXP__ 16384
// AARCH64:#define __LDBL_MAX__ 1.18973149535723176508575932662800702e+4932L
// AARCH64:#define __LDBL_MIN_10_EXP__ (-4931)
// AARCH64:#define __LDBL_MIN_EXP__ (-16381)
// AARCH64:#define __LDBL_MIN__ 3.36210314311209350626267781732175260e-4932L
// AARCH64:#define __LONG_LONG_MAX__ 9223372036854775807LL
// AARCH64:#define __LONG_MAX__ 9223372036854775807L
// AARCH64:#define __LP64__ 1
// AARCH64:#define __POINTER_WIDTH__ 64
// AARCH64:#define __PTRDIFF_TYPE__ long int
// AARCH64:#define __PTRDIFF_WIDTH__ 64
// AARCH64:#define __SCHAR_MAX__ 127
// AARCH64:#define __SHRT_MAX__ 32767
// AARCH64:#define __SIG_ATOMIC_MAX__ 2147483647
// AARCH64:#define __SIG_ATOMIC_WIDTH__ 32
// AARCH64:#define __SIZEOF_DOUBLE__ 8
// AARCH64:#define __SIZEOF_FLOAT__ 4
// AARCH64:#define __SIZEOF_INT128__ 16
// AARCH64:#define __SIZEOF_INT__ 4
// AARCH64:#define __SIZEOF_LONG_DOUBLE__ 16
// AARCH64:#define __SIZEOF_LONG_LONG__ 8
// AARCH64:#define __SIZEOF_LONG__ 8
// AARCH64:#define __SIZEOF_POINTER__ 8
// AARCH64:#define __SIZEOF_PTRDIFF_T__ 8
// AARCH64:#define __SIZEOF_SHORT__ 2
// AARCH64:#define __SIZEOF_SIZE_T__ 8
// AARCH64:#define __SIZEOF_WCHAR_T__ 4
// AARCH64:#define __SIZEOF_WINT_T__ 4
// AARCH64:#define __SIZE_MAX__ 18446744073709551615UL
// AARCH64:#define __SIZE_TYPE__ long unsigned int
// AARCH64:#define __SIZE_WIDTH__ 64
// AARCH64-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 16UL
// AARCH64:#define __UINT16_C_SUFFIX__
// AARCH64:#define __UINT16_MAX__ 65535
// AARCH64:#define __UINT16_TYPE__ unsigned short
// AARCH64:#define __UINT32_C_SUFFIX__ U
// AARCH64:#define __UINT32_MAX__ 4294967295U
// AARCH64:#define __UINT32_TYPE__ unsigned int
// AARCH64:#define __UINT64_C_SUFFIX__ UL
// AARCH64:#define __UINT64_MAX__ 18446744073709551615UL
// AARCH64:#define __UINT64_TYPE__ long unsigned int
// AARCH64:#define __UINT8_C_SUFFIX__
// AARCH64:#define __UINT8_MAX__ 255
// AARCH64:#define __UINT8_TYPE__ unsigned char
// AARCH64:#define __UINTMAX_C_SUFFIX__ UL
// AARCH64:#define __UINTMAX_MAX__ 18446744073709551615UL
// AARCH64:#define __UINTMAX_TYPE__ long unsigned int
// AARCH64:#define __UINTMAX_WIDTH__ 64
// AARCH64:#define __UINTPTR_MAX__ 18446744073709551615UL
// AARCH64:#define __UINTPTR_TYPE__ long unsigned int
// AARCH64:#define __UINTPTR_WIDTH__ 64
// AARCH64:#define __UINT_FAST16_MAX__ 65535
// AARCH64:#define __UINT_FAST16_TYPE__ unsigned short
// AARCH64:#define __UINT_FAST32_MAX__ 4294967295U
// AARCH64:#define __UINT_FAST32_TYPE__ unsigned int
// AARCH64:#define __UINT_FAST64_MAX__ 18446744073709551615UL
// AARCH64:#define __UINT_FAST64_TYPE__ long unsigned int
// AARCH64:#define __UINT_FAST8_MAX__ 255
// AARCH64:#define __UINT_FAST8_TYPE__ unsigned char
// AARCH64:#define __UINT_LEAST16_MAX__ 65535
// AARCH64:#define __UINT_LEAST16_TYPE__ unsigned short
// AARCH64:#define __UINT_LEAST32_MAX__ 4294967295U
// AARCH64:#define __UINT_LEAST32_TYPE__ unsigned int
// AARCH64:#define __UINT_LEAST64_MAX__ 18446744073709551615UL
// AARCH64:#define __UINT_LEAST64_TYPE__ long unsigned int
// AARCH64:#define __UINT_LEAST8_MAX__ 255
// AARCH64:#define __UINT_LEAST8_TYPE__ unsigned char
// AARCH64:#define __USER_LABEL_PREFIX__
// AARCH64:#define __WCHAR_MAX__ 4294967295U
// AARCH64:#define __WCHAR_TYPE__ unsigned int
// AARCH64:#define __WCHAR_UNSIGNED__ 1
// AARCH64:#define __WCHAR_WIDTH__ 32
// AARCH64:#define __WINT_TYPE__ int
// AARCH64:#define __WINT_WIDTH__ 32
// AARCH64:#define __aarch64__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=aarch64_be-none-none < /dev/null | FileCheck -match-full-lines -check-prefix AARCH64-BE %s
//
// AARCH64-BE:#define _LP64 1
// AARCH64-BE:#define __AARCH64EB__ 1
// AARCH64-BE-NOT:#define __AARCH64EL__ 1
// AARCH64-BE:#define __AARCH_BIG_ENDIAN 1
// AARCH64-BE:#define __ARM_64BIT_STATE 1
// AARCH64-BE:#define __ARM_ARCH 8
// AARCH64-BE:#define __ARM_ARCH_ISA_A64 1
// AARCH64-BE:#define __ARM_BIG_ENDIAN 1
// AARCH64-BE:#define __BIGGEST_ALIGNMENT__ 16
// AARCH64-BE:#define __BYTE_ORDER__ __ORDER_BIG_ENDIAN__
// AARCH64-BE:#define __CHAR16_TYPE__ unsigned short
// AARCH64-BE:#define __CHAR32_TYPE__ unsigned int
// AARCH64-BE:#define __CHAR_BIT__ 8
// AARCH64-BE:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// AARCH64-BE:#define __DBL_DIG__ 15
// AARCH64-BE:#define __DBL_EPSILON__ 2.2204460492503131e-16
// AARCH64-BE:#define __DBL_HAS_DENORM__ 1
// AARCH64-BE:#define __DBL_HAS_INFINITY__ 1
// AARCH64-BE:#define __DBL_HAS_QUIET_NAN__ 1
// AARCH64-BE:#define __DBL_MANT_DIG__ 53
// AARCH64-BE:#define __DBL_MAX_10_EXP__ 308
// AARCH64-BE:#define __DBL_MAX_EXP__ 1024
// AARCH64-BE:#define __DBL_MAX__ 1.7976931348623157e+308
// AARCH64-BE:#define __DBL_MIN_10_EXP__ (-307)
// AARCH64-BE:#define __DBL_MIN_EXP__ (-1021)
// AARCH64-BE:#define __DBL_MIN__ 2.2250738585072014e-308
// AARCH64-BE:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// AARCH64-BE:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// AARCH64-BE:#define __FLT_DIG__ 6
// AARCH64-BE:#define __FLT_EPSILON__ 1.19209290e-7F
// AARCH64-BE:#define __FLT_EVAL_METHOD__ 0
// AARCH64-BE:#define __FLT_HAS_DENORM__ 1
// AARCH64-BE:#define __FLT_HAS_INFINITY__ 1
// AARCH64-BE:#define __FLT_HAS_QUIET_NAN__ 1
// AARCH64-BE:#define __FLT_MANT_DIG__ 24
// AARCH64-BE:#define __FLT_MAX_10_EXP__ 38
// AARCH64-BE:#define __FLT_MAX_EXP__ 128
// AARCH64-BE:#define __FLT_MAX__ 3.40282347e+38F
// AARCH64-BE:#define __FLT_MIN_10_EXP__ (-37)
// AARCH64-BE:#define __FLT_MIN_EXP__ (-125)
// AARCH64-BE:#define __FLT_MIN__ 1.17549435e-38F
// AARCH64-BE:#define __FLT_RADIX__ 2
// AARCH64-BE:#define __INT16_C_SUFFIX__
// AARCH64-BE:#define __INT16_FMTd__ "hd"
// AARCH64-BE:#define __INT16_FMTi__ "hi"
// AARCH64-BE:#define __INT16_MAX__ 32767
// AARCH64-BE:#define __INT16_TYPE__ short
// AARCH64-BE:#define __INT32_C_SUFFIX__
// AARCH64-BE:#define __INT32_FMTd__ "d"
// AARCH64-BE:#define __INT32_FMTi__ "i"
// AARCH64-BE:#define __INT32_MAX__ 2147483647
// AARCH64-BE:#define __INT32_TYPE__ int
// AARCH64-BE:#define __INT64_C_SUFFIX__ L
// AARCH64-BE:#define __INT64_FMTd__ "ld"
// AARCH64-BE:#define __INT64_FMTi__ "li"
// AARCH64-BE:#define __INT64_MAX__ 9223372036854775807L
// AARCH64-BE:#define __INT64_TYPE__ long int
// AARCH64-BE:#define __INT8_C_SUFFIX__
// AARCH64-BE:#define __INT8_FMTd__ "hhd"
// AARCH64-BE:#define __INT8_FMTi__ "hhi"
// AARCH64-BE:#define __INT8_MAX__ 127
// AARCH64-BE:#define __INT8_TYPE__ signed char
// AARCH64-BE:#define __INTMAX_C_SUFFIX__ L
// AARCH64-BE:#define __INTMAX_FMTd__ "ld"
// AARCH64-BE:#define __INTMAX_FMTi__ "li"
// AARCH64-BE:#define __INTMAX_MAX__ 9223372036854775807L
// AARCH64-BE:#define __INTMAX_TYPE__ long int
// AARCH64-BE:#define __INTMAX_WIDTH__ 64
// AARCH64-BE:#define __INTPTR_FMTd__ "ld"
// AARCH64-BE:#define __INTPTR_FMTi__ "li"
// AARCH64-BE:#define __INTPTR_MAX__ 9223372036854775807L
// AARCH64-BE:#define __INTPTR_TYPE__ long int
// AARCH64-BE:#define __INTPTR_WIDTH__ 64
// AARCH64-BE:#define __INT_FAST16_FMTd__ "hd"
// AARCH64-BE:#define __INT_FAST16_FMTi__ "hi"
// AARCH64-BE:#define __INT_FAST16_MAX__ 32767
// AARCH64-BE:#define __INT_FAST16_TYPE__ short
// AARCH64-BE:#define __INT_FAST32_FMTd__ "d"
// AARCH64-BE:#define __INT_FAST32_FMTi__ "i"
// AARCH64-BE:#define __INT_FAST32_MAX__ 2147483647
// AARCH64-BE:#define __INT_FAST32_TYPE__ int
// AARCH64-BE:#define __INT_FAST64_FMTd__ "ld"
// AARCH64-BE:#define __INT_FAST64_FMTi__ "li"
// AARCH64-BE:#define __INT_FAST64_MAX__ 9223372036854775807L
// AARCH64-BE:#define __INT_FAST64_TYPE__ long int
// AARCH64-BE:#define __INT_FAST8_FMTd__ "hhd"
// AARCH64-BE:#define __INT_FAST8_FMTi__ "hhi"
// AARCH64-BE:#define __INT_FAST8_MAX__ 127
// AARCH64-BE:#define __INT_FAST8_TYPE__ signed char
// AARCH64-BE:#define __INT_LEAST16_FMTd__ "hd"
// AARCH64-BE:#define __INT_LEAST16_FMTi__ "hi"
// AARCH64-BE:#define __INT_LEAST16_MAX__ 32767
// AARCH64-BE:#define __INT_LEAST16_TYPE__ short
// AARCH64-BE:#define __INT_LEAST32_FMTd__ "d"
// AARCH64-BE:#define __INT_LEAST32_FMTi__ "i"
// AARCH64-BE:#define __INT_LEAST32_MAX__ 2147483647
// AARCH64-BE:#define __INT_LEAST32_TYPE__ int
// AARCH64-BE:#define __INT_LEAST64_FMTd__ "ld"
// AARCH64-BE:#define __INT_LEAST64_FMTi__ "li"
// AARCH64-BE:#define __INT_LEAST64_MAX__ 9223372036854775807L
// AARCH64-BE:#define __INT_LEAST64_TYPE__ long int
// AARCH64-BE:#define __INT_LEAST8_FMTd__ "hhd"
// AARCH64-BE:#define __INT_LEAST8_FMTi__ "hhi"
// AARCH64-BE:#define __INT_LEAST8_MAX__ 127
// AARCH64-BE:#define __INT_LEAST8_TYPE__ signed char
// AARCH64-BE:#define __INT_MAX__ 2147483647
// AARCH64-BE:#define __LDBL_DENORM_MIN__ 6.47517511943802511092443895822764655e-4966L
// AARCH64-BE:#define __LDBL_DIG__ 33
// AARCH64-BE:#define __LDBL_EPSILON__ 1.92592994438723585305597794258492732e-34L
// AARCH64-BE:#define __LDBL_HAS_DENORM__ 1
// AARCH64-BE:#define __LDBL_HAS_INFINITY__ 1
// AARCH64-BE:#define __LDBL_HAS_QUIET_NAN__ 1
// AARCH64-BE:#define __LDBL_MANT_DIG__ 113
// AARCH64-BE:#define __LDBL_MAX_10_EXP__ 4932
// AARCH64-BE:#define __LDBL_MAX_EXP__ 16384
// AARCH64-BE:#define __LDBL_MAX__ 1.18973149535723176508575932662800702e+4932L
// AARCH64-BE:#define __LDBL_MIN_10_EXP__ (-4931)
// AARCH64-BE:#define __LDBL_MIN_EXP__ (-16381)
// AARCH64-BE:#define __LDBL_MIN__ 3.36210314311209350626267781732175260e-4932L
// AARCH64-BE:#define __LONG_LONG_MAX__ 9223372036854775807LL
// AARCH64-BE:#define __LONG_MAX__ 9223372036854775807L
// AARCH64-BE:#define __LP64__ 1
// AARCH64-BE:#define __POINTER_WIDTH__ 64
// AARCH64-BE:#define __PTRDIFF_TYPE__ long int
// AARCH64-BE:#define __PTRDIFF_WIDTH__ 64
// AARCH64-BE:#define __SCHAR_MAX__ 127
// AARCH64-BE:#define __SHRT_MAX__ 32767
// AARCH64-BE:#define __SIG_ATOMIC_MAX__ 2147483647
// AARCH64-BE:#define __SIG_ATOMIC_WIDTH__ 32
// AARCH64-BE:#define __SIZEOF_DOUBLE__ 8
// AARCH64-BE:#define __SIZEOF_FLOAT__ 4
// AARCH64-BE:#define __SIZEOF_INT128__ 16
// AARCH64-BE:#define __SIZEOF_INT__ 4
// AARCH64-BE:#define __SIZEOF_LONG_DOUBLE__ 16
// AARCH64-BE:#define __SIZEOF_LONG_LONG__ 8
// AARCH64-BE:#define __SIZEOF_LONG__ 8
// AARCH64-BE:#define __SIZEOF_POINTER__ 8
// AARCH64-BE:#define __SIZEOF_PTRDIFF_T__ 8
// AARCH64-BE:#define __SIZEOF_SHORT__ 2
// AARCH64-BE:#define __SIZEOF_SIZE_T__ 8
// AARCH64-BE:#define __SIZEOF_WCHAR_T__ 4
// AARCH64-BE:#define __SIZEOF_WINT_T__ 4
// AARCH64-BE:#define __SIZE_MAX__ 18446744073709551615UL
// AARCH64-BE:#define __SIZE_TYPE__ long unsigned int
// AARCH64-BE:#define __SIZE_WIDTH__ 64
// AARCH64-BE:#define __UINT16_C_SUFFIX__
// AARCH64-BE:#define __UINT16_MAX__ 65535
// AARCH64-BE:#define __UINT16_TYPE__ unsigned short
// AARCH64-BE:#define __UINT32_C_SUFFIX__ U
// AARCH64-BE:#define __UINT32_MAX__ 4294967295U
// AARCH64-BE:#define __UINT32_TYPE__ unsigned int
// AARCH64-BE:#define __UINT64_C_SUFFIX__ UL
// AARCH64-BE:#define __UINT64_MAX__ 18446744073709551615UL
// AARCH64-BE:#define __UINT64_TYPE__ long unsigned int
// AARCH64-BE:#define __UINT8_C_SUFFIX__
// AARCH64-BE:#define __UINT8_MAX__ 255
// AARCH64-BE:#define __UINT8_TYPE__ unsigned char
// AARCH64-BE:#define __UINTMAX_C_SUFFIX__ UL
// AARCH64-BE:#define __UINTMAX_MAX__ 18446744073709551615UL
// AARCH64-BE:#define __UINTMAX_TYPE__ long unsigned int
// AARCH64-BE:#define __UINTMAX_WIDTH__ 64
// AARCH64-BE:#define __UINTPTR_MAX__ 18446744073709551615UL
// AARCH64-BE:#define __UINTPTR_TYPE__ long unsigned int
// AARCH64-BE:#define __UINTPTR_WIDTH__ 64
// AARCH64-BE:#define __UINT_FAST16_MAX__ 65535
// AARCH64-BE:#define __UINT_FAST16_TYPE__ unsigned short
// AARCH64-BE:#define __UINT_FAST32_MAX__ 4294967295U
// AARCH64-BE:#define __UINT_FAST32_TYPE__ unsigned int
// AARCH64-BE:#define __UINT_FAST64_MAX__ 18446744073709551615UL
// AARCH64-BE:#define __UINT_FAST64_TYPE__ long unsigned int
// AARCH64-BE:#define __UINT_FAST8_MAX__ 255
// AARCH64-BE:#define __UINT_FAST8_TYPE__ unsigned char
// AARCH64-BE:#define __UINT_LEAST16_MAX__ 65535
// AARCH64-BE:#define __UINT_LEAST16_TYPE__ unsigned short
// AARCH64-BE:#define __UINT_LEAST32_MAX__ 4294967295U
// AARCH64-BE:#define __UINT_LEAST32_TYPE__ unsigned int
// AARCH64-BE:#define __UINT_LEAST64_MAX__ 18446744073709551615UL
// AARCH64-BE:#define __UINT_LEAST64_TYPE__ long unsigned int
// AARCH64-BE:#define __UINT_LEAST8_MAX__ 255
// AARCH64-BE:#define __UINT_LEAST8_TYPE__ unsigned char
// AARCH64-BE:#define __USER_LABEL_PREFIX__
// AARCH64-BE:#define __WCHAR_MAX__ 4294967295U
// AARCH64-BE:#define __WCHAR_TYPE__ unsigned int
// AARCH64-BE:#define __WCHAR_UNSIGNED__ 1
// AARCH64-BE:#define __WCHAR_WIDTH__ 32
// AARCH64-BE:#define __WINT_TYPE__ int
// AARCH64-BE:#define __WINT_WIDTH__ 32
// AARCH64-BE:#define __aarch64__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=aarch64-netbsd < /dev/null | FileCheck -match-full-lines -check-prefix AARCH64-NETBSD %s
//
// AARCH64-NETBSD:#define _LP64 1
// AARCH64-NETBSD-NOT:#define __AARCH64EB__ 1
// AARCH64-NETBSD:#define __AARCH64EL__ 1
// AARCH64-NETBSD-NOT:#define __AARCH_BIG_ENDIAN 1
// AARCH64-NETBSD:#define __ARM_64BIT_STATE 1
// AARCH64-NETBSD:#define __ARM_ARCH 8
// AARCH64-NETBSD:#define __ARM_ARCH_ISA_A64 1
// AARCH64-NETBSD-NOT:#define __ARM_BIG_ENDIAN 1
// AARCH64-NETBSD:#define __BIGGEST_ALIGNMENT__ 16
// AARCH64-NETBSD:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// AARCH64-NETBSD:#define __CHAR16_TYPE__ unsigned short
// AARCH64-NETBSD:#define __CHAR32_TYPE__ unsigned int
// AARCH64-NETBSD:#define __CHAR_BIT__ 8
// AARCH64-NETBSD:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// AARCH64-NETBSD:#define __DBL_DIG__ 15
// AARCH64-NETBSD:#define __DBL_EPSILON__ 2.2204460492503131e-16
// AARCH64-NETBSD:#define __DBL_HAS_DENORM__ 1
// AARCH64-NETBSD:#define __DBL_HAS_INFINITY__ 1
// AARCH64-NETBSD:#define __DBL_HAS_QUIET_NAN__ 1
// AARCH64-NETBSD:#define __DBL_MANT_DIG__ 53
// AARCH64-NETBSD:#define __DBL_MAX_10_EXP__ 308
// AARCH64-NETBSD:#define __DBL_MAX_EXP__ 1024
// AARCH64-NETBSD:#define __DBL_MAX__ 1.7976931348623157e+308
// AARCH64-NETBSD:#define __DBL_MIN_10_EXP__ (-307)
// AARCH64-NETBSD:#define __DBL_MIN_EXP__ (-1021)
// AARCH64-NETBSD:#define __DBL_MIN__ 2.2250738585072014e-308
// AARCH64-NETBSD:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// AARCH64-NETBSD:#define __ELF__ 1
// AARCH64-NETBSD:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// AARCH64-NETBSD:#define __FLT_DIG__ 6
// AARCH64-NETBSD:#define __FLT_EPSILON__ 1.19209290e-7F
// AARCH64-NETBSD:#define __FLT_EVAL_METHOD__ 0
// AARCH64-NETBSD:#define __FLT_HAS_DENORM__ 1
// AARCH64-NETBSD:#define __FLT_HAS_INFINITY__ 1
// AARCH64-NETBSD:#define __FLT_HAS_QUIET_NAN__ 1
// AARCH64-NETBSD:#define __FLT_MANT_DIG__ 24
// AARCH64-NETBSD:#define __FLT_MAX_10_EXP__ 38
// AARCH64-NETBSD:#define __FLT_MAX_EXP__ 128
// AARCH64-NETBSD:#define __FLT_MAX__ 3.40282347e+38F
// AARCH64-NETBSD:#define __FLT_MIN_10_EXP__ (-37)
// AARCH64-NETBSD:#define __FLT_MIN_EXP__ (-125)
// AARCH64-NETBSD:#define __FLT_MIN__ 1.17549435e-38F
// AARCH64-NETBSD:#define __FLT_RADIX__ 2
// AARCH64-NETBSD:#define __INT16_C_SUFFIX__
// AARCH64-NETBSD:#define __INT16_FMTd__ "hd"
// AARCH64-NETBSD:#define __INT16_FMTi__ "hi"
// AARCH64-NETBSD:#define __INT16_MAX__ 32767
// AARCH64-NETBSD:#define __INT16_TYPE__ short
// AARCH64-NETBSD:#define __INT32_C_SUFFIX__
// AARCH64-NETBSD:#define __INT32_FMTd__ "d"
// AARCH64-NETBSD:#define __INT32_FMTi__ "i"
// AARCH64-NETBSD:#define __INT32_MAX__ 2147483647
// AARCH64-NETBSD:#define __INT32_TYPE__ int
// AARCH64-NETBSD:#define __INT64_C_SUFFIX__ LL
// AARCH64-NETBSD:#define __INT64_FMTd__ "lld"
// AARCH64-NETBSD:#define __INT64_FMTi__ "lli"
// AARCH64-NETBSD:#define __INT64_MAX__ 9223372036854775807LL
// AARCH64-NETBSD:#define __INT64_TYPE__ long long int
// AARCH64-NETBSD:#define __INT8_C_SUFFIX__
// AARCH64-NETBSD:#define __INT8_FMTd__ "hhd"
// AARCH64-NETBSD:#define __INT8_FMTi__ "hhi"
// AARCH64-NETBSD:#define __INT8_MAX__ 127
// AARCH64-NETBSD:#define __INT8_TYPE__ signed char
// AARCH64-NETBSD:#define __INTMAX_C_SUFFIX__ LL
// AARCH64-NETBSD:#define __INTMAX_FMTd__ "lld"
// AARCH64-NETBSD:#define __INTMAX_FMTi__ "lli"
// AARCH64-NETBSD:#define __INTMAX_MAX__ 9223372036854775807LL
// AARCH64-NETBSD:#define __INTMAX_TYPE__ long long int
// AARCH64-NETBSD:#define __INTMAX_WIDTH__ 64
// AARCH64-NETBSD:#define __INTPTR_FMTd__ "ld"
// AARCH64-NETBSD:#define __INTPTR_FMTi__ "li"
// AARCH64-NETBSD:#define __INTPTR_MAX__ 9223372036854775807L
// AARCH64-NETBSD:#define __INTPTR_TYPE__ long int
// AARCH64-NETBSD:#define __INTPTR_WIDTH__ 64
// AARCH64-NETBSD:#define __INT_FAST16_FMTd__ "hd"
// AARCH64-NETBSD:#define __INT_FAST16_FMTi__ "hi"
// AARCH64-NETBSD:#define __INT_FAST16_MAX__ 32767
// AARCH64-NETBSD:#define __INT_FAST16_TYPE__ short
// AARCH64-NETBSD:#define __INT_FAST32_FMTd__ "d"
// AARCH64-NETBSD:#define __INT_FAST32_FMTi__ "i"
// AARCH64-NETBSD:#define __INT_FAST32_MAX__ 2147483647
// AARCH64-NETBSD:#define __INT_FAST32_TYPE__ int
// AARCH64-NETBSD:#define __INT_FAST64_FMTd__ "ld"
// AARCH64-NETBSD:#define __INT_FAST64_FMTi__ "li"
// AARCH64-NETBSD:#define __INT_FAST64_MAX__ 9223372036854775807L
// AARCH64-NETBSD:#define __INT_FAST64_TYPE__ long int
// AARCH64-NETBSD:#define __INT_FAST8_FMTd__ "hhd"
// AARCH64-NETBSD:#define __INT_FAST8_FMTi__ "hhi"
// AARCH64-NETBSD:#define __INT_FAST8_MAX__ 127
// AARCH64-NETBSD:#define __INT_FAST8_TYPE__ signed char
// AARCH64-NETBSD:#define __INT_LEAST16_FMTd__ "hd"
// AARCH64-NETBSD:#define __INT_LEAST16_FMTi__ "hi"
// AARCH64-NETBSD:#define __INT_LEAST16_MAX__ 32767
// AARCH64-NETBSD:#define __INT_LEAST16_TYPE__ short
// AARCH64-NETBSD:#define __INT_LEAST32_FMTd__ "d"
// AARCH64-NETBSD:#define __INT_LEAST32_FMTi__ "i"
// AARCH64-NETBSD:#define __INT_LEAST32_MAX__ 2147483647
// AARCH64-NETBSD:#define __INT_LEAST32_TYPE__ int
// AARCH64-NETBSD:#define __INT_LEAST64_FMTd__ "ld"
// AARCH64-NETBSD:#define __INT_LEAST64_FMTi__ "li"
// AARCH64-NETBSD:#define __INT_LEAST64_MAX__ 9223372036854775807L
// AARCH64-NETBSD:#define __INT_LEAST64_TYPE__ long int
// AARCH64-NETBSD:#define __INT_LEAST8_FMTd__ "hhd"
// AARCH64-NETBSD:#define __INT_LEAST8_FMTi__ "hhi"
// AARCH64-NETBSD:#define __INT_LEAST8_MAX__ 127
// AARCH64-NETBSD:#define __INT_LEAST8_TYPE__ signed char
// AARCH64-NETBSD:#define __INT_MAX__ 2147483647
// AARCH64-NETBSD:#define __LDBL_DENORM_MIN__ 6.47517511943802511092443895822764655e-4966L
// AARCH64-NETBSD:#define __LDBL_DIG__ 33
// AARCH64-NETBSD:#define __LDBL_EPSILON__ 1.92592994438723585305597794258492732e-34L
// AARCH64-NETBSD:#define __LDBL_HAS_DENORM__ 1
// AARCH64-NETBSD:#define __LDBL_HAS_INFINITY__ 1
// AARCH64-NETBSD:#define __LDBL_HAS_QUIET_NAN__ 1
// AARCH64-NETBSD:#define __LDBL_MANT_DIG__ 113
// AARCH64-NETBSD:#define __LDBL_MAX_10_EXP__ 4932
// AARCH64-NETBSD:#define __LDBL_MAX_EXP__ 16384
// AARCH64-NETBSD:#define __LDBL_MAX__ 1.18973149535723176508575932662800702e+4932L
// AARCH64-NETBSD:#define __LDBL_MIN_10_EXP__ (-4931)
// AARCH64-NETBSD:#define __LDBL_MIN_EXP__ (-16381)
// AARCH64-NETBSD:#define __LDBL_MIN__ 3.36210314311209350626267781732175260e-4932L
// AARCH64-NETBSD:#define __LITTLE_ENDIAN__ 1
// AARCH64-NETBSD:#define __LONG_LONG_MAX__ 9223372036854775807LL
// AARCH64-NETBSD:#define __LONG_MAX__ 9223372036854775807L
// AARCH64-NETBSD:#define __LP64__ 1
// AARCH64-NETBSD:#define __NetBSD__ 1
// AARCH64-NETBSD:#define __POINTER_WIDTH__ 64
// AARCH64-NETBSD:#define __PTRDIFF_TYPE__ long int
// AARCH64-NETBSD:#define __PTRDIFF_WIDTH__ 64
// AARCH64-NETBSD:#define __SCHAR_MAX__ 127
// AARCH64-NETBSD:#define __SHRT_MAX__ 32767
// AARCH64-NETBSD:#define __SIG_ATOMIC_MAX__ 2147483647
// AARCH64-NETBSD:#define __SIG_ATOMIC_WIDTH__ 32
// AARCH64-NETBSD:#define __SIZEOF_DOUBLE__ 8
// AARCH64-NETBSD:#define __SIZEOF_FLOAT__ 4
// AARCH64-NETBSD:#define __SIZEOF_INT__ 4
// AARCH64-NETBSD:#define __SIZEOF_LONG_DOUBLE__ 16
// AARCH64-NETBSD:#define __SIZEOF_LONG_LONG__ 8
// AARCH64-NETBSD:#define __SIZEOF_LONG__ 8
// AARCH64-NETBSD:#define __SIZEOF_POINTER__ 8
// AARCH64-NETBSD:#define __SIZEOF_PTRDIFF_T__ 8
// AARCH64-NETBSD:#define __SIZEOF_SHORT__ 2
// AARCH64-NETBSD:#define __SIZEOF_SIZE_T__ 8
// AARCH64-NETBSD:#define __SIZEOF_WCHAR_T__ 4
// AARCH64-NETBSD:#define __SIZEOF_WINT_T__ 4
// AARCH64-NETBSD:#define __SIZE_MAX__ 18446744073709551615UL
// AARCH64-NETBSD:#define __SIZE_TYPE__ long unsigned int
// AARCH64-NETBSD:#define __SIZE_WIDTH__ 64
// AARCH64-NETBSD:#define __UINT16_C_SUFFIX__
// AARCH64-NETBSD:#define __UINT16_MAX__ 65535
// AARCH64-NETBSD:#define __UINT16_TYPE__ unsigned short
// AARCH64-NETBSD:#define __UINT32_C_SUFFIX__ U
// AARCH64-NETBSD:#define __UINT32_MAX__ 4294967295U
// AARCH64-NETBSD:#define __UINT32_TYPE__ unsigned int
// AARCH64-NETBSD:#define __UINT64_C_SUFFIX__ ULL
// AARCH64-NETBSD:#define __UINT64_MAX__ 18446744073709551615ULL
// AARCH64-NETBSD:#define __UINT64_TYPE__ long long unsigned int
// AARCH64-NETBSD:#define __UINT8_C_SUFFIX__
// AARCH64-NETBSD:#define __UINT8_MAX__ 255
// AARCH64-NETBSD:#define __UINT8_TYPE__ unsigned char
// AARCH64-NETBSD:#define __UINTMAX_C_SUFFIX__ ULL
// AARCH64-NETBSD:#define __UINTMAX_MAX__ 18446744073709551615ULL
// AARCH64-NETBSD:#define __UINTMAX_TYPE__ long long unsigned int
// AARCH64-NETBSD:#define __UINTMAX_WIDTH__ 64
// AARCH64-NETBSD:#define __UINTPTR_MAX__ 18446744073709551615UL
// AARCH64-NETBSD:#define __UINTPTR_TYPE__ long unsigned int
// AARCH64-NETBSD:#define __UINTPTR_WIDTH__ 64
// AARCH64-NETBSD:#define __UINT_FAST16_MAX__ 65535
// AARCH64-NETBSD:#define __UINT_FAST16_TYPE__ unsigned short
// AARCH64-NETBSD:#define __UINT_FAST32_MAX__ 4294967295U
// AARCH64-NETBSD:#define __UINT_FAST32_TYPE__ unsigned int
// AARCH64-NETBSD:#define __UINT_FAST64_MAX__ 18446744073709551615UL
// AARCH64-NETBSD:#define __UINT_FAST64_TYPE__ long unsigned int
// AARCH64-NETBSD:#define __UINT_FAST8_MAX__ 255
// AARCH64-NETBSD:#define __UINT_FAST8_TYPE__ unsigned char
// AARCH64-NETBSD:#define __UINT_LEAST16_MAX__ 65535
// AARCH64-NETBSD:#define __UINT_LEAST16_TYPE__ unsigned short
// AARCH64-NETBSD:#define __UINT_LEAST32_MAX__ 4294967295U
// AARCH64-NETBSD:#define __UINT_LEAST32_TYPE__ unsigned int
// AARCH64-NETBSD:#define __UINT_LEAST64_MAX__ 18446744073709551615UL
// AARCH64-NETBSD:#define __UINT_LEAST64_TYPE__ long unsigned int
// AARCH64-NETBSD:#define __UINT_LEAST8_MAX__ 255
// AARCH64-NETBSD:#define __UINT_LEAST8_TYPE__ unsigned char
// AARCH64-NETBSD:#define __USER_LABEL_PREFIX__
// AARCH64-NETBSD:#define __WCHAR_MAX__ 2147483647
// AARCH64-NETBSD:#define __WCHAR_TYPE__ int
// AARCH64-NETBSD:#define __WCHAR_WIDTH__ 32
// AARCH64-NETBSD:#define __WINT_TYPE__ int
// AARCH64-NETBSD:#define __WINT_WIDTH__ 32
// AARCH64-NETBSD:#define __aarch64__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=aarch64-openbsd < /dev/null | FileCheck -match-full-lines -check-prefix AARCH64-OPENBSD %s
//
// AARCH64-OPENBSD:#define _LP64 1
// AARCH64-OPENBSD-NOT:#define __AARCH64EB__ 1
// AARCH64-OPENBSD:#define __AARCH64EL__ 1
// AARCH64-OPENBSD-NOT:#define __AARCH_BIG_ENDIAN 1
// AARCH64-OPENBSD:#define __ARM_64BIT_STATE 1
// AARCH64-OPENBSD:#define __ARM_ARCH 8
// AARCH64-OPENBSD:#define __ARM_ARCH_ISA_A64 1
// AARCH64-OPENBSD-NOT:#define __ARM_BIG_ENDIAN 1
// AARCH64-OPENBSD:#define __BIGGEST_ALIGNMENT__ 16
// AARCH64-OPENBSD:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// AARCH64-OPENBSD:#define __CHAR16_TYPE__ unsigned short
// AARCH64-OPENBSD:#define __CHAR32_TYPE__ unsigned int
// AARCH64-OPENBSD:#define __CHAR_BIT__ 8
// AARCH64-OPENBSD:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// AARCH64-OPENBSD:#define __DBL_DIG__ 15
// AARCH64-OPENBSD:#define __DBL_EPSILON__ 2.2204460492503131e-16
// AARCH64-OPENBSD:#define __DBL_HAS_DENORM__ 1
// AARCH64-OPENBSD:#define __DBL_HAS_INFINITY__ 1
// AARCH64-OPENBSD:#define __DBL_HAS_QUIET_NAN__ 1
// AARCH64-OPENBSD:#define __DBL_MANT_DIG__ 53
// AARCH64-OPENBSD:#define __DBL_MAX_10_EXP__ 308
// AARCH64-OPENBSD:#define __DBL_MAX_EXP__ 1024
// AARCH64-OPENBSD:#define __DBL_MAX__ 1.7976931348623157e+308
// AARCH64-OPENBSD:#define __DBL_MIN_10_EXP__ (-307)
// AARCH64-OPENBSD:#define __DBL_MIN_EXP__ (-1021)
// AARCH64-OPENBSD:#define __DBL_MIN__ 2.2250738585072014e-308
// AARCH64-OPENBSD:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// AARCH64-OPENBSD:#define __ELF__ 1
// AARCH64-OPENBSD:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// AARCH64-OPENBSD:#define __FLT_DIG__ 6
// AARCH64-OPENBSD:#define __FLT_EPSILON__ 1.19209290e-7F
// AARCH64-OPENBSD:#define __FLT_EVAL_METHOD__ 0
// AARCH64-OPENBSD:#define __FLT_HAS_DENORM__ 1
// AARCH64-OPENBSD:#define __FLT_HAS_INFINITY__ 1
// AARCH64-OPENBSD:#define __FLT_HAS_QUIET_NAN__ 1
// AARCH64-OPENBSD:#define __FLT_MANT_DIG__ 24
// AARCH64-OPENBSD:#define __FLT_MAX_10_EXP__ 38
// AARCH64-OPENBSD:#define __FLT_MAX_EXP__ 128
// AARCH64-OPENBSD:#define __FLT_MAX__ 3.40282347e+38F
// AARCH64-OPENBSD:#define __FLT_MIN_10_EXP__ (-37)
// AARCH64-OPENBSD:#define __FLT_MIN_EXP__ (-125)
// AARCH64-OPENBSD:#define __FLT_MIN__ 1.17549435e-38F
// AARCH64-OPENBSD:#define __FLT_RADIX__ 2
// AARCH64-OPENBSD:#define __INT16_C_SUFFIX__
// AARCH64-OPENBSD:#define __INT16_FMTd__ "hd"
// AARCH64-OPENBSD:#define __INT16_FMTi__ "hi"
// AARCH64-OPENBSD:#define __INT16_MAX__ 32767
// AARCH64-OPENBSD:#define __INT16_TYPE__ short
// AARCH64-OPENBSD:#define __INT32_C_SUFFIX__
// AARCH64-OPENBSD:#define __INT32_FMTd__ "d"
// AARCH64-OPENBSD:#define __INT32_FMTi__ "i"
// AARCH64-OPENBSD:#define __INT32_MAX__ 2147483647
// AARCH64-OPENBSD:#define __INT32_TYPE__ int
// AARCH64-OPENBSD:#define __INT64_C_SUFFIX__ LL
// AARCH64-OPENBSD:#define __INT64_FMTd__ "lld"
// AARCH64-OPENBSD:#define __INT64_FMTi__ "lli"
// AARCH64-OPENBSD:#define __INT64_MAX__ 9223372036854775807LL
// AARCH64-OPENBSD:#define __INT64_TYPE__ long long int
// AARCH64-OPENBSD:#define __INT8_C_SUFFIX__
// AARCH64-OPENBSD:#define __INT8_FMTd__ "hhd"
// AARCH64-OPENBSD:#define __INT8_FMTi__ "hhi"
// AARCH64-OPENBSD:#define __INT8_MAX__ 127
// AARCH64-OPENBSD:#define __INT8_TYPE__ signed char
// AARCH64-OPENBSD:#define __INTMAX_C_SUFFIX__ LL
// AARCH64-OPENBSD:#define __INTMAX_FMTd__ "lld"
// AARCH64-OPENBSD:#define __INTMAX_FMTi__ "lli"
// AARCH64-OPENBSD:#define __INTMAX_MAX__ 9223372036854775807LL
// AARCH64-OPENBSD:#define __INTMAX_TYPE__ long long int
// AARCH64-OPENBSD:#define __INTMAX_WIDTH__ 64
// AARCH64-OPENBSD:#define __INTPTR_FMTd__ "ld"
// AARCH64-OPENBSD:#define __INTPTR_FMTi__ "li"
// AARCH64-OPENBSD:#define __INTPTR_MAX__ 9223372036854775807L
// AARCH64-OPENBSD:#define __INTPTR_TYPE__ long int
// AARCH64-OPENBSD:#define __INTPTR_WIDTH__ 64
// AARCH64-OPENBSD:#define __INT_FAST16_FMTd__ "hd"
// AARCH64-OPENBSD:#define __INT_FAST16_FMTi__ "hi"
// AARCH64-OPENBSD:#define __INT_FAST16_MAX__ 32767
// AARCH64-OPENBSD:#define __INT_FAST16_TYPE__ short
// AARCH64-OPENBSD:#define __INT_FAST32_FMTd__ "d"
// AARCH64-OPENBSD:#define __INT_FAST32_FMTi__ "i"
// AARCH64-OPENBSD:#define __INT_FAST32_MAX__ 2147483647
// AARCH64-OPENBSD:#define __INT_FAST32_TYPE__ int
// AARCH64-OPENBSD:#define __INT_FAST64_FMTd__ "ld"
// AARCH64-OPENBSD:#define __INT_FAST64_FMTi__ "li"
// AARCH64-OPENBSD:#define __INT_FAST64_MAX__ 9223372036854775807L
// AARCH64-OPENBSD:#define __INT_FAST64_TYPE__ long int
// AARCH64-OPENBSD:#define __INT_FAST8_FMTd__ "hhd"
// AARCH64-OPENBSD:#define __INT_FAST8_FMTi__ "hhi"
// AARCH64-OPENBSD:#define __INT_FAST8_MAX__ 127
// AARCH64-OPENBSD:#define __INT_FAST8_TYPE__ signed char
// AARCH64-OPENBSD:#define __INT_LEAST16_FMTd__ "hd"
// AARCH64-OPENBSD:#define __INT_LEAST16_FMTi__ "hi"
// AARCH64-OPENBSD:#define __INT_LEAST16_MAX__ 32767
// AARCH64-OPENBSD:#define __INT_LEAST16_TYPE__ short
// AARCH64-OPENBSD:#define __INT_LEAST32_FMTd__ "d"
// AARCH64-OPENBSD:#define __INT_LEAST32_FMTi__ "i"
// AARCH64-OPENBSD:#define __INT_LEAST32_MAX__ 2147483647
// AARCH64-OPENSD:#define __INT_LEAST32_TYPE__ int
// AARCH64-OPENBSD:#define __INT_LEAST64_FMTd__ "ld"
// AARCH64-OPENBSD:#define __INT_LEAST64_FMTi__ "li"
// AARCH64-OPENBSD:#define __INT_LEAST64_MAX__ 9223372036854775807L
// AARCH64-OPENBSD:#define __INT_LEAST64_TYPE__ long int
// AARCH64-OPENBSD:#define __INT_LEAST8_FMTd__ "hhd"
// AARCH64-OPENBSD:#define __INT_LEAST8_FMTi__ "hhi"
// AARCH64-OPENBSD:#define __INT_LEAST8_MAX__ 127
// AARCH64-OPENBSD:#define __INT_LEAST8_TYPE__ signed char
// AARCH64-OPENBSD:#define __INT_MAX__ 2147483647
// AARCH64-OPENBSD:#define __LDBL_DENORM_MIN__ 6.47517511943802511092443895822764655e-4966L
// AARCH64-OPENBSD:#define __LDBL_DIG__ 33
// AARCH64-OPENBSD:#define __LDBL_EPSILON__ 1.92592994438723585305597794258492732e-34L
// AARCH64-OPENBSD:#define __LDBL_HAS_DENORM__ 1
// AARCH64-OPENBSD:#define __LDBL_HAS_INFINITY__ 1
// AARCH64-OPENBSD:#define __LDBL_HAS_QUIET_NAN__ 1
// AARCH64-OPENBSD:#define __LDBL_MANT_DIG__ 113
// AARCH64-OPENBSD:#define __LDBL_MAX_10_EXP__ 4932
// AARCH64-OPENBSD:#define __LDBL_MAX_EXP__ 16384
// AARCH64-OPENBSD:#define __LDBL_MAX__ 1.18973149535723176508575932662800702e+4932L
// AARCH64-OPENBSD:#define __LDBL_MIN_10_EXP__ (-4931)
// AARCH64-OPENBSD:#define __LDBL_MIN_EXP__ (-16381)
// AARCH64-OPENBSD:#define __LDBL_MIN__ 3.36210314311209350626267781732175260e-4932L
// AARCH64-OPENBSD:#define __LITTLE_ENDIAN__ 1
// AARCH64-OPENBSD:#define __LONG_LONG_MAX__ 9223372036854775807LL
// AARCH64-OPENBSD:#define __LONG_MAX__ 9223372036854775807L
// AARCH64-OPENBSD:#define __LP64__ 1
// AARCH64-OPENBSD:#define __OpenBSD__ 1
// AARCH64-OPENBSD:#define __POINTER_WIDTH__ 64
// AARCH64-OPENBSD:#define __PTRDIFF_TYPE__ long int
// AARCH64-OPENBSD:#define __PTRDIFF_WIDTH__ 64
// AARCH64-OPENBSD:#define __SCHAR_MAX__ 127
// AARCH64-OPENBSD:#define __SHRT_MAX__ 32767
// AARCH64-OPENBSD:#define __SIG_ATOMIC_MAX__ 2147483647
// AARCH64-OPENBSD:#define __SIG_ATOMIC_WIDTH__ 32
// AARCH64-OPENBSD:#define __SIZEOF_DOUBLE__ 8
// AARCH64-OPENBSD:#define __SIZEOF_FLOAT__ 4
// AARCH64-OPENBSD:#define __SIZEOF_INT__ 4
// AARCH64-OPENBSD:#define __SIZEOF_LONG_DOUBLE__ 16
// AARCH64-OPENBSD:#define __SIZEOF_LONG_LONG__ 8
// AARCH64-OPENBSD:#define __SIZEOF_LONG__ 8
// AARCH64-OPENBSD:#define __SIZEOF_POINTER__ 8
// AARCH64-OPENBSD:#define __SIZEOF_PTRDIFF_T__ 8
// AARCH64-OPENBSD:#define __SIZEOF_SHORT__ 2
// AARCH64-OPENBSD:#define __SIZEOF_SIZE_T__ 8
// AARCH64-OPENBSD:#define __SIZEOF_WCHAR_T__ 4
// AARCH64-OPENBSD:#define __SIZEOF_WINT_T__ 4
// AARCH64-OPENBSD:#define __SIZE_MAX__ 18446744073709551615UL
// AARCH64-OPENBSD:#define __SIZE_TYPE__ long unsigned int
// AARCH64-OPENBSD:#define __SIZE_WIDTH__ 64
// AARCH64-OPENBSD:#define __UINT16_C_SUFFIX__
// AARCH64-OPENBSD:#define __UINT16_MAX__ 65535
// AARCH64-OPENBSD:#define __UINT16_TYPE__ unsigned short
// AARCH64-OPENBSD:#define __UINT32_C_SUFFIX__ U
// AARCH64-OPENBSD:#define __UINT32_MAX__ 4294967295U
// AARCH64-OPENBSD:#define __UINT32_TYPE__ unsigned int
// AARCH64-OPENBSD:#define __UINT64_C_SUFFIX__ ULL
// AARCH64-OPENBSD:#define __UINT64_MAX__ 18446744073709551615ULL
// AARCH64-OPENBSD:#define __UINT64_TYPE__ long long unsigned int
// AARCH64-OPENBSD:#define __UINT8_C_SUFFIX__
// AARCH64-OPENBSD:#define __UINT8_MAX__ 255
// AARCH64-OPENBSD:#define __UINT8_TYPE__ unsigned char
// AARCH64-OPENBSD:#define __UINTMAX_C_SUFFIX__ ULL
// AARCH64-OPENBSD:#define __UINTMAX_MAX__ 18446744073709551615ULL
// AARCH64-OPENBSD:#define __UINTMAX_TYPE__ long long unsigned int
// AARCH64-OPENBSD:#define __UINTMAX_WIDTH__ 64
// AARCH64-OPENBSD:#define __UINTPTR_MAX__ 18446744073709551615UL
// AARCH64-OPENBSD:#define __UINTPTR_TYPE__ long unsigned int
// AARCH64-OPENBSD:#define __UINTPTR_WIDTH__ 64
// AARCH64-OPENBSD:#define __UINT_FAST16_MAX__ 65535
// AARCH64-OPENBSD:#define __UINT_FAST16_TYPE__ unsigned short
// AARCH64-OPENBSD:#define __UINT_FAST32_MAX__ 4294967295U
// AARCH64-OPENBSD:#define __UINT_FAST32_TYPE__ unsigned int
// AARCH64-OPENBSD:#define __UINT_FAST64_MAX__ 18446744073709551615UL
// AARCH64-OPENBSD:#define __UINT_FAST64_TYPE__ long unsigned int
// AARCH64-OPENBSD:#define __UINT_FAST8_MAX__ 255
// AARCH64-OPENBSD:#define __UINT_FAST8_TYPE__ unsigned char
// AARCH64-OPENBSD:#define __UINT_LEAST16_MAX__ 65535
// AARCH64-OPENBSD:#define __UINT_LEAST16_TYPE__ unsigned short
// AARCH64-OPENBSD:#define __UINT_LEAST32_MAX__ 4294967295U
// AARCH64-OPENBSD:#define __UINT_LEAST32_TYPE__ unsigned int
// AARCH64-OPENBSD:#define __UINT_LEAST64_MAX__ 18446744073709551615UL
// AARCH64-OPENBSD:#define __UINT_LEAST64_TYPE__ long unsigned int
// AARCH64-OPENBSD:#define __UINT_LEAST8_MAX__ 255
// AARCH64-OPENBSD:#define __UINT_LEAST8_TYPE__ unsigned char
// AARCH64-OPENBSD:#define __USER_LABEL_PREFIX__
// AARCH64-OPENBSD:#define __WCHAR_MAX__ 2147483647
// AARCH64-OPENBSD:#define __WCHAR_TYPE__ int
// AARCH64-OPENBSD:#define __WCHAR_WIDTH__ 32
// AARCH64-OPENBSD:#define __WINT_TYPE__ int
// AARCH64-OPENBSD:#define __WINT_WIDTH__ 32
// AARCH64-OPENBSD:#define __aarch64__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=aarch64-freebsd11 < /dev/null | FileCheck -match-full-lines -check-prefix AARCH64-FREEBSD %s
//
// AARCH64-FREEBSD:#define _LP64 1
// AARCH64-FREEBSD-NOT:#define __AARCH64EB__ 1
// AARCH64-FREEBSD:#define __AARCH64EL__ 1
// AARCH64-FREEBSD-NOT:#define __AARCH_BIG_ENDIAN 1
// AARCH64-FREEBSD:#define __ARM_64BIT_STATE 1
// AARCH64-FREEBSD:#define __ARM_ARCH 8
// AARCH64-FREEBSD:#define __ARM_ARCH_ISA_A64 1
// AARCH64-FREEBSD-NOT:#define __ARM_BIG_ENDIAN 1
// AARCH64-FREEBSD:#define __BIGGEST_ALIGNMENT__ 16
// AARCH64-FREEBSD:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// AARCH64-FREEBSD:#define __CHAR16_TYPE__ unsigned short
// AARCH64-FREEBSD:#define __CHAR32_TYPE__ unsigned int
// AARCH64-FREEBSD:#define __CHAR_BIT__ 8
// AARCH64-FREEBSD:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// AARCH64-FREEBSD:#define __DBL_DIG__ 15
// AARCH64-FREEBSD:#define __DBL_EPSILON__ 2.2204460492503131e-16
// AARCH64-FREEBSD:#define __DBL_HAS_DENORM__ 1
// AARCH64-FREEBSD:#define __DBL_HAS_INFINITY__ 1
// AARCH64-FREEBSD:#define __DBL_HAS_QUIET_NAN__ 1
// AARCH64-FREEBSD:#define __DBL_MANT_DIG__ 53
// AARCH64-FREEBSD:#define __DBL_MAX_10_EXP__ 308
// AARCH64-FREEBSD:#define __DBL_MAX_EXP__ 1024
// AARCH64-FREEBSD:#define __DBL_MAX__ 1.7976931348623157e+308
// AARCH64-FREEBSD:#define __DBL_MIN_10_EXP__ (-307)
// AARCH64-FREEBSD:#define __DBL_MIN_EXP__ (-1021)
// AARCH64-FREEBSD:#define __DBL_MIN__ 2.2250738585072014e-308
// AARCH64-FREEBSD:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// AARCH64-FREEBSD:#define __ELF__ 1
// AARCH64-FREEBSD:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// AARCH64-FREEBSD:#define __FLT_DIG__ 6
// AARCH64-FREEBSD:#define __FLT_EPSILON__ 1.19209290e-7F
// AARCH64-FREEBSD:#define __FLT_EVAL_METHOD__ 0
// AARCH64-FREEBSD:#define __FLT_HAS_DENORM__ 1
// AARCH64-FREEBSD:#define __FLT_HAS_INFINITY__ 1
// AARCH64-FREEBSD:#define __FLT_HAS_QUIET_NAN__ 1
// AARCH64-FREEBSD:#define __FLT_MANT_DIG__ 24
// AARCH64-FREEBSD:#define __FLT_MAX_10_EXP__ 38
// AARCH64-FREEBSD:#define __FLT_MAX_EXP__ 128
// AARCH64-FREEBSD:#define __FLT_MAX__ 3.40282347e+38F
// AARCH64-FREEBSD:#define __FLT_MIN_10_EXP__ (-37)
// AARCH64-FREEBSD:#define __FLT_MIN_EXP__ (-125)
// AARCH64-FREEBSD:#define __FLT_MIN__ 1.17549435e-38F
// AARCH64-FREEBSD:#define __FLT_RADIX__ 2
// AARCH64-FREEBSD:#define __FreeBSD__ 11
// AARCH64-FREEBSD:#define __INT16_C_SUFFIX__
// AARCH64-FREEBSD:#define __INT16_FMTd__ "hd"
// AARCH64-FREEBSD:#define __INT16_FMTi__ "hi"
// AARCH64-FREEBSD:#define __INT16_MAX__ 32767
// AARCH64-FREEBSD:#define __INT16_TYPE__ short
// AARCH64-FREEBSD:#define __INT32_C_SUFFIX__
// AARCH64-FREEBSD:#define __INT32_FMTd__ "d"
// AARCH64-FREEBSD:#define __INT32_FMTi__ "i"
// AARCH64-FREEBSD:#define __INT32_MAX__ 2147483647
// AARCH64-FREEBSD:#define __INT32_TYPE__ int
// AARCH64-FREEBSD:#define __INT64_C_SUFFIX__ L
// AARCH64-FREEBSD:#define __INT64_FMTd__ "ld"
// AARCH64-FREEBSD:#define __INT64_FMTi__ "li"
// AARCH64-FREEBSD:#define __INT64_MAX__ 9223372036854775807L
// AARCH64-FREEBSD:#define __INT64_TYPE__ long int
// AARCH64-FREEBSD:#define __INT8_C_SUFFIX__
// AARCH64-FREEBSD:#define __INT8_FMTd__ "hhd"
// AARCH64-FREEBSD:#define __INT8_FMTi__ "hhi"
// AARCH64-FREEBSD:#define __INT8_MAX__ 127
// AARCH64-FREEBSD:#define __INT8_TYPE__ signed char
// AARCH64-FREEBSD:#define __INTMAX_C_SUFFIX__ L
// AARCH64-FREEBSD:#define __INTMAX_FMTd__ "ld"
// AARCH64-FREEBSD:#define __INTMAX_FMTi__ "li"
// AARCH64-FREEBSD:#define __INTMAX_MAX__ 9223372036854775807L
// AARCH64-FREEBSD:#define __INTMAX_TYPE__ long int
// AARCH64-FREEBSD:#define __INTMAX_WIDTH__ 64
// AARCH64-FREEBSD:#define __INTPTR_FMTd__ "ld"
// AARCH64-FREEBSD:#define __INTPTR_FMTi__ "li"
// AARCH64-FREEBSD:#define __INTPTR_MAX__ 9223372036854775807L
// AARCH64-FREEBSD:#define __INTPTR_TYPE__ long int
// AARCH64-FREEBSD:#define __INTPTR_WIDTH__ 64
// AARCH64-FREEBSD:#define __INT_FAST16_FMTd__ "hd"
// AARCH64-FREEBSD:#define __INT_FAST16_FMTi__ "hi"
// AARCH64-FREEBSD:#define __INT_FAST16_MAX__ 32767
// AARCH64-FREEBSD:#define __INT_FAST16_TYPE__ short
// AARCH64-FREEBSD:#define __INT_FAST32_FMTd__ "d"
// AARCH64-FREEBSD:#define __INT_FAST32_FMTi__ "i"
// AARCH64-FREEBSD:#define __INT_FAST32_MAX__ 2147483647
// AARCH64-FREEBSD:#define __INT_FAST32_TYPE__ int
// AARCH64-FREEBSD:#define __INT_FAST64_FMTd__ "ld"
// AARCH64-FREEBSD:#define __INT_FAST64_FMTi__ "li"
// AARCH64-FREEBSD:#define __INT_FAST64_MAX__ 9223372036854775807L
// AARCH64-FREEBSD:#define __INT_FAST64_TYPE__ long int
// AARCH64-FREEBSD:#define __INT_FAST8_FMTd__ "hhd"
// AARCH64-FREEBSD:#define __INT_FAST8_FMTi__ "hhi"
// AARCH64-FREEBSD:#define __INT_FAST8_MAX__ 127
// AARCH64-FREEBSD:#define __INT_FAST8_TYPE__ signed char
// AARCH64-FREEBSD:#define __INT_LEAST16_FMTd__ "hd"
// AARCH64-FREEBSD:#define __INT_LEAST16_FMTi__ "hi"
// AARCH64-FREEBSD:#define __INT_LEAST16_MAX__ 32767
// AARCH64-FREEBSD:#define __INT_LEAST16_TYPE__ short
// AARCH64-FREEBSD:#define __INT_LEAST32_FMTd__ "d"
// AARCH64-FREEBSD:#define __INT_LEAST32_FMTi__ "i"
// AARCH64-FREEBSD:#define __INT_LEAST32_MAX__ 2147483647
// AARCH64-FREEBSD:#define __INT_LEAST32_TYPE__ int
// AARCH64-FREEBSD:#define __INT_LEAST64_FMTd__ "ld"
// AARCH64-FREEBSD:#define __INT_LEAST64_FMTi__ "li"
// AARCH64-FREEBSD:#define __INT_LEAST64_MAX__ 9223372036854775807L
// AARCH64-FREEBSD:#define __INT_LEAST64_TYPE__ long int
// AARCH64-FREEBSD:#define __INT_LEAST8_FMTd__ "hhd"
// AARCH64-FREEBSD:#define __INT_LEAST8_FMTi__ "hhi"
// AARCH64-FREEBSD:#define __INT_LEAST8_MAX__ 127
// AARCH64-FREEBSD:#define __INT_LEAST8_TYPE__ signed char
// AARCH64-FREEBSD:#define __INT_MAX__ 2147483647
// AARCH64-FREEBSD:#define __LDBL_DENORM_MIN__ 6.47517511943802511092443895822764655e-4966L
// AARCH64-FREEBSD:#define __LDBL_DIG__ 33
// AARCH64-FREEBSD:#define __LDBL_EPSILON__ 1.92592994438723585305597794258492732e-34L
// AARCH64-FREEBSD:#define __LDBL_HAS_DENORM__ 1
// AARCH64-FREEBSD:#define __LDBL_HAS_INFINITY__ 1
// AARCH64-FREEBSD:#define __LDBL_HAS_QUIET_NAN__ 1
// AARCH64-FREEBSD:#define __LDBL_MANT_DIG__ 113
// AARCH64-FREEBSD:#define __LDBL_MAX_10_EXP__ 4932
// AARCH64-FREEBSD:#define __LDBL_MAX_EXP__ 16384
// AARCH64-FREEBSD:#define __LDBL_MAX__ 1.18973149535723176508575932662800702e+4932L
// AARCH64-FREEBSD:#define __LDBL_MIN_10_EXP__ (-4931)
// AARCH64-FREEBSD:#define __LDBL_MIN_EXP__ (-16381)
// AARCH64-FREEBSD:#define __LDBL_MIN__ 3.36210314311209350626267781732175260e-4932L
// AARCH64-FREEBSD:#define __LITTLE_ENDIAN__ 1
// AARCH64-FREEBSD:#define __LONG_LONG_MAX__ 9223372036854775807LL
// AARCH64-FREEBSD:#define __LONG_MAX__ 9223372036854775807L
// AARCH64-FREEBSD:#define __LP64__ 1
// AARCH64-FREEBSD:#define __POINTER_WIDTH__ 64
// AARCH64-FREEBSD:#define __PTRDIFF_TYPE__ long int
// AARCH64-FREEBSD:#define __PTRDIFF_WIDTH__ 64
// AARCH64-FREEBSD:#define __SCHAR_MAX__ 127
// AARCH64-FREEBSD:#define __SHRT_MAX__ 32767
// AARCH64-FREEBSD:#define __SIG_ATOMIC_MAX__ 2147483647
// AARCH64-FREEBSD:#define __SIG_ATOMIC_WIDTH__ 32
// AARCH64-FREEBSD:#define __SIZEOF_DOUBLE__ 8
// AARCH64-FREEBSD:#define __SIZEOF_FLOAT__ 4
// AARCH64-FREEBSD:#define __SIZEOF_INT128__ 16
// AARCH64-FREEBSD:#define __SIZEOF_INT__ 4
// AARCH64-FREEBSD:#define __SIZEOF_LONG_DOUBLE__ 16
// AARCH64-FREEBSD:#define __SIZEOF_LONG_LONG__ 8
// AARCH64-FREEBSD:#define __SIZEOF_LONG__ 8
// AARCH64-FREEBSD:#define __SIZEOF_POINTER__ 8
// AARCH64-FREEBSD:#define __SIZEOF_PTRDIFF_T__ 8
// AARCH64-FREEBSD:#define __SIZEOF_SHORT__ 2
// AARCH64-FREEBSD:#define __SIZEOF_SIZE_T__ 8
// AARCH64-FREEBSD:#define __SIZEOF_WCHAR_T__ 4
// AARCH64-FREEBSD:#define __SIZEOF_WINT_T__ 4
// AARCH64-FREEBSD:#define __SIZE_MAX__ 18446744073709551615UL
// AARCH64-FREEBSD:#define __SIZE_TYPE__ long unsigned int
// AARCH64-FREEBSD:#define __SIZE_WIDTH__ 64
// AARCH64-FREEBSD:#define __UINT16_C_SUFFIX__
// AARCH64-FREEBSD:#define __UINT16_MAX__ 65535
// AARCH64-FREEBSD:#define __UINT16_TYPE__ unsigned short
// AARCH64-FREEBSD:#define __UINT32_C_SUFFIX__ U
// AARCH64-FREEBSD:#define __UINT32_MAX__ 4294967295U
// AARCH64-FREEBSD:#define __UINT32_TYPE__ unsigned int
// AARCH64-FREEBSD:#define __UINT64_C_SUFFIX__ UL
// AARCH64-FREEBSD:#define __UINT64_MAX__ 18446744073709551615UL
// AARCH64-FREEBSD:#define __UINT64_TYPE__ long unsigned int
// AARCH64-FREEBSD:#define __UINT8_C_SUFFIX__
// AARCH64-FREEBSD:#define __UINT8_MAX__ 255
// AARCH64-FREEBSD:#define __UINT8_TYPE__ unsigned char
// AARCH64-FREEBSD:#define __UINTMAX_C_SUFFIX__ UL
// AARCH64-FREEBSD:#define __UINTMAX_MAX__ 18446744073709551615UL
// AARCH64-FREEBSD:#define __UINTMAX_TYPE__ long unsigned int
// AARCH64-FREEBSD:#define __UINTMAX_WIDTH__ 64
// AARCH64-FREEBSD:#define __UINTPTR_MAX__ 18446744073709551615UL
// AARCH64-FREEBSD:#define __UINTPTR_TYPE__ long unsigned int
// AARCH64-FREEBSD:#define __UINTPTR_WIDTH__ 64
// AARCH64-FREEBSD:#define __UINT_FAST16_MAX__ 65535
// AARCH64-FREEBSD:#define __UINT_FAST16_TYPE__ unsigned short
// AARCH64-FREEBSD:#define __UINT_FAST32_MAX__ 4294967295U
// AARCH64-FREEBSD:#define __UINT_FAST32_TYPE__ unsigned int
// AARCH64-FREEBSD:#define __UINT_FAST64_MAX__ 18446744073709551615UL
// AARCH64-FREEBSD:#define __UINT_FAST64_TYPE__ long unsigned int
// AARCH64-FREEBSD:#define __UINT_FAST8_MAX__ 255
// AARCH64-FREEBSD:#define __UINT_FAST8_TYPE__ unsigned char
// AARCH64-FREEBSD:#define __UINT_LEAST16_MAX__ 65535
// AARCH64-FREEBSD:#define __UINT_LEAST16_TYPE__ unsigned short
// AARCH64-FREEBSD:#define __UINT_LEAST32_MAX__ 4294967295U
// AARCH64-FREEBSD:#define __UINT_LEAST32_TYPE__ unsigned int
// AARCH64-FREEBSD:#define __UINT_LEAST64_MAX__ 18446744073709551615UL
// AARCH64-FREEBSD:#define __UINT_LEAST64_TYPE__ long unsigned int
// AARCH64-FREEBSD:#define __UINT_LEAST8_MAX__ 255
// AARCH64-FREEBSD:#define __UINT_LEAST8_TYPE__ unsigned char
// AARCH64-FREEBSD:#define __USER_LABEL_PREFIX__
// AARCH64-FREEBSD:#define __WCHAR_MAX__ 4294967295U
// AARCH64-FREEBSD:#define __WCHAR_TYPE__ unsigned int
// AARCH64-FREEBSD:#define __WCHAR_UNSIGNED__ 1
// AARCH64-FREEBSD:#define __WCHAR_WIDTH__ 32
// AARCH64-FREEBSD:#define __WINT_TYPE__ int
// AARCH64-FREEBSD:#define __WINT_WIDTH__ 32
// AARCH64-FREEBSD:#define __aarch64__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=aarch64-apple-ios7.0 < /dev/null | FileCheck -match-full-lines -check-prefix AARCH64-DARWIN %s
//
// AARCH64-DARWIN: #define _LP64 1
// AARCH64-NOT: #define __AARCH64EB__ 1
// AARCH64-DARWIN: #define __AARCH64EL__ 1
// AARCH64-NOT: #define __AARCH_BIG_ENDIAN 1
// AARCH64-DARWIN: #define __ARM_64BIT_STATE 1
// AARCH64-DARWIN: #define __ARM_ARCH 8
// AARCH64-DARWIN: #define __ARM_ARCH_ISA_A64 1
// AARCH64-NOT: #define __ARM_BIG_ENDIAN 1
// AARCH64-DARWIN: #define __BIGGEST_ALIGNMENT__ 8
// AARCH64-DARWIN: #define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// AARCH64-DARWIN: #define __CHAR16_TYPE__ unsigned short
// AARCH64-DARWIN: #define __CHAR32_TYPE__ unsigned int
// AARCH64-DARWIN: #define __CHAR_BIT__ 8
// AARCH64-DARWIN: #define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// AARCH64-DARWIN: #define __DBL_DIG__ 15
// AARCH64-DARWIN: #define __DBL_EPSILON__ 2.2204460492503131e-16
// AARCH64-DARWIN: #define __DBL_HAS_DENORM__ 1
// AARCH64-DARWIN: #define __DBL_HAS_INFINITY__ 1
// AARCH64-DARWIN: #define __DBL_HAS_QUIET_NAN__ 1
// AARCH64-DARWIN: #define __DBL_MANT_DIG__ 53
// AARCH64-DARWIN: #define __DBL_MAX_10_EXP__ 308
// AARCH64-DARWIN: #define __DBL_MAX_EXP__ 1024
// AARCH64-DARWIN: #define __DBL_MAX__ 1.7976931348623157e+308
// AARCH64-DARWIN: #define __DBL_MIN_10_EXP__ (-307)
// AARCH64-DARWIN: #define __DBL_MIN_EXP__ (-1021)
// AARCH64-DARWIN: #define __DBL_MIN__ 2.2250738585072014e-308
// AARCH64-DARWIN: #define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// AARCH64-DARWIN: #define __FLT_DENORM_MIN__ 1.40129846e-45F
// AARCH64-DARWIN: #define __FLT_DIG__ 6
// AARCH64-DARWIN: #define __FLT_EPSILON__ 1.19209290e-7F
// AARCH64-DARWIN: #define __FLT_EVAL_METHOD__ 0
// AARCH64-DARWIN: #define __FLT_HAS_DENORM__ 1
// AARCH64-DARWIN: #define __FLT_HAS_INFINITY__ 1
// AARCH64-DARWIN: #define __FLT_HAS_QUIET_NAN__ 1
// AARCH64-DARWIN: #define __FLT_MANT_DIG__ 24
// AARCH64-DARWIN: #define __FLT_MAX_10_EXP__ 38
// AARCH64-DARWIN: #define __FLT_MAX_EXP__ 128
// AARCH64-DARWIN: #define __FLT_MAX__ 3.40282347e+38F
// AARCH64-DARWIN: #define __FLT_MIN_10_EXP__ (-37)
// AARCH64-DARWIN: #define __FLT_MIN_EXP__ (-125)
// AARCH64-DARWIN: #define __FLT_MIN__ 1.17549435e-38F
// AARCH64-DARWIN: #define __FLT_RADIX__ 2
// AARCH64-DARWIN: #define __INT16_C_SUFFIX__
// AARCH64-DARWIN: #define __INT16_FMTd__ "hd"
// AARCH64-DARWIN: #define __INT16_FMTi__ "hi"
// AARCH64-DARWIN: #define __INT16_MAX__ 32767
// AARCH64-DARWIN: #define __INT16_TYPE__ short
// AARCH64-DARWIN: #define __INT32_C_SUFFIX__
// AARCH64-DARWIN: #define __INT32_FMTd__ "d"
// AARCH64-DARWIN: #define __INT32_FMTi__ "i"
// AARCH64-DARWIN: #define __INT32_MAX__ 2147483647
// AARCH64-DARWIN: #define __INT32_TYPE__ int
// AARCH64-DARWIN: #define __INT64_C_SUFFIX__ LL
// AARCH64-DARWIN: #define __INT64_FMTd__ "lld"
// AARCH64-DARWIN: #define __INT64_FMTi__ "lli"
// AARCH64-DARWIN: #define __INT64_MAX__ 9223372036854775807LL
// AARCH64-DARWIN: #define __INT64_TYPE__ long long int
// AARCH64-DARWIN: #define __INT8_C_SUFFIX__
// AARCH64-DARWIN: #define __INT8_FMTd__ "hhd"
// AARCH64-DARWIN: #define __INT8_FMTi__ "hhi"
// AARCH64-DARWIN: #define __INT8_MAX__ 127
// AARCH64-DARWIN: #define __INT8_TYPE__ signed char
// AARCH64-DARWIN: #define __INTMAX_C_SUFFIX__ L
// AARCH64-DARWIN: #define __INTMAX_FMTd__ "ld"
// AARCH64-DARWIN: #define __INTMAX_FMTi__ "li"
// AARCH64-DARWIN: #define __INTMAX_MAX__ 9223372036854775807L
// AARCH64-DARWIN: #define __INTMAX_TYPE__ long int
// AARCH64-DARWIN: #define __INTMAX_WIDTH__ 64
// AARCH64-DARWIN: #define __INTPTR_FMTd__ "ld"
// AARCH64-DARWIN: #define __INTPTR_FMTi__ "li"
// AARCH64-DARWIN: #define __INTPTR_MAX__ 9223372036854775807L
// AARCH64-DARWIN: #define __INTPTR_TYPE__ long int
// AARCH64-DARWIN: #define __INTPTR_WIDTH__ 64
// AARCH64-DARWIN: #define __INT_FAST16_FMTd__ "hd"
// AARCH64-DARWIN: #define __INT_FAST16_FMTi__ "hi"
// AARCH64-DARWIN: #define __INT_FAST16_MAX__ 32767
// AARCH64-DARWIN: #define __INT_FAST16_TYPE__ short
// AARCH64-DARWIN: #define __INT_FAST32_FMTd__ "d"
// AARCH64-DARWIN: #define __INT_FAST32_FMTi__ "i"
// AARCH64-DARWIN: #define __INT_FAST32_MAX__ 2147483647
// AARCH64-DARWIN: #define __INT_FAST32_TYPE__ int
// AARCH64-DARWIN: #define __INT_FAST64_FMTd__ "ld"
// AARCH64-DARWIN: #define __INT_FAST64_FMTi__ "li"
// AARCH64-DARWIN: #define __INT_FAST64_MAX__ 9223372036854775807L
// AARCH64-DARWIN: #define __INT_FAST64_TYPE__ long int
// AARCH64-DARWIN: #define __INT_FAST8_FMTd__ "hhd"
// AARCH64-DARWIN: #define __INT_FAST8_FMTi__ "hhi"
// AARCH64-DARWIN: #define __INT_FAST8_MAX__ 127
// AARCH64-DARWIN: #define __INT_FAST8_TYPE__ signed char
// AARCH64-DARWIN: #define __INT_LEAST16_FMTd__ "hd"
// AARCH64-DARWIN: #define __INT_LEAST16_FMTi__ "hi"
// AARCH64-DARWIN: #define __INT_LEAST16_MAX__ 32767
// AARCH64-DARWIN: #define __INT_LEAST16_TYPE__ short
// AARCH64-DARWIN: #define __INT_LEAST32_FMTd__ "d"
// AARCH64-DARWIN: #define __INT_LEAST32_FMTi__ "i"
// AARCH64-DARWIN: #define __INT_LEAST32_MAX__ 2147483647
// AARCH64-DARWIN: #define __INT_LEAST32_TYPE__ int
// AARCH64-DARWIN: #define __INT_LEAST64_FMTd__ "ld"
// AARCH64-DARWIN: #define __INT_LEAST64_FMTi__ "li"
// AARCH64-DARWIN: #define __INT_LEAST64_MAX__ 9223372036854775807L
// AARCH64-DARWIN: #define __INT_LEAST64_TYPE__ long int
// AARCH64-DARWIN: #define __INT_LEAST8_FMTd__ "hhd"
// AARCH64-DARWIN: #define __INT_LEAST8_FMTi__ "hhi"
// AARCH64-DARWIN: #define __INT_LEAST8_MAX__ 127
// AARCH64-DARWIN: #define __INT_LEAST8_TYPE__ signed char
// AARCH64-DARWIN: #define __INT_MAX__ 2147483647
// AARCH64-DARWIN: #define __LDBL_DENORM_MIN__ 4.9406564584124654e-324L
// AARCH64-DARWIN: #define __LDBL_DIG__ 15
// AARCH64-DARWIN: #define __LDBL_EPSILON__ 2.2204460492503131e-16L
// AARCH64-DARWIN: #define __LDBL_HAS_DENORM__ 1
// AARCH64-DARWIN: #define __LDBL_HAS_INFINITY__ 1
// AARCH64-DARWIN: #define __LDBL_HAS_QUIET_NAN__ 1
// AARCH64-DARWIN: #define __LDBL_MANT_DIG__ 53
// AARCH64-DARWIN: #define __LDBL_MAX_10_EXP__ 308
// AARCH64-DARWIN: #define __LDBL_MAX_EXP__ 1024
// AARCH64-DARWIN: #define __LDBL_MAX__ 1.7976931348623157e+308L
// AARCH64-DARWIN: #define __LDBL_MIN_10_EXP__ (-307)
// AARCH64-DARWIN: #define __LDBL_MIN_EXP__ (-1021)
// AARCH64-DARWIN: #define __LDBL_MIN__ 2.2250738585072014e-308L
// AARCH64-DARWIN: #define __LONG_LONG_MAX__ 9223372036854775807LL
// AARCH64-DARWIN: #define __LONG_MAX__ 9223372036854775807L
// AARCH64-DARWIN: #define __LP64__ 1
// AARCH64-DARWIN: #define __POINTER_WIDTH__ 64
// AARCH64-DARWIN: #define __PTRDIFF_TYPE__ long int
// AARCH64-DARWIN: #define __PTRDIFF_WIDTH__ 64
// AARCH64-DARWIN: #define __SCHAR_MAX__ 127
// AARCH64-DARWIN: #define __SHRT_MAX__ 32767
// AARCH64-DARWIN: #define __SIG_ATOMIC_MAX__ 2147483647
// AARCH64-DARWIN: #define __SIG_ATOMIC_WIDTH__ 32
// AARCH64-DARWIN: #define __SIZEOF_DOUBLE__ 8
// AARCH64-DARWIN: #define __SIZEOF_FLOAT__ 4
// AARCH64-DARWIN: #define __SIZEOF_INT128__ 16
// AARCH64-DARWIN: #define __SIZEOF_INT__ 4
// AARCH64-DARWIN: #define __SIZEOF_LONG_DOUBLE__ 8
// AARCH64-DARWIN: #define __SIZEOF_LONG_LONG__ 8
// AARCH64-DARWIN: #define __SIZEOF_LONG__ 8
// AARCH64-DARWIN: #define __SIZEOF_POINTER__ 8
// AARCH64-DARWIN: #define __SIZEOF_PTRDIFF_T__ 8
// AARCH64-DARWIN: #define __SIZEOF_SHORT__ 2
// AARCH64-DARWIN: #define __SIZEOF_SIZE_T__ 8
// AARCH64-DARWIN: #define __SIZEOF_WCHAR_T__ 4
// AARCH64-DARWIN: #define __SIZEOF_WINT_T__ 4
// AARCH64-DARWIN: #define __SIZE_MAX__ 18446744073709551615UL
// AARCH64-DARWIN: #define __SIZE_TYPE__ long unsigned int
// AARCH64-DARWIN: #define __SIZE_WIDTH__ 64
// AARCH64-DARWIN: #define __UINT16_C_SUFFIX__
// AARCH64-DARWIN: #define __UINT16_MAX__ 65535
// AARCH64-DARWIN: #define __UINT16_TYPE__ unsigned short
// AARCH64-DARWIN: #define __UINT32_C_SUFFIX__ U
// AARCH64-DARWIN: #define __UINT32_MAX__ 4294967295U
// AARCH64-DARWIN: #define __UINT32_TYPE__ unsigned int
// AARCH64-DARWIN: #define __UINT64_C_SUFFIX__ ULL
// AARCH64-DARWIN: #define __UINT64_MAX__ 18446744073709551615ULL
// AARCH64-DARWIN: #define __UINT64_TYPE__ long long unsigned int
// AARCH64-DARWIN: #define __UINT8_C_SUFFIX__
// AARCH64-DARWIN: #define __UINT8_MAX__ 255
// AARCH64-DARWIN: #define __UINT8_TYPE__ unsigned char
// AARCH64-DARWIN: #define __UINTMAX_C_SUFFIX__ UL
// AARCH64-DARWIN: #define __UINTMAX_MAX__ 18446744073709551615UL
// AARCH64-DARWIN: #define __UINTMAX_TYPE__ long unsigned int
// AARCH64-DARWIN: #define __UINTMAX_WIDTH__ 64
// AARCH64-DARWIN: #define __UINTPTR_MAX__ 18446744073709551615UL
// AARCH64-DARWIN: #define __UINTPTR_TYPE__ long unsigned int
// AARCH64-DARWIN: #define __UINTPTR_WIDTH__ 64
// AARCH64-DARWIN: #define __UINT_FAST16_MAX__ 65535
// AARCH64-DARWIN: #define __UINT_FAST16_TYPE__ unsigned short
// AARCH64-DARWIN: #define __UINT_FAST32_MAX__ 4294967295U
// AARCH64-DARWIN: #define __UINT_FAST32_TYPE__ unsigned int
// AARCH64-DARWIN: #define __UINT_FAST64_MAX__ 18446744073709551615UL
// AARCH64-DARWIN: #define __UINT_FAST64_TYPE__ long unsigned int
// AARCH64-DARWIN: #define __UINT_FAST8_MAX__ 255
// AARCH64-DARWIN: #define __UINT_FAST8_TYPE__ unsigned char
// AARCH64-DARWIN: #define __UINT_LEAST16_MAX__ 65535
// AARCH64-DARWIN: #define __UINT_LEAST16_TYPE__ unsigned short
// AARCH64-DARWIN: #define __UINT_LEAST32_MAX__ 4294967295U
// AARCH64-DARWIN: #define __UINT_LEAST32_TYPE__ unsigned int
// AARCH64-DARWIN: #define __UINT_LEAST64_MAX__ 18446744073709551615UL
// AARCH64-DARWIN: #define __UINT_LEAST64_TYPE__ long unsigned int
// AARCH64-DARWIN: #define __UINT_LEAST8_MAX__ 255
// AARCH64-DARWIN: #define __UINT_LEAST8_TYPE__ unsigned char
// AARCH64-DARWIN: #define __USER_LABEL_PREFIX__ _
// AARCH64-DARWIN: #define __WCHAR_MAX__ 2147483647
// AARCH64-DARWIN: #define __WCHAR_TYPE__ int
// AARCH64-DARWIN-NOT: #define __WCHAR_UNSIGNED__
// AARCH64-DARWIN: #define __WCHAR_WIDTH__ 32
// AARCH64-DARWIN: #define __WINT_TYPE__ int
// AARCH64-DARWIN: #define __WINT_WIDTH__ 32
// AARCH64-DARWIN: #define __aarch64__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=arm-none-none < /dev/null | FileCheck -match-full-lines -check-prefix ARM %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=arm-none-none < /dev/null | FileCheck -match-full-lines -check-prefix ARM -check-prefix ARM-CXX %s
//
// ARM-NOT:#define _LP64
// ARM:#define __APCS_32__ 1
// ARM-NOT:#define __ARMEB__ 1
// ARM:#define __ARMEL__ 1
// ARM:#define __ARM_ARCH_4T__ 1
// ARM-NOT:#define __ARM_BIG_ENDIAN 1
// ARM:#define __BIGGEST_ALIGNMENT__ 8
// ARM:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// ARM:#define __CHAR16_TYPE__ unsigned short
// ARM:#define __CHAR32_TYPE__ unsigned int
// ARM:#define __CHAR_BIT__ 8
// ARM:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// ARM:#define __DBL_DIG__ 15
// ARM:#define __DBL_EPSILON__ 2.2204460492503131e-16
// ARM:#define __DBL_HAS_DENORM__ 1
// ARM:#define __DBL_HAS_INFINITY__ 1
// ARM:#define __DBL_HAS_QUIET_NAN__ 1
// ARM:#define __DBL_MANT_DIG__ 53
// ARM:#define __DBL_MAX_10_EXP__ 308
// ARM:#define __DBL_MAX_EXP__ 1024
// ARM:#define __DBL_MAX__ 1.7976931348623157e+308
// ARM:#define __DBL_MIN_10_EXP__ (-307)
// ARM:#define __DBL_MIN_EXP__ (-1021)
// ARM:#define __DBL_MIN__ 2.2250738585072014e-308
// ARM:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// ARM:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// ARM:#define __FLT_DIG__ 6
// ARM:#define __FLT_EPSILON__ 1.19209290e-7F
// ARM:#define __FLT_EVAL_METHOD__ 0
// ARM:#define __FLT_HAS_DENORM__ 1
// ARM:#define __FLT_HAS_INFINITY__ 1
// ARM:#define __FLT_HAS_QUIET_NAN__ 1
// ARM:#define __FLT_MANT_DIG__ 24
// ARM:#define __FLT_MAX_10_EXP__ 38
// ARM:#define __FLT_MAX_EXP__ 128
// ARM:#define __FLT_MAX__ 3.40282347e+38F
// ARM:#define __FLT_MIN_10_EXP__ (-37)
// ARM:#define __FLT_MIN_EXP__ (-125)
// ARM:#define __FLT_MIN__ 1.17549435e-38F
// ARM:#define __FLT_RADIX__ 2
// ARM:#define __INT16_C_SUFFIX__
// ARM:#define __INT16_FMTd__ "hd"
// ARM:#define __INT16_FMTi__ "hi"
// ARM:#define __INT16_MAX__ 32767
// ARM:#define __INT16_TYPE__ short
// ARM:#define __INT32_C_SUFFIX__
// ARM:#define __INT32_FMTd__ "d"
// ARM:#define __INT32_FMTi__ "i"
// ARM:#define __INT32_MAX__ 2147483647
// ARM:#define __INT32_TYPE__ int
// ARM:#define __INT64_C_SUFFIX__ LL
// ARM:#define __INT64_FMTd__ "lld"
// ARM:#define __INT64_FMTi__ "lli"
// ARM:#define __INT64_MAX__ 9223372036854775807LL
// ARM:#define __INT64_TYPE__ long long int
// ARM:#define __INT8_C_SUFFIX__
// ARM:#define __INT8_FMTd__ "hhd"
// ARM:#define __INT8_FMTi__ "hhi"
// ARM:#define __INT8_MAX__ 127
// ARM:#define __INT8_TYPE__ signed char
// ARM:#define __INTMAX_C_SUFFIX__ LL
// ARM:#define __INTMAX_FMTd__ "lld"
// ARM:#define __INTMAX_FMTi__ "lli"
// ARM:#define __INTMAX_MAX__ 9223372036854775807LL
// ARM:#define __INTMAX_TYPE__ long long int
// ARM:#define __INTMAX_WIDTH__ 64
// ARM:#define __INTPTR_FMTd__ "ld"
// ARM:#define __INTPTR_FMTi__ "li"
// ARM:#define __INTPTR_MAX__ 2147483647L
// ARM:#define __INTPTR_TYPE__ long int
// ARM:#define __INTPTR_WIDTH__ 32
// ARM:#define __INT_FAST16_FMTd__ "hd"
// ARM:#define __INT_FAST16_FMTi__ "hi"
// ARM:#define __INT_FAST16_MAX__ 32767
// ARM:#define __INT_FAST16_TYPE__ short
// ARM:#define __INT_FAST32_FMTd__ "d"
// ARM:#define __INT_FAST32_FMTi__ "i"
// ARM:#define __INT_FAST32_MAX__ 2147483647
// ARM:#define __INT_FAST32_TYPE__ int
// ARM:#define __INT_FAST64_FMTd__ "lld"
// ARM:#define __INT_FAST64_FMTi__ "lli"
// ARM:#define __INT_FAST64_MAX__ 9223372036854775807LL
// ARM:#define __INT_FAST64_TYPE__ long long int
// ARM:#define __INT_FAST8_FMTd__ "hhd"
// ARM:#define __INT_FAST8_FMTi__ "hhi"
// ARM:#define __INT_FAST8_MAX__ 127
// ARM:#define __INT_FAST8_TYPE__ signed char
// ARM:#define __INT_LEAST16_FMTd__ "hd"
// ARM:#define __INT_LEAST16_FMTi__ "hi"
// ARM:#define __INT_LEAST16_MAX__ 32767
// ARM:#define __INT_LEAST16_TYPE__ short
// ARM:#define __INT_LEAST32_FMTd__ "d"
// ARM:#define __INT_LEAST32_FMTi__ "i"
// ARM:#define __INT_LEAST32_MAX__ 2147483647
// ARM:#define __INT_LEAST32_TYPE__ int
// ARM:#define __INT_LEAST64_FMTd__ "lld"
// ARM:#define __INT_LEAST64_FMTi__ "lli"
// ARM:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// ARM:#define __INT_LEAST64_TYPE__ long long int
// ARM:#define __INT_LEAST8_FMTd__ "hhd"
// ARM:#define __INT_LEAST8_FMTi__ "hhi"
// ARM:#define __INT_LEAST8_MAX__ 127
// ARM:#define __INT_LEAST8_TYPE__ signed char
// ARM:#define __INT_MAX__ 2147483647
// ARM:#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324L
// ARM:#define __LDBL_DIG__ 15
// ARM:#define __LDBL_EPSILON__ 2.2204460492503131e-16L
// ARM:#define __LDBL_HAS_DENORM__ 1
// ARM:#define __LDBL_HAS_INFINITY__ 1
// ARM:#define __LDBL_HAS_QUIET_NAN__ 1
// ARM:#define __LDBL_MANT_DIG__ 53
// ARM:#define __LDBL_MAX_10_EXP__ 308
// ARM:#define __LDBL_MAX_EXP__ 1024
// ARM:#define __LDBL_MAX__ 1.7976931348623157e+308L
// ARM:#define __LDBL_MIN_10_EXP__ (-307)
// ARM:#define __LDBL_MIN_EXP__ (-1021)
// ARM:#define __LDBL_MIN__ 2.2250738585072014e-308L
// ARM:#define __LITTLE_ENDIAN__ 1
// ARM:#define __LONG_LONG_MAX__ 9223372036854775807LL
// ARM:#define __LONG_MAX__ 2147483647L
// ARM-NOT:#define __LP64__
// ARM:#define __POINTER_WIDTH__ 32
// ARM:#define __PTRDIFF_TYPE__ int
// ARM:#define __PTRDIFF_WIDTH__ 32
// ARM:#define __REGISTER_PREFIX__
// ARM:#define __SCHAR_MAX__ 127
// ARM:#define __SHRT_MAX__ 32767
// ARM:#define __SIG_ATOMIC_MAX__ 2147483647
// ARM:#define __SIG_ATOMIC_WIDTH__ 32
// ARM:#define __SIZEOF_DOUBLE__ 8
// ARM:#define __SIZEOF_FLOAT__ 4
// ARM:#define __SIZEOF_INT__ 4
// ARM:#define __SIZEOF_LONG_DOUBLE__ 8
// ARM:#define __SIZEOF_LONG_LONG__ 8
// ARM:#define __SIZEOF_LONG__ 4
// ARM:#define __SIZEOF_POINTER__ 4
// ARM:#define __SIZEOF_PTRDIFF_T__ 4
// ARM:#define __SIZEOF_SHORT__ 2
// ARM:#define __SIZEOF_SIZE_T__ 4
// ARM:#define __SIZEOF_WCHAR_T__ 4
// ARM:#define __SIZEOF_WINT_T__ 4
// ARM:#define __SIZE_MAX__ 4294967295U
// ARM:#define __SIZE_TYPE__ unsigned int
// ARM:#define __SIZE_WIDTH__ 32
// ARM-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 8U
// ARM:#define __UINT16_C_SUFFIX__
// ARM:#define __UINT16_MAX__ 65535
// ARM:#define __UINT16_TYPE__ unsigned short
// ARM:#define __UINT32_C_SUFFIX__ U
// ARM:#define __UINT32_MAX__ 4294967295U
// ARM:#define __UINT32_TYPE__ unsigned int
// ARM:#define __UINT64_C_SUFFIX__ ULL
// ARM:#define __UINT64_MAX__ 18446744073709551615ULL
// ARM:#define __UINT64_TYPE__ long long unsigned int
// ARM:#define __UINT8_C_SUFFIX__
// ARM:#define __UINT8_MAX__ 255
// ARM:#define __UINT8_TYPE__ unsigned char
// ARM:#define __UINTMAX_C_SUFFIX__ ULL
// ARM:#define __UINTMAX_MAX__ 18446744073709551615ULL
// ARM:#define __UINTMAX_TYPE__ long long unsigned int
// ARM:#define __UINTMAX_WIDTH__ 64
// ARM:#define __UINTPTR_MAX__ 4294967295UL
// ARM:#define __UINTPTR_TYPE__ long unsigned int
// ARM:#define __UINTPTR_WIDTH__ 32
// ARM:#define __UINT_FAST16_MAX__ 65535
// ARM:#define __UINT_FAST16_TYPE__ unsigned short
// ARM:#define __UINT_FAST32_MAX__ 4294967295U
// ARM:#define __UINT_FAST32_TYPE__ unsigned int
// ARM:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// ARM:#define __UINT_FAST64_TYPE__ long long unsigned int
// ARM:#define __UINT_FAST8_MAX__ 255
// ARM:#define __UINT_FAST8_TYPE__ unsigned char
// ARM:#define __UINT_LEAST16_MAX__ 65535
// ARM:#define __UINT_LEAST16_TYPE__ unsigned short
// ARM:#define __UINT_LEAST32_MAX__ 4294967295U
// ARM:#define __UINT_LEAST32_TYPE__ unsigned int
// ARM:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// ARM:#define __UINT_LEAST64_TYPE__ long long unsigned int
// ARM:#define __UINT_LEAST8_MAX__ 255
// ARM:#define __UINT_LEAST8_TYPE__ unsigned char
// ARM:#define __USER_LABEL_PREFIX__
// ARM:#define __WCHAR_MAX__ 4294967295U
// ARM:#define __WCHAR_TYPE__ unsigned int
// ARM:#define __WCHAR_WIDTH__ 32
// ARM:#define __WINT_TYPE__ int
// ARM:#define __WINT_WIDTH__ 32
// ARM:#define __arm 1
// ARM:#define __arm__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=armeb-none-none < /dev/null | FileCheck -match-full-lines -check-prefix ARM-BE %s
//
// ARM-BE-NOT:#define _LP64
// ARM-BE:#define __APCS_32__ 1
// ARM-BE:#define __ARMEB__ 1
// ARM-BE-NOT:#define __ARMEL__ 1
// ARM-BE:#define __ARM_ARCH_4T__ 1
// ARM-BE:#define __ARM_BIG_ENDIAN 1
// ARM-BE:#define __BIGGEST_ALIGNMENT__ 8
// ARM-BE:#define __BIG_ENDIAN__ 1
// ARM-BE:#define __BYTE_ORDER__ __ORDER_BIG_ENDIAN__
// ARM-BE:#define __CHAR16_TYPE__ unsigned short
// ARM-BE:#define __CHAR32_TYPE__ unsigned int
// ARM-BE:#define __CHAR_BIT__ 8
// ARM-BE:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// ARM-BE:#define __DBL_DIG__ 15
// ARM-BE:#define __DBL_EPSILON__ 2.2204460492503131e-16
// ARM-BE:#define __DBL_HAS_DENORM__ 1
// ARM-BE:#define __DBL_HAS_INFINITY__ 1
// ARM-BE:#define __DBL_HAS_QUIET_NAN__ 1
// ARM-BE:#define __DBL_MANT_DIG__ 53
// ARM-BE:#define __DBL_MAX_10_EXP__ 308
// ARM-BE:#define __DBL_MAX_EXP__ 1024
// ARM-BE:#define __DBL_MAX__ 1.7976931348623157e+308
// ARM-BE:#define __DBL_MIN_10_EXP__ (-307)
// ARM-BE:#define __DBL_MIN_EXP__ (-1021)
// ARM-BE:#define __DBL_MIN__ 2.2250738585072014e-308
// ARM-BE:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// ARM-BE:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// ARM-BE:#define __FLT_DIG__ 6
// ARM-BE:#define __FLT_EPSILON__ 1.19209290e-7F
// ARM-BE:#define __FLT_EVAL_METHOD__ 0
// ARM-BE:#define __FLT_HAS_DENORM__ 1
// ARM-BE:#define __FLT_HAS_INFINITY__ 1
// ARM-BE:#define __FLT_HAS_QUIET_NAN__ 1
// ARM-BE:#define __FLT_MANT_DIG__ 24
// ARM-BE:#define __FLT_MAX_10_EXP__ 38
// ARM-BE:#define __FLT_MAX_EXP__ 128
// ARM-BE:#define __FLT_MAX__ 3.40282347e+38F
// ARM-BE:#define __FLT_MIN_10_EXP__ (-37)
// ARM-BE:#define __FLT_MIN_EXP__ (-125)
// ARM-BE:#define __FLT_MIN__ 1.17549435e-38F
// ARM-BE:#define __FLT_RADIX__ 2
// ARM-BE:#define __INT16_C_SUFFIX__
// ARM-BE:#define __INT16_FMTd__ "hd"
// ARM-BE:#define __INT16_FMTi__ "hi"
// ARM-BE:#define __INT16_MAX__ 32767
// ARM-BE:#define __INT16_TYPE__ short
// ARM-BE:#define __INT32_C_SUFFIX__
// ARM-BE:#define __INT32_FMTd__ "d"
// ARM-BE:#define __INT32_FMTi__ "i"
// ARM-BE:#define __INT32_MAX__ 2147483647
// ARM-BE:#define __INT32_TYPE__ int
// ARM-BE:#define __INT64_C_SUFFIX__ LL
// ARM-BE:#define __INT64_FMTd__ "lld"
// ARM-BE:#define __INT64_FMTi__ "lli"
// ARM-BE:#define __INT64_MAX__ 9223372036854775807LL
// ARM-BE:#define __INT64_TYPE__ long long int
// ARM-BE:#define __INT8_C_SUFFIX__
// ARM-BE:#define __INT8_FMTd__ "hhd"
// ARM-BE:#define __INT8_FMTi__ "hhi"
// ARM-BE:#define __INT8_MAX__ 127
// ARM-BE:#define __INT8_TYPE__ signed char
// ARM-BE:#define __INTMAX_C_SUFFIX__ LL
// ARM-BE:#define __INTMAX_FMTd__ "lld"
// ARM-BE:#define __INTMAX_FMTi__ "lli"
// ARM-BE:#define __INTMAX_MAX__ 9223372036854775807LL
// ARM-BE:#define __INTMAX_TYPE__ long long int
// ARM-BE:#define __INTMAX_WIDTH__ 64
// ARM-BE:#define __INTPTR_FMTd__ "ld"
// ARM-BE:#define __INTPTR_FMTi__ "li"
// ARM-BE:#define __INTPTR_MAX__ 2147483647L
// ARM-BE:#define __INTPTR_TYPE__ long int
// ARM-BE:#define __INTPTR_WIDTH__ 32
// ARM-BE:#define __INT_FAST16_FMTd__ "hd"
// ARM-BE:#define __INT_FAST16_FMTi__ "hi"
// ARM-BE:#define __INT_FAST16_MAX__ 32767
// ARM-BE:#define __INT_FAST16_TYPE__ short
// ARM-BE:#define __INT_FAST32_FMTd__ "d"
// ARM-BE:#define __INT_FAST32_FMTi__ "i"
// ARM-BE:#define __INT_FAST32_MAX__ 2147483647
// ARM-BE:#define __INT_FAST32_TYPE__ int
// ARM-BE:#define __INT_FAST64_FMTd__ "lld"
// ARM-BE:#define __INT_FAST64_FMTi__ "lli"
// ARM-BE:#define __INT_FAST64_MAX__ 9223372036854775807LL
// ARM-BE:#define __INT_FAST64_TYPE__ long long int
// ARM-BE:#define __INT_FAST8_FMTd__ "hhd"
// ARM-BE:#define __INT_FAST8_FMTi__ "hhi"
// ARM-BE:#define __INT_FAST8_MAX__ 127
// ARM-BE:#define __INT_FAST8_TYPE__ signed char
// ARM-BE:#define __INT_LEAST16_FMTd__ "hd"
// ARM-BE:#define __INT_LEAST16_FMTi__ "hi"
// ARM-BE:#define __INT_LEAST16_MAX__ 32767
// ARM-BE:#define __INT_LEAST16_TYPE__ short
// ARM-BE:#define __INT_LEAST32_FMTd__ "d"
// ARM-BE:#define __INT_LEAST32_FMTi__ "i"
// ARM-BE:#define __INT_LEAST32_MAX__ 2147483647
// ARM-BE:#define __INT_LEAST32_TYPE__ int
// ARM-BE:#define __INT_LEAST64_FMTd__ "lld"
// ARM-BE:#define __INT_LEAST64_FMTi__ "lli"
// ARM-BE:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// ARM-BE:#define __INT_LEAST64_TYPE__ long long int
// ARM-BE:#define __INT_LEAST8_FMTd__ "hhd"
// ARM-BE:#define __INT_LEAST8_FMTi__ "hhi"
// ARM-BE:#define __INT_LEAST8_MAX__ 127
// ARM-BE:#define __INT_LEAST8_TYPE__ signed char
// ARM-BE:#define __INT_MAX__ 2147483647
// ARM-BE:#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324L
// ARM-BE:#define __LDBL_DIG__ 15
// ARM-BE:#define __LDBL_EPSILON__ 2.2204460492503131e-16L
// ARM-BE:#define __LDBL_HAS_DENORM__ 1
// ARM-BE:#define __LDBL_HAS_INFINITY__ 1
// ARM-BE:#define __LDBL_HAS_QUIET_NAN__ 1
// ARM-BE:#define __LDBL_MANT_DIG__ 53
// ARM-BE:#define __LDBL_MAX_10_EXP__ 308
// ARM-BE:#define __LDBL_MAX_EXP__ 1024
// ARM-BE:#define __LDBL_MAX__ 1.7976931348623157e+308L
// ARM-BE:#define __LDBL_MIN_10_EXP__ (-307)
// ARM-BE:#define __LDBL_MIN_EXP__ (-1021)
// ARM-BE:#define __LDBL_MIN__ 2.2250738585072014e-308L
// ARM-BE:#define __LONG_LONG_MAX__ 9223372036854775807LL
// ARM-BE:#define __LONG_MAX__ 2147483647L
// ARM-BE-NOT:#define __LP64__
// ARM-BE:#define __POINTER_WIDTH__ 32
// ARM-BE:#define __PTRDIFF_TYPE__ int
// ARM-BE:#define __PTRDIFF_WIDTH__ 32
// ARM-BE:#define __REGISTER_PREFIX__
// ARM-BE:#define __SCHAR_MAX__ 127
// ARM-BE:#define __SHRT_MAX__ 32767
// ARM-BE:#define __SIG_ATOMIC_MAX__ 2147483647
// ARM-BE:#define __SIG_ATOMIC_WIDTH__ 32
// ARM-BE:#define __SIZEOF_DOUBLE__ 8
// ARM-BE:#define __SIZEOF_FLOAT__ 4
// ARM-BE:#define __SIZEOF_INT__ 4
// ARM-BE:#define __SIZEOF_LONG_DOUBLE__ 8
// ARM-BE:#define __SIZEOF_LONG_LONG__ 8
// ARM-BE:#define __SIZEOF_LONG__ 4
// ARM-BE:#define __SIZEOF_POINTER__ 4
// ARM-BE:#define __SIZEOF_PTRDIFF_T__ 4
// ARM-BE:#define __SIZEOF_SHORT__ 2
// ARM-BE:#define __SIZEOF_SIZE_T__ 4
// ARM-BE:#define __SIZEOF_WCHAR_T__ 4
// ARM-BE:#define __SIZEOF_WINT_T__ 4
// ARM-BE:#define __SIZE_MAX__ 4294967295U
// ARM-BE:#define __SIZE_TYPE__ unsigned int
// ARM-BE:#define __SIZE_WIDTH__ 32
// ARM-BE:#define __UINT16_C_SUFFIX__
// ARM-BE:#define __UINT16_MAX__ 65535
// ARM-BE:#define __UINT16_TYPE__ unsigned short
// ARM-BE:#define __UINT32_C_SUFFIX__ U
// ARM-BE:#define __UINT32_MAX__ 4294967295U
// ARM-BE:#define __UINT32_TYPE__ unsigned int
// ARM-BE:#define __UINT64_C_SUFFIX__ ULL
// ARM-BE:#define __UINT64_MAX__ 18446744073709551615ULL
// ARM-BE:#define __UINT64_TYPE__ long long unsigned int
// ARM-BE:#define __UINT8_C_SUFFIX__
// ARM-BE:#define __UINT8_MAX__ 255
// ARM-BE:#define __UINT8_TYPE__ unsigned char
// ARM-BE:#define __UINTMAX_C_SUFFIX__ ULL
// ARM-BE:#define __UINTMAX_MAX__ 18446744073709551615ULL
// ARM-BE:#define __UINTMAX_TYPE__ long long unsigned int
// ARM-BE:#define __UINTMAX_WIDTH__ 64
// ARM-BE:#define __UINTPTR_MAX__ 4294967295UL
// ARM-BE:#define __UINTPTR_TYPE__ long unsigned int
// ARM-BE:#define __UINTPTR_WIDTH__ 32
// ARM-BE:#define __UINT_FAST16_MAX__ 65535
// ARM-BE:#define __UINT_FAST16_TYPE__ unsigned short
// ARM-BE:#define __UINT_FAST32_MAX__ 4294967295U
// ARM-BE:#define __UINT_FAST32_TYPE__ unsigned int
// ARM-BE:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// ARM-BE:#define __UINT_FAST64_TYPE__ long long unsigned int
// ARM-BE:#define __UINT_FAST8_MAX__ 255
// ARM-BE:#define __UINT_FAST8_TYPE__ unsigned char
// ARM-BE:#define __UINT_LEAST16_MAX__ 65535
// ARM-BE:#define __UINT_LEAST16_TYPE__ unsigned short
// ARM-BE:#define __UINT_LEAST32_MAX__ 4294967295U
// ARM-BE:#define __UINT_LEAST32_TYPE__ unsigned int
// ARM-BE:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// ARM-BE:#define __UINT_LEAST64_TYPE__ long long unsigned int
// ARM-BE:#define __UINT_LEAST8_MAX__ 255
// ARM-BE:#define __UINT_LEAST8_TYPE__ unsigned char
// ARM-BE:#define __USER_LABEL_PREFIX__
// ARM-BE:#define __WCHAR_MAX__ 4294967295U
// ARM-BE:#define __WCHAR_TYPE__ unsigned int
// ARM-BE:#define __WCHAR_WIDTH__ 32
// ARM-BE:#define __WINT_TYPE__ int
// ARM-BE:#define __WINT_WIDTH__ 32
// ARM-BE:#define __arm 1
// ARM-BE:#define __arm__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=arm-none-linux-gnueabi -target-feature +soft-float -target-feature +soft-float-abi < /dev/null | FileCheck -match-full-lines -check-prefix ARMEABISOFTFP %s
//
// ARMEABISOFTFP-NOT:#define _LP64
// ARMEABISOFTFP:#define __APCS_32__ 1
// ARMEABISOFTFP-NOT:#define __ARMEB__ 1
// ARMEABISOFTFP:#define __ARMEL__ 1
// ARMEABISOFTFP:#define __ARM_ARCH 4
// ARMEABISOFTFP:#define __ARM_ARCH_4T__ 1
// ARMEABISOFTFP-NOT:#define __ARM_BIG_ENDIAN 1
// ARMEABISOFTFP:#define __ARM_EABI__ 1
// ARMEABISOFTFP:#define __ARM_PCS 1
// ARMEABISOFTFP-NOT:#define __ARM_PCS_VFP 1
// ARMEABISOFTFP:#define __BIGGEST_ALIGNMENT__ 8
// ARMEABISOFTFP:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// ARMEABISOFTFP:#define __CHAR16_TYPE__ unsigned short
// ARMEABISOFTFP:#define __CHAR32_TYPE__ unsigned int
// ARMEABISOFTFP:#define __CHAR_BIT__ 8
// ARMEABISOFTFP:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// ARMEABISOFTFP:#define __DBL_DIG__ 15
// ARMEABISOFTFP:#define __DBL_EPSILON__ 2.2204460492503131e-16
// ARMEABISOFTFP:#define __DBL_HAS_DENORM__ 1
// ARMEABISOFTFP:#define __DBL_HAS_INFINITY__ 1
// ARMEABISOFTFP:#define __DBL_HAS_QUIET_NAN__ 1
// ARMEABISOFTFP:#define __DBL_MANT_DIG__ 53
// ARMEABISOFTFP:#define __DBL_MAX_10_EXP__ 308
// ARMEABISOFTFP:#define __DBL_MAX_EXP__ 1024
// ARMEABISOFTFP:#define __DBL_MAX__ 1.7976931348623157e+308
// ARMEABISOFTFP:#define __DBL_MIN_10_EXP__ (-307)
// ARMEABISOFTFP:#define __DBL_MIN_EXP__ (-1021)
// ARMEABISOFTFP:#define __DBL_MIN__ 2.2250738585072014e-308
// ARMEABISOFTFP:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// ARMEABISOFTFP:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// ARMEABISOFTFP:#define __FLT_DIG__ 6
// ARMEABISOFTFP:#define __FLT_EPSILON__ 1.19209290e-7F
// ARMEABISOFTFP:#define __FLT_EVAL_METHOD__ 0
// ARMEABISOFTFP:#define __FLT_HAS_DENORM__ 1
// ARMEABISOFTFP:#define __FLT_HAS_INFINITY__ 1
// ARMEABISOFTFP:#define __FLT_HAS_QUIET_NAN__ 1
// ARMEABISOFTFP:#define __FLT_MANT_DIG__ 24
// ARMEABISOFTFP:#define __FLT_MAX_10_EXP__ 38
// ARMEABISOFTFP:#define __FLT_MAX_EXP__ 128
// ARMEABISOFTFP:#define __FLT_MAX__ 3.40282347e+38F
// ARMEABISOFTFP:#define __FLT_MIN_10_EXP__ (-37)
// ARMEABISOFTFP:#define __FLT_MIN_EXP__ (-125)
// ARMEABISOFTFP:#define __FLT_MIN__ 1.17549435e-38F
// ARMEABISOFTFP:#define __FLT_RADIX__ 2
// ARMEABISOFTFP:#define __INT16_C_SUFFIX__
// ARMEABISOFTFP:#define __INT16_FMTd__ "hd"
// ARMEABISOFTFP:#define __INT16_FMTi__ "hi"
// ARMEABISOFTFP:#define __INT16_MAX__ 32767
// ARMEABISOFTFP:#define __INT16_TYPE__ short
// ARMEABISOFTFP:#define __INT32_C_SUFFIX__
// ARMEABISOFTFP:#define __INT32_FMTd__ "d"
// ARMEABISOFTFP:#define __INT32_FMTi__ "i"
// ARMEABISOFTFP:#define __INT32_MAX__ 2147483647
// ARMEABISOFTFP:#define __INT32_TYPE__ int
// ARMEABISOFTFP:#define __INT64_C_SUFFIX__ LL
// ARMEABISOFTFP:#define __INT64_FMTd__ "lld"
// ARMEABISOFTFP:#define __INT64_FMTi__ "lli"
// ARMEABISOFTFP:#define __INT64_MAX__ 9223372036854775807LL
// ARMEABISOFTFP:#define __INT64_TYPE__ long long int
// ARMEABISOFTFP:#define __INT8_C_SUFFIX__
// ARMEABISOFTFP:#define __INT8_FMTd__ "hhd"
// ARMEABISOFTFP:#define __INT8_FMTi__ "hhi"
// ARMEABISOFTFP:#define __INT8_MAX__ 127
// ARMEABISOFTFP:#define __INT8_TYPE__ signed char
// ARMEABISOFTFP:#define __INTMAX_C_SUFFIX__ LL
// ARMEABISOFTFP:#define __INTMAX_FMTd__ "lld"
// ARMEABISOFTFP:#define __INTMAX_FMTi__ "lli"
// ARMEABISOFTFP:#define __INTMAX_MAX__ 9223372036854775807LL
// ARMEABISOFTFP:#define __INTMAX_TYPE__ long long int
// ARMEABISOFTFP:#define __INTMAX_WIDTH__ 64
// ARMEABISOFTFP:#define __INTPTR_FMTd__ "ld"
// ARMEABISOFTFP:#define __INTPTR_FMTi__ "li"
// ARMEABISOFTFP:#define __INTPTR_MAX__ 2147483647L
// ARMEABISOFTFP:#define __INTPTR_TYPE__ long int
// ARMEABISOFTFP:#define __INTPTR_WIDTH__ 32
// ARMEABISOFTFP:#define __INT_FAST16_FMTd__ "hd"
// ARMEABISOFTFP:#define __INT_FAST16_FMTi__ "hi"
// ARMEABISOFTFP:#define __INT_FAST16_MAX__ 32767
// ARMEABISOFTFP:#define __INT_FAST16_TYPE__ short
// ARMEABISOFTFP:#define __INT_FAST32_FMTd__ "d"
// ARMEABISOFTFP:#define __INT_FAST32_FMTi__ "i"
// ARMEABISOFTFP:#define __INT_FAST32_MAX__ 2147483647
// ARMEABISOFTFP:#define __INT_FAST32_TYPE__ int
// ARMEABISOFTFP:#define __INT_FAST64_FMTd__ "lld"
// ARMEABISOFTFP:#define __INT_FAST64_FMTi__ "lli"
// ARMEABISOFTFP:#define __INT_FAST64_MAX__ 9223372036854775807LL
// ARMEABISOFTFP:#define __INT_FAST64_TYPE__ long long int
// ARMEABISOFTFP:#define __INT_FAST8_FMTd__ "hhd"
// ARMEABISOFTFP:#define __INT_FAST8_FMTi__ "hhi"
// ARMEABISOFTFP:#define __INT_FAST8_MAX__ 127
// ARMEABISOFTFP:#define __INT_FAST8_TYPE__ signed char
// ARMEABISOFTFP:#define __INT_LEAST16_FMTd__ "hd"
// ARMEABISOFTFP:#define __INT_LEAST16_FMTi__ "hi"
// ARMEABISOFTFP:#define __INT_LEAST16_MAX__ 32767
// ARMEABISOFTFP:#define __INT_LEAST16_TYPE__ short
// ARMEABISOFTFP:#define __INT_LEAST32_FMTd__ "d"
// ARMEABISOFTFP:#define __INT_LEAST32_FMTi__ "i"
// ARMEABISOFTFP:#define __INT_LEAST32_MAX__ 2147483647
// ARMEABISOFTFP:#define __INT_LEAST32_TYPE__ int
// ARMEABISOFTFP:#define __INT_LEAST64_FMTd__ "lld"
// ARMEABISOFTFP:#define __INT_LEAST64_FMTi__ "lli"
// ARMEABISOFTFP:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// ARMEABISOFTFP:#define __INT_LEAST64_TYPE__ long long int
// ARMEABISOFTFP:#define __INT_LEAST8_FMTd__ "hhd"
// ARMEABISOFTFP:#define __INT_LEAST8_FMTi__ "hhi"
// ARMEABISOFTFP:#define __INT_LEAST8_MAX__ 127
// ARMEABISOFTFP:#define __INT_LEAST8_TYPE__ signed char
// ARMEABISOFTFP:#define __INT_MAX__ 2147483647
// ARMEABISOFTFP:#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324L
// ARMEABISOFTFP:#define __LDBL_DIG__ 15
// ARMEABISOFTFP:#define __LDBL_EPSILON__ 2.2204460492503131e-16L
// ARMEABISOFTFP:#define __LDBL_HAS_DENORM__ 1
// ARMEABISOFTFP:#define __LDBL_HAS_INFINITY__ 1
// ARMEABISOFTFP:#define __LDBL_HAS_QUIET_NAN__ 1
// ARMEABISOFTFP:#define __LDBL_MANT_DIG__ 53
// ARMEABISOFTFP:#define __LDBL_MAX_10_EXP__ 308
// ARMEABISOFTFP:#define __LDBL_MAX_EXP__ 1024
// ARMEABISOFTFP:#define __LDBL_MAX__ 1.7976931348623157e+308L
// ARMEABISOFTFP:#define __LDBL_MIN_10_EXP__ (-307)
// ARMEABISOFTFP:#define __LDBL_MIN_EXP__ (-1021)
// ARMEABISOFTFP:#define __LDBL_MIN__ 2.2250738585072014e-308L
// ARMEABISOFTFP:#define __LITTLE_ENDIAN__ 1
// ARMEABISOFTFP:#define __LONG_LONG_MAX__ 9223372036854775807LL
// ARMEABISOFTFP:#define __LONG_MAX__ 2147483647L
// ARMEABISOFTFP-NOT:#define __LP64__
// ARMEABISOFTFP:#define __POINTER_WIDTH__ 32
// ARMEABISOFTFP:#define __PTRDIFF_TYPE__ int
// ARMEABISOFTFP:#define __PTRDIFF_WIDTH__ 32
// ARMEABISOFTFP:#define __REGISTER_PREFIX__
// ARMEABISOFTFP:#define __SCHAR_MAX__ 127
// ARMEABISOFTFP:#define __SHRT_MAX__ 32767
// ARMEABISOFTFP:#define __SIG_ATOMIC_MAX__ 2147483647
// ARMEABISOFTFP:#define __SIG_ATOMIC_WIDTH__ 32
// ARMEABISOFTFP:#define __SIZEOF_DOUBLE__ 8
// ARMEABISOFTFP:#define __SIZEOF_FLOAT__ 4
// ARMEABISOFTFP:#define __SIZEOF_INT__ 4
// ARMEABISOFTFP:#define __SIZEOF_LONG_DOUBLE__ 8
// ARMEABISOFTFP:#define __SIZEOF_LONG_LONG__ 8
// ARMEABISOFTFP:#define __SIZEOF_LONG__ 4
// ARMEABISOFTFP:#define __SIZEOF_POINTER__ 4
// ARMEABISOFTFP:#define __SIZEOF_PTRDIFF_T__ 4
// ARMEABISOFTFP:#define __SIZEOF_SHORT__ 2
// ARMEABISOFTFP:#define __SIZEOF_SIZE_T__ 4
// ARMEABISOFTFP:#define __SIZEOF_WCHAR_T__ 4
// ARMEABISOFTFP:#define __SIZEOF_WINT_T__ 4
// ARMEABISOFTFP:#define __SIZE_MAX__ 4294967295U
// ARMEABISOFTFP:#define __SIZE_TYPE__ unsigned int
// ARMEABISOFTFP:#define __SIZE_WIDTH__ 32
// ARMEABISOFTFP:#define __SOFTFP__ 1
// ARMEABISOFTFP:#define __UINT16_C_SUFFIX__
// ARMEABISOFTFP:#define __UINT16_MAX__ 65535
// ARMEABISOFTFP:#define __UINT16_TYPE__ unsigned short
// ARMEABISOFTFP:#define __UINT32_C_SUFFIX__ U
// ARMEABISOFTFP:#define __UINT32_MAX__ 4294967295U
// ARMEABISOFTFP:#define __UINT32_TYPE__ unsigned int
// ARMEABISOFTFP:#define __UINT64_C_SUFFIX__ ULL
// ARMEABISOFTFP:#define __UINT64_MAX__ 18446744073709551615ULL
// ARMEABISOFTFP:#define __UINT64_TYPE__ long long unsigned int
// ARMEABISOFTFP:#define __UINT8_C_SUFFIX__
// ARMEABISOFTFP:#define __UINT8_MAX__ 255
// ARMEABISOFTFP:#define __UINT8_TYPE__ unsigned char
// ARMEABISOFTFP:#define __UINTMAX_C_SUFFIX__ ULL
// ARMEABISOFTFP:#define __UINTMAX_MAX__ 18446744073709551615ULL
// ARMEABISOFTFP:#define __UINTMAX_TYPE__ long long unsigned int
// ARMEABISOFTFP:#define __UINTMAX_WIDTH__ 64
// ARMEABISOFTFP:#define __UINTPTR_MAX__ 4294967295UL
// ARMEABISOFTFP:#define __UINTPTR_TYPE__ long unsigned int
// ARMEABISOFTFP:#define __UINTPTR_WIDTH__ 32
// ARMEABISOFTFP:#define __UINT_FAST16_MAX__ 65535
// ARMEABISOFTFP:#define __UINT_FAST16_TYPE__ unsigned short
// ARMEABISOFTFP:#define __UINT_FAST32_MAX__ 4294967295U
// ARMEABISOFTFP:#define __UINT_FAST32_TYPE__ unsigned int
// ARMEABISOFTFP:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// ARMEABISOFTFP:#define __UINT_FAST64_TYPE__ long long unsigned int
// ARMEABISOFTFP:#define __UINT_FAST8_MAX__ 255
// ARMEABISOFTFP:#define __UINT_FAST8_TYPE__ unsigned char
// ARMEABISOFTFP:#define __UINT_LEAST16_MAX__ 65535
// ARMEABISOFTFP:#define __UINT_LEAST16_TYPE__ unsigned short
// ARMEABISOFTFP:#define __UINT_LEAST32_MAX__ 4294967295U
// ARMEABISOFTFP:#define __UINT_LEAST32_TYPE__ unsigned int
// ARMEABISOFTFP:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// ARMEABISOFTFP:#define __UINT_LEAST64_TYPE__ long long unsigned int
// ARMEABISOFTFP:#define __UINT_LEAST8_MAX__ 255
// ARMEABISOFTFP:#define __UINT_LEAST8_TYPE__ unsigned char
// ARMEABISOFTFP:#define __USER_LABEL_PREFIX__
// ARMEABISOFTFP:#define __WCHAR_MAX__ 4294967295U
// ARMEABISOFTFP:#define __WCHAR_TYPE__ unsigned int
// ARMEABISOFTFP:#define __WCHAR_WIDTH__ 32
// ARMEABISOFTFP:#define __WINT_TYPE__ unsigned int
// ARMEABISOFTFP:#define __WINT_WIDTH__ 32
// ARMEABISOFTFP:#define __arm 1
// ARMEABISOFTFP:#define __arm__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=arm-none-linux-gnueabi < /dev/null | FileCheck -match-full-lines -check-prefix ARMEABIHARDFP %s
//
// ARMEABIHARDFP-NOT:#define _LP64
// ARMEABIHARDFP:#define __APCS_32__ 1
// ARMEABIHARDFP-NOT:#define __ARMEB__ 1
// ARMEABIHARDFP:#define __ARMEL__ 1
// ARMEABIHARDFP:#define __ARM_ARCH 4
// ARMEABIHARDFP:#define __ARM_ARCH_4T__ 1
// ARMEABIHARDFP-NOT:#define __ARM_BIG_ENDIAN 1
// ARMEABIHARDFP:#define __ARM_EABI__ 1
// ARMEABIHARDFP:#define __ARM_PCS 1
// ARMEABIHARDFP:#define __ARM_PCS_VFP 1
// ARMEABIHARDFP:#define __BIGGEST_ALIGNMENT__ 8
// ARMEABIHARDFP:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// ARMEABIHARDFP:#define __CHAR16_TYPE__ unsigned short
// ARMEABIHARDFP:#define __CHAR32_TYPE__ unsigned int
// ARMEABIHARDFP:#define __CHAR_BIT__ 8
// ARMEABIHARDFP:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// ARMEABIHARDFP:#define __DBL_DIG__ 15
// ARMEABIHARDFP:#define __DBL_EPSILON__ 2.2204460492503131e-16
// ARMEABIHARDFP:#define __DBL_HAS_DENORM__ 1
// ARMEABIHARDFP:#define __DBL_HAS_INFINITY__ 1
// ARMEABIHARDFP:#define __DBL_HAS_QUIET_NAN__ 1
// ARMEABIHARDFP:#define __DBL_MANT_DIG__ 53
// ARMEABIHARDFP:#define __DBL_MAX_10_EXP__ 308
// ARMEABIHARDFP:#define __DBL_MAX_EXP__ 1024
// ARMEABIHARDFP:#define __DBL_MAX__ 1.7976931348623157e+308
// ARMEABIHARDFP:#define __DBL_MIN_10_EXP__ (-307)
// ARMEABIHARDFP:#define __DBL_MIN_EXP__ (-1021)
// ARMEABIHARDFP:#define __DBL_MIN__ 2.2250738585072014e-308
// ARMEABIHARDFP:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// ARMEABIHARDFP:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// ARMEABIHARDFP:#define __FLT_DIG__ 6
// ARMEABIHARDFP:#define __FLT_EPSILON__ 1.19209290e-7F
// ARMEABIHARDFP:#define __FLT_EVAL_METHOD__ 0
// ARMEABIHARDFP:#define __FLT_HAS_DENORM__ 1
// ARMEABIHARDFP:#define __FLT_HAS_INFINITY__ 1
// ARMEABIHARDFP:#define __FLT_HAS_QUIET_NAN__ 1
// ARMEABIHARDFP:#define __FLT_MANT_DIG__ 24
// ARMEABIHARDFP:#define __FLT_MAX_10_EXP__ 38
// ARMEABIHARDFP:#define __FLT_MAX_EXP__ 128
// ARMEABIHARDFP:#define __FLT_MAX__ 3.40282347e+38F
// ARMEABIHARDFP:#define __FLT_MIN_10_EXP__ (-37)
// ARMEABIHARDFP:#define __FLT_MIN_EXP__ (-125)
// ARMEABIHARDFP:#define __FLT_MIN__ 1.17549435e-38F
// ARMEABIHARDFP:#define __FLT_RADIX__ 2
// ARMEABIHARDFP:#define __INT16_C_SUFFIX__
// ARMEABIHARDFP:#define __INT16_FMTd__ "hd"
// ARMEABIHARDFP:#define __INT16_FMTi__ "hi"
// ARMEABIHARDFP:#define __INT16_MAX__ 32767
// ARMEABIHARDFP:#define __INT16_TYPE__ short
// ARMEABIHARDFP:#define __INT32_C_SUFFIX__
// ARMEABIHARDFP:#define __INT32_FMTd__ "d"
// ARMEABIHARDFP:#define __INT32_FMTi__ "i"
// ARMEABIHARDFP:#define __INT32_MAX__ 2147483647
// ARMEABIHARDFP:#define __INT32_TYPE__ int
// ARMEABIHARDFP:#define __INT64_C_SUFFIX__ LL
// ARMEABIHARDFP:#define __INT64_FMTd__ "lld"
// ARMEABIHARDFP:#define __INT64_FMTi__ "lli"
// ARMEABIHARDFP:#define __INT64_MAX__ 9223372036854775807LL
// ARMEABIHARDFP:#define __INT64_TYPE__ long long int
// ARMEABIHARDFP:#define __INT8_C_SUFFIX__
// ARMEABIHARDFP:#define __INT8_FMTd__ "hhd"
// ARMEABIHARDFP:#define __INT8_FMTi__ "hhi"
// ARMEABIHARDFP:#define __INT8_MAX__ 127
// ARMEABIHARDFP:#define __INT8_TYPE__ signed char
// ARMEABIHARDFP:#define __INTMAX_C_SUFFIX__ LL
// ARMEABIHARDFP:#define __INTMAX_FMTd__ "lld"
// ARMEABIHARDFP:#define __INTMAX_FMTi__ "lli"
// ARMEABIHARDFP:#define __INTMAX_MAX__ 9223372036854775807LL
// ARMEABIHARDFP:#define __INTMAX_TYPE__ long long int
// ARMEABIHARDFP:#define __INTMAX_WIDTH__ 64
// ARMEABIHARDFP:#define __INTPTR_FMTd__ "ld"
// ARMEABIHARDFP:#define __INTPTR_FMTi__ "li"
// ARMEABIHARDFP:#define __INTPTR_MAX__ 2147483647L
// ARMEABIHARDFP:#define __INTPTR_TYPE__ long int
// ARMEABIHARDFP:#define __INTPTR_WIDTH__ 32
// ARMEABIHARDFP:#define __INT_FAST16_FMTd__ "hd"
// ARMEABIHARDFP:#define __INT_FAST16_FMTi__ "hi"
// ARMEABIHARDFP:#define __INT_FAST16_MAX__ 32767
// ARMEABIHARDFP:#define __INT_FAST16_TYPE__ short
// ARMEABIHARDFP:#define __INT_FAST32_FMTd__ "d"
// ARMEABIHARDFP:#define __INT_FAST32_FMTi__ "i"
// ARMEABIHARDFP:#define __INT_FAST32_MAX__ 2147483647
// ARMEABIHARDFP:#define __INT_FAST32_TYPE__ int
// ARMEABIHARDFP:#define __INT_FAST64_FMTd__ "lld"
// ARMEABIHARDFP:#define __INT_FAST64_FMTi__ "lli"
// ARMEABIHARDFP:#define __INT_FAST64_MAX__ 9223372036854775807LL
// ARMEABIHARDFP:#define __INT_FAST64_TYPE__ long long int
// ARMEABIHARDFP:#define __INT_FAST8_FMTd__ "hhd"
// ARMEABIHARDFP:#define __INT_FAST8_FMTi__ "hhi"
// ARMEABIHARDFP:#define __INT_FAST8_MAX__ 127
// ARMEABIHARDFP:#define __INT_FAST8_TYPE__ signed char
// ARMEABIHARDFP:#define __INT_LEAST16_FMTd__ "hd"
// ARMEABIHARDFP:#define __INT_LEAST16_FMTi__ "hi"
// ARMEABIHARDFP:#define __INT_LEAST16_MAX__ 32767
// ARMEABIHARDFP:#define __INT_LEAST16_TYPE__ short
// ARMEABIHARDFP:#define __INT_LEAST32_FMTd__ "d"
// ARMEABIHARDFP:#define __INT_LEAST32_FMTi__ "i"
// ARMEABIHARDFP:#define __INT_LEAST32_MAX__ 2147483647
// ARMEABIHARDFP:#define __INT_LEAST32_TYPE__ int
// ARMEABIHARDFP:#define __INT_LEAST64_FMTd__ "lld"
// ARMEABIHARDFP:#define __INT_LEAST64_FMTi__ "lli"
// ARMEABIHARDFP:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// ARMEABIHARDFP:#define __INT_LEAST64_TYPE__ long long int
// ARMEABIHARDFP:#define __INT_LEAST8_FMTd__ "hhd"
// ARMEABIHARDFP:#define __INT_LEAST8_FMTi__ "hhi"
// ARMEABIHARDFP:#define __INT_LEAST8_MAX__ 127
// ARMEABIHARDFP:#define __INT_LEAST8_TYPE__ signed char
// ARMEABIHARDFP:#define __INT_MAX__ 2147483647
// ARMEABIHARDFP:#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324L
// ARMEABIHARDFP:#define __LDBL_DIG__ 15
// ARMEABIHARDFP:#define __LDBL_EPSILON__ 2.2204460492503131e-16L
// ARMEABIHARDFP:#define __LDBL_HAS_DENORM__ 1
// ARMEABIHARDFP:#define __LDBL_HAS_INFINITY__ 1
// ARMEABIHARDFP:#define __LDBL_HAS_QUIET_NAN__ 1
// ARMEABIHARDFP:#define __LDBL_MANT_DIG__ 53
// ARMEABIHARDFP:#define __LDBL_MAX_10_EXP__ 308
// ARMEABIHARDFP:#define __LDBL_MAX_EXP__ 1024
// ARMEABIHARDFP:#define __LDBL_MAX__ 1.7976931348623157e+308L
// ARMEABIHARDFP:#define __LDBL_MIN_10_EXP__ (-307)
// ARMEABIHARDFP:#define __LDBL_MIN_EXP__ (-1021)
// ARMEABIHARDFP:#define __LDBL_MIN__ 2.2250738585072014e-308L
// ARMEABIHARDFP:#define __LITTLE_ENDIAN__ 1
// ARMEABIHARDFP:#define __LONG_LONG_MAX__ 9223372036854775807LL
// ARMEABIHARDFP:#define __LONG_MAX__ 2147483647L
// ARMEABIHARDFP-NOT:#define __LP64__
// ARMEABIHARDFP:#define __POINTER_WIDTH__ 32
// ARMEABIHARDFP:#define __PTRDIFF_TYPE__ int
// ARMEABIHARDFP:#define __PTRDIFF_WIDTH__ 32
// ARMEABIHARDFP:#define __REGISTER_PREFIX__
// ARMEABIHARDFP:#define __SCHAR_MAX__ 127
// ARMEABIHARDFP:#define __SHRT_MAX__ 32767
// ARMEABIHARDFP:#define __SIG_ATOMIC_MAX__ 2147483647
// ARMEABIHARDFP:#define __SIG_ATOMIC_WIDTH__ 32
// ARMEABIHARDFP:#define __SIZEOF_DOUBLE__ 8
// ARMEABIHARDFP:#define __SIZEOF_FLOAT__ 4
// ARMEABIHARDFP:#define __SIZEOF_INT__ 4
// ARMEABIHARDFP:#define __SIZEOF_LONG_DOUBLE__ 8
// ARMEABIHARDFP:#define __SIZEOF_LONG_LONG__ 8
// ARMEABIHARDFP:#define __SIZEOF_LONG__ 4
// ARMEABIHARDFP:#define __SIZEOF_POINTER__ 4
// ARMEABIHARDFP:#define __SIZEOF_PTRDIFF_T__ 4
// ARMEABIHARDFP:#define __SIZEOF_SHORT__ 2
// ARMEABIHARDFP:#define __SIZEOF_SIZE_T__ 4
// ARMEABIHARDFP:#define __SIZEOF_WCHAR_T__ 4
// ARMEABIHARDFP:#define __SIZEOF_WINT_T__ 4
// ARMEABIHARDFP:#define __SIZE_MAX__ 4294967295U
// ARMEABIHARDFP:#define __SIZE_TYPE__ unsigned int
// ARMEABIHARDFP:#define __SIZE_WIDTH__ 32
// ARMEABIHARDFP-NOT:#define __SOFTFP__ 1
// ARMEABIHARDFP:#define __UINT16_C_SUFFIX__
// ARMEABIHARDFP:#define __UINT16_MAX__ 65535
// ARMEABIHARDFP:#define __UINT16_TYPE__ unsigned short
// ARMEABIHARDFP:#define __UINT32_C_SUFFIX__ U
// ARMEABIHARDFP:#define __UINT32_MAX__ 4294967295U
// ARMEABIHARDFP:#define __UINT32_TYPE__ unsigned int
// ARMEABIHARDFP:#define __UINT64_C_SUFFIX__ ULL
// ARMEABIHARDFP:#define __UINT64_MAX__ 18446744073709551615ULL
// ARMEABIHARDFP:#define __UINT64_TYPE__ long long unsigned int
// ARMEABIHARDFP:#define __UINT8_C_SUFFIX__
// ARMEABIHARDFP:#define __UINT8_MAX__ 255
// ARMEABIHARDFP:#define __UINT8_TYPE__ unsigned char
// ARMEABIHARDFP:#define __UINTMAX_C_SUFFIX__ ULL
// ARMEABIHARDFP:#define __UINTMAX_MAX__ 18446744073709551615ULL
// ARMEABIHARDFP:#define __UINTMAX_TYPE__ long long unsigned int
// ARMEABIHARDFP:#define __UINTMAX_WIDTH__ 64
// ARMEABIHARDFP:#define __UINTPTR_MAX__ 4294967295UL
// ARMEABIHARDFP:#define __UINTPTR_TYPE__ long unsigned int
// ARMEABIHARDFP:#define __UINTPTR_WIDTH__ 32
// ARMEABIHARDFP:#define __UINT_FAST16_MAX__ 65535
// ARMEABIHARDFP:#define __UINT_FAST16_TYPE__ unsigned short
// ARMEABIHARDFP:#define __UINT_FAST32_MAX__ 4294967295U
// ARMEABIHARDFP:#define __UINT_FAST32_TYPE__ unsigned int
// ARMEABIHARDFP:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// ARMEABIHARDFP:#define __UINT_FAST64_TYPE__ long long unsigned int
// ARMEABIHARDFP:#define __UINT_FAST8_MAX__ 255
// ARMEABIHARDFP:#define __UINT_FAST8_TYPE__ unsigned char
// ARMEABIHARDFP:#define __UINT_LEAST16_MAX__ 65535
// ARMEABIHARDFP:#define __UINT_LEAST16_TYPE__ unsigned short
// ARMEABIHARDFP:#define __UINT_LEAST32_MAX__ 4294967295U
// ARMEABIHARDFP:#define __UINT_LEAST32_TYPE__ unsigned int
// ARMEABIHARDFP:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// ARMEABIHARDFP:#define __UINT_LEAST64_TYPE__ long long unsigned int
// ARMEABIHARDFP:#define __UINT_LEAST8_MAX__ 255
// ARMEABIHARDFP:#define __UINT_LEAST8_TYPE__ unsigned char
// ARMEABIHARDFP:#define __USER_LABEL_PREFIX__
// ARMEABIHARDFP:#define __WCHAR_MAX__ 4294967295U
// ARMEABIHARDFP:#define __WCHAR_TYPE__ unsigned int
// ARMEABIHARDFP:#define __WCHAR_WIDTH__ 32
// ARMEABIHARDFP:#define __WINT_TYPE__ unsigned int
// ARMEABIHARDFP:#define __WINT_WIDTH__ 32
// ARMEABIHARDFP:#define __arm 1
// ARMEABIHARDFP:#define __arm__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=armv6-unknown-cloudabi-eabihf < /dev/null | FileCheck -match-full-lines -check-prefix ARMV6-CLOUDABI %s
//
// ARMV6-CLOUDABI:#define __CloudABI__ 1
// ARMV6-CLOUDABI:#define __arm__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=arm-netbsd-eabi < /dev/null | FileCheck -match-full-lines -check-prefix ARM-NETBSD %s
//
// ARM-NETBSD-NOT:#define _LP64
// ARM-NETBSD:#define __APCS_32__ 1
// ARM-NETBSD-NOT:#define __ARMEB__ 1
// ARM-NETBSD:#define __ARMEL__ 1
// ARM-NETBSD:#define __ARM_ARCH_4T__ 1
// ARM-NETBSD:#define __ARM_DWARF_EH__ 1
// ARM-NETBSD:#define __ARM_EABI__ 1
// ARM-NETBSD-NOT:#define __ARM_BIG_ENDIAN 1
// ARM-NETBSD:#define __BIGGEST_ALIGNMENT__ 8
// ARM-NETBSD:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// ARM-NETBSD:#define __CHAR16_TYPE__ unsigned short
// ARM-NETBSD:#define __CHAR32_TYPE__ unsigned int
// ARM-NETBSD:#define __CHAR_BIT__ 8
// ARM-NETBSD:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// ARM-NETBSD:#define __DBL_DIG__ 15
// ARM-NETBSD:#define __DBL_EPSILON__ 2.2204460492503131e-16
// ARM-NETBSD:#define __DBL_HAS_DENORM__ 1
// ARM-NETBSD:#define __DBL_HAS_INFINITY__ 1
// ARM-NETBSD:#define __DBL_HAS_QUIET_NAN__ 1
// ARM-NETBSD:#define __DBL_MANT_DIG__ 53
// ARM-NETBSD:#define __DBL_MAX_10_EXP__ 308
// ARM-NETBSD:#define __DBL_MAX_EXP__ 1024
// ARM-NETBSD:#define __DBL_MAX__ 1.7976931348623157e+308
// ARM-NETBSD:#define __DBL_MIN_10_EXP__ (-307)
// ARM-NETBSD:#define __DBL_MIN_EXP__ (-1021)
// ARM-NETBSD:#define __DBL_MIN__ 2.2250738585072014e-308
// ARM-NETBSD:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// ARM-NETBSD:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// ARM-NETBSD:#define __FLT_DIG__ 6
// ARM-NETBSD:#define __FLT_EPSILON__ 1.19209290e-7F
// ARM-NETBSD:#define __FLT_EVAL_METHOD__ 0
// ARM-NETBSD:#define __FLT_HAS_DENORM__ 1
// ARM-NETBSD:#define __FLT_HAS_INFINITY__ 1
// ARM-NETBSD:#define __FLT_HAS_QUIET_NAN__ 1
// ARM-NETBSD:#define __FLT_MANT_DIG__ 24
// ARM-NETBSD:#define __FLT_MAX_10_EXP__ 38
// ARM-NETBSD:#define __FLT_MAX_EXP__ 128
// ARM-NETBSD:#define __FLT_MAX__ 3.40282347e+38F
// ARM-NETBSD:#define __FLT_MIN_10_EXP__ (-37)
// ARM-NETBSD:#define __FLT_MIN_EXP__ (-125)
// ARM-NETBSD:#define __FLT_MIN__ 1.17549435e-38F
// ARM-NETBSD:#define __FLT_RADIX__ 2
// ARM-NETBSD:#define __INT16_C_SUFFIX__
// ARM-NETBSD:#define __INT16_FMTd__ "hd"
// ARM-NETBSD:#define __INT16_FMTi__ "hi"
// ARM-NETBSD:#define __INT16_MAX__ 32767
// ARM-NETBSD:#define __INT16_TYPE__ short
// ARM-NETBSD:#define __INT32_C_SUFFIX__
// ARM-NETBSD:#define __INT32_FMTd__ "d"
// ARM-NETBSD:#define __INT32_FMTi__ "i"
// ARM-NETBSD:#define __INT32_MAX__ 2147483647
// ARM-NETBSD:#define __INT32_TYPE__ int
// ARM-NETBSD:#define __INT64_C_SUFFIX__ LL
// ARM-NETBSD:#define __INT64_FMTd__ "lld"
// ARM-NETBSD:#define __INT64_FMTi__ "lli"
// ARM-NETBSD:#define __INT64_MAX__ 9223372036854775807LL
// ARM-NETBSD:#define __INT64_TYPE__ long long int
// ARM-NETBSD:#define __INT8_C_SUFFIX__
// ARM-NETBSD:#define __INT8_FMTd__ "hhd"
// ARM-NETBSD:#define __INT8_FMTi__ "hhi"
// ARM-NETBSD:#define __INT8_MAX__ 127
// ARM-NETBSD:#define __INT8_TYPE__ signed char
// ARM-NETBSD:#define __INTMAX_C_SUFFIX__ LL
// ARM-NETBSD:#define __INTMAX_FMTd__ "lld"
// ARM-NETBSD:#define __INTMAX_FMTi__ "lli"
// ARM-NETBSD:#define __INTMAX_MAX__ 9223372036854775807LL
// ARM-NETBSD:#define __INTMAX_TYPE__ long long int
// ARM-NETBSD:#define __INTMAX_WIDTH__ 64
// ARM-NETBSD:#define __INTPTR_FMTd__ "ld"
// ARM-NETBSD:#define __INTPTR_FMTi__ "li"
// ARM-NETBSD:#define __INTPTR_MAX__ 2147483647L
// ARM-NETBSD:#define __INTPTR_TYPE__ long int
// ARM-NETBSD:#define __INTPTR_WIDTH__ 32
// ARM-NETBSD:#define __INT_FAST16_FMTd__ "hd"
// ARM-NETBSD:#define __INT_FAST16_FMTi__ "hi"
// ARM-NETBSD:#define __INT_FAST16_MAX__ 32767
// ARM-NETBSD:#define __INT_FAST16_TYPE__ short
// ARM-NETBSD:#define __INT_FAST32_FMTd__ "d"
// ARM-NETBSD:#define __INT_FAST32_FMTi__ "i"
// ARM-NETBSD:#define __INT_FAST32_MAX__ 2147483647
// ARM-NETBSD:#define __INT_FAST32_TYPE__ int
// ARM-NETBSD:#define __INT_FAST64_FMTd__ "lld"
// ARM-NETBSD:#define __INT_FAST64_FMTi__ "lli"
// ARM-NETBSD:#define __INT_FAST64_MAX__ 9223372036854775807LL
// ARM-NETBSD:#define __INT_FAST64_TYPE__ long long int
// ARM-NETBSD:#define __INT_FAST8_FMTd__ "hhd"
// ARM-NETBSD:#define __INT_FAST8_FMTi__ "hhi"
// ARM-NETBSD:#define __INT_FAST8_MAX__ 127
// ARM-NETBSD:#define __INT_FAST8_TYPE__ signed char
// ARM-NETBSD:#define __INT_LEAST16_FMTd__ "hd"
// ARM-NETBSD:#define __INT_LEAST16_FMTi__ "hi"
// ARM-NETBSD:#define __INT_LEAST16_MAX__ 32767
// ARM-NETBSD:#define __INT_LEAST16_TYPE__ short
// ARM-NETBSD:#define __INT_LEAST32_FMTd__ "d"
// ARM-NETBSD:#define __INT_LEAST32_FMTi__ "i"
// ARM-NETBSD:#define __INT_LEAST32_MAX__ 2147483647
// ARM-NETBSD:#define __INT_LEAST32_TYPE__ int
// ARM-NETBSD:#define __INT_LEAST64_FMTd__ "lld"
// ARM-NETBSD:#define __INT_LEAST64_FMTi__ "lli"
// ARM-NETBSD:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// ARM-NETBSD:#define __INT_LEAST64_TYPE__ long long int
// ARM-NETBSD:#define __INT_LEAST8_FMTd__ "hhd"
// ARM-NETBSD:#define __INT_LEAST8_FMTi__ "hhi"
// ARM-NETBSD:#define __INT_LEAST8_MAX__ 127
// ARM-NETBSD:#define __INT_LEAST8_TYPE__ signed char
// ARM-NETBSD:#define __INT_MAX__ 2147483647
// ARM-NETBSD:#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324L
// ARM-NETBSD:#define __LDBL_DIG__ 15
// ARM-NETBSD:#define __LDBL_EPSILON__ 2.2204460492503131e-16L
// ARM-NETBSD:#define __LDBL_HAS_DENORM__ 1
// ARM-NETBSD:#define __LDBL_HAS_INFINITY__ 1
// ARM-NETBSD:#define __LDBL_HAS_QUIET_NAN__ 1
// ARM-NETBSD:#define __LDBL_MANT_DIG__ 53
// ARM-NETBSD:#define __LDBL_MAX_10_EXP__ 308
// ARM-NETBSD:#define __LDBL_MAX_EXP__ 1024
// ARM-NETBSD:#define __LDBL_MAX__ 1.7976931348623157e+308L
// ARM-NETBSD:#define __LDBL_MIN_10_EXP__ (-307)
// ARM-NETBSD:#define __LDBL_MIN_EXP__ (-1021)
// ARM-NETBSD:#define __LDBL_MIN__ 2.2250738585072014e-308L
// ARM-NETBSD:#define __LITTLE_ENDIAN__ 1
// ARM-NETBSD:#define __LONG_LONG_MAX__ 9223372036854775807LL
// ARM-NETBSD:#define __LONG_MAX__ 2147483647L
// ARM-NETBSD-NOT:#define __LP64__
// ARM-NETBSD:#define __POINTER_WIDTH__ 32
// ARM-NETBSD:#define __PTRDIFF_TYPE__ long int
// ARM-NETBSD:#define __PTRDIFF_WIDTH__ 32
// ARM-NETBSD:#define __REGISTER_PREFIX__
// ARM-NETBSD:#define __SCHAR_MAX__ 127
// ARM-NETBSD:#define __SHRT_MAX__ 32767
// ARM-NETBSD:#define __SIG_ATOMIC_MAX__ 2147483647
// ARM-NETBSD:#define __SIG_ATOMIC_WIDTH__ 32
// ARM-NETBSD:#define __SIZEOF_DOUBLE__ 8
// ARM-NETBSD:#define __SIZEOF_FLOAT__ 4
// ARM-NETBSD:#define __SIZEOF_INT__ 4
// ARM-NETBSD:#define __SIZEOF_LONG_DOUBLE__ 8
// ARM-NETBSD:#define __SIZEOF_LONG_LONG__ 8
// ARM-NETBSD:#define __SIZEOF_LONG__ 4
// ARM-NETBSD:#define __SIZEOF_POINTER__ 4
// ARM-NETBSD:#define __SIZEOF_PTRDIFF_T__ 4
// ARM-NETBSD:#define __SIZEOF_SHORT__ 2
// ARM-NETBSD:#define __SIZEOF_SIZE_T__ 4
// ARM-NETBSD:#define __SIZEOF_WCHAR_T__ 4
// ARM-NETBSD:#define __SIZEOF_WINT_T__ 4
// ARM-NETBSD:#define __SIZE_MAX__ 4294967295UL
// ARM-NETBSD:#define __SIZE_TYPE__ long unsigned int
// ARM-NETBSD:#define __SIZE_WIDTH__ 32
// ARM-NETBSD:#define __UINT16_C_SUFFIX__
// ARM-NETBSD:#define __UINT16_MAX__ 65535
// ARM-NETBSD:#define __UINT16_TYPE__ unsigned short
// ARM-NETBSD:#define __UINT32_C_SUFFIX__ U
// ARM-NETBSD:#define __UINT32_MAX__ 4294967295U
// ARM-NETBSD:#define __UINT32_TYPE__ unsigned int
// ARM-NETBSD:#define __UINT64_C_SUFFIX__ ULL
// ARM-NETBSD:#define __UINT64_MAX__ 18446744073709551615ULL
// ARM-NETBSD:#define __UINT64_TYPE__ long long unsigned int
// ARM-NETBSD:#define __UINT8_C_SUFFIX__
// ARM-NETBSD:#define __UINT8_MAX__ 255
// ARM-NETBSD:#define __UINT8_TYPE__ unsigned char
// ARM-NETBSD:#define __UINTMAX_C_SUFFIX__ ULL
// ARM-NETBSD:#define __UINTMAX_MAX__ 18446744073709551615ULL
// ARM-NETBSD:#define __UINTMAX_TYPE__ long long unsigned int
// ARM-NETBSD:#define __UINTMAX_WIDTH__ 64
// ARM-NETBSD:#define __UINTPTR_MAX__ 4294967295UL
// ARM-NETBSD:#define __UINTPTR_TYPE__ long unsigned int
// ARM-NETBSD:#define __UINTPTR_WIDTH__ 32
// ARM-NETBSD:#define __UINT_FAST16_MAX__ 65535
// ARM-NETBSD:#define __UINT_FAST16_TYPE__ unsigned short
// ARM-NETBSD:#define __UINT_FAST32_MAX__ 4294967295U
// ARM-NETBSD:#define __UINT_FAST32_TYPE__ unsigned int
// ARM-NETBSD:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// ARM-NETBSD:#define __UINT_FAST64_TYPE__ long long unsigned int
// ARM-NETBSD:#define __UINT_FAST8_MAX__ 255
// ARM-NETBSD:#define __UINT_FAST8_TYPE__ unsigned char
// ARM-NETBSD:#define __UINT_LEAST16_MAX__ 65535
// ARM-NETBSD:#define __UINT_LEAST16_TYPE__ unsigned short
// ARM-NETBSD:#define __UINT_LEAST32_MAX__ 4294967295U
// ARM-NETBSD:#define __UINT_LEAST32_TYPE__ unsigned int
// ARM-NETBSD:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// ARM-NETBSD:#define __UINT_LEAST64_TYPE__ long long unsigned int
// ARM-NETBSD:#define __UINT_LEAST8_MAX__ 255
// ARM-NETBSD:#define __UINT_LEAST8_TYPE__ unsigned char
// ARM-NETBSD:#define __USER_LABEL_PREFIX__
// ARM-NETBSD:#define __WCHAR_MAX__ 2147483647
// ARM-NETBSD:#define __WCHAR_TYPE__ int
// ARM-NETBSD:#define __WCHAR_WIDTH__ 32
// ARM-NETBSD:#define __WINT_TYPE__ int
// ARM-NETBSD:#define __WINT_WIDTH__ 32
// ARM-NETBSD:#define __arm 1
// ARM-NETBSD:#define __arm__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=arm-none-eabi < /dev/null | FileCheck -match-full-lines -check-prefix ARM-NONE-EABI %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=arm-none-eabihf < /dev/null | FileCheck -match-full-lines -check-prefix ARM-NONE-EABI %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=aarch64-none-eabi < /dev/null | FileCheck -match-full-lines -check-prefix ARM-NONE-EABI %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=aarch64-none-eabihf < /dev/null | FileCheck -match-full-lines -check-prefix ARM-NONE-EABI %s
// ARM-NONE-EABI: #define __ELF__ 1

// No MachO targets use the full EABI, even if AAPCS is used.
// RUN: %clang -target x86_64-apple-darwin -arch armv7s -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARM-MACHO-NO-EABI %s
// RUN: %clang -target x86_64-apple-darwin -arch armv6m -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARM-MACHO-NO-EABI %s
// RUN: %clang -target x86_64-apple-darwin -arch armv7m -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARM-MACHO-NO-EABI %s
// RUN: %clang -target x86_64-apple-darwin -arch armv7em -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARM-MACHO-NO-EABI %s
// RUN: %clang -target x86_64-apple-darwin -arch armv7 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARM-MACHO-NO-EABI %s
// ARM-MACHO-NO-EABI-NOT: #define __ARM_EABI__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=armv7-bitrig-gnueabihf < /dev/null | FileCheck -match-full-lines -check-prefix ARM-BITRIG %s
// ARM-BITRIG:#define __ARM_DWARF_EH__ 1
// ARM-BITRIG:#define __SIZEOF_SIZE_T__ 4
// ARM-BITRIG:#define __SIZE_MAX__ 4294967295UL
// ARM-BITRIG:#define __SIZE_TYPE__ long unsigned int
// ARM-BITRIG:#define __SIZE_WIDTH__ 32

// Check that -mhwdiv works properly for targets which don't have the hwdiv feature enabled by default.

// RUN: %clang -target arm -mhwdiv=arm -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMHWDIV-ARM %s
// ARMHWDIV-ARM:#define __ARM_ARCH_EXT_IDIV__ 1

// RUN: %clang -target arm -mthumb -mhwdiv=thumb -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=THUMBHWDIV-THUMB %s
// THUMBHWDIV-THUMB:#define __ARM_ARCH_EXT_IDIV__ 1

// RUN: %clang -target arm -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARM-FALSE %s
// ARM-FALSE-NOT:#define __ARM_ARCH_EXT_IDIV__

// RUN: %clang -target arm -mthumb -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=THUMB-FALSE %s
// THUMB-FALSE-NOT:#define __ARM_ARCH_EXT_IDIV__

// RUN: %clang -target arm -mhwdiv=thumb -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=THUMBHWDIV-ARM-FALSE %s
// THUMBHWDIV-ARM-FALSE-NOT:#define __ARM_ARCH_EXT_IDIV__

// RUN: %clang -target arm -mthumb -mhwdiv=arm -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMHWDIV-THUMB-FALSE %s
// ARMHWDIV-THUMB-FALSE-NOT:#define __ARM_ARCH_EXT_IDIV__

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=armv8-none-none < /dev/null | FileCheck -match-full-lines -check-prefix ARMv8 %s
// ARMv8: #define __THUMB_INTERWORK__ 1
// ARMv8-NOT: #define __thumb2__

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=armebv8-none-none < /dev/null | FileCheck -match-full-lines -check-prefix ARMebv8 %s
// ARMebv8: #define __THUMB_INTERWORK__ 1
// ARMebv8-NOT: #define __thumb2__

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=thumbv8 < /dev/null | FileCheck -match-full-lines -check-prefix Thumbv8 %s
// Thumbv8: #define __THUMB_INTERWORK__ 1
// Thumbv8: #define __thumb2__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=thumbebv8 < /dev/null | FileCheck -match-full-lines -check-prefix Thumbebv8 %s
// Thumbebv8: #define __THUMB_INTERWORK__ 1
// Thumbebv8: #define __thumb2__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=thumbv5 < /dev/null | FileCheck -match-full-lines -check-prefix Thumbv5 %s
// Thumbv5: #define __THUMB_INTERWORK__ 1
// Thumbv5-NOT: #define __thumb2__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=thumbv6t2 < /dev/null | FileCheck -match-full-lines -check-prefix Thumbv6t2 %s
// Thumbv6t2: #define __THUMB_INTERWORK__ 1
// Thumbv6t2: #define __thumb2__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=thumbv7 < /dev/null | FileCheck -match-full-lines -check-prefix Thumbv7 %s
// Thumbv7: #define __THUMB_INTERWORK__ 1
// Thumbv7: #define __thumb2__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=thumbebv7 < /dev/null | FileCheck -match-full-lines -check-prefix Thumbebv7 %s
// Thumbebv7: #define __THUMB_INTERWORK__ 1
// Thumbebv7: #define __thumb2__ 1

//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=i386-none-none < /dev/null | FileCheck -match-full-lines -check-prefix I386 %s
//
// I386-NOT:#define _LP64
// I386:#define __BIGGEST_ALIGNMENT__ 16
// I386:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// I386:#define __CHAR16_TYPE__ unsigned short
// I386:#define __CHAR32_TYPE__ unsigned int
// I386:#define __CHAR_BIT__ 8
// I386:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// I386:#define __DBL_DIG__ 15
// I386:#define __DBL_EPSILON__ 2.2204460492503131e-16
// I386:#define __DBL_HAS_DENORM__ 1
// I386:#define __DBL_HAS_INFINITY__ 1
// I386:#define __DBL_HAS_QUIET_NAN__ 1
// I386:#define __DBL_MANT_DIG__ 53
// I386:#define __DBL_MAX_10_EXP__ 308
// I386:#define __DBL_MAX_EXP__ 1024
// I386:#define __DBL_MAX__ 1.7976931348623157e+308
// I386:#define __DBL_MIN_10_EXP__ (-307)
// I386:#define __DBL_MIN_EXP__ (-1021)
// I386:#define __DBL_MIN__ 2.2250738585072014e-308
// I386:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// I386:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// I386:#define __FLT_DIG__ 6
// I386:#define __FLT_EPSILON__ 1.19209290e-7F
// I386:#define __FLT_EVAL_METHOD__ 2
// I386:#define __FLT_HAS_DENORM__ 1
// I386:#define __FLT_HAS_INFINITY__ 1
// I386:#define __FLT_HAS_QUIET_NAN__ 1
// I386:#define __FLT_MANT_DIG__ 24
// I386:#define __FLT_MAX_10_EXP__ 38
// I386:#define __FLT_MAX_EXP__ 128
// I386:#define __FLT_MAX__ 3.40282347e+38F
// I386:#define __FLT_MIN_10_EXP__ (-37)
// I386:#define __FLT_MIN_EXP__ (-125)
// I386:#define __FLT_MIN__ 1.17549435e-38F
// I386:#define __FLT_RADIX__ 2
// I386:#define __INT16_C_SUFFIX__
// I386:#define __INT16_FMTd__ "hd"
// I386:#define __INT16_FMTi__ "hi"
// I386:#define __INT16_MAX__ 32767
// I386:#define __INT16_TYPE__ short
// I386:#define __INT32_C_SUFFIX__
// I386:#define __INT32_FMTd__ "d"
// I386:#define __INT32_FMTi__ "i"
// I386:#define __INT32_MAX__ 2147483647
// I386:#define __INT32_TYPE__ int
// I386:#define __INT64_C_SUFFIX__ LL
// I386:#define __INT64_FMTd__ "lld"
// I386:#define __INT64_FMTi__ "lli"
// I386:#define __INT64_MAX__ 9223372036854775807LL
// I386:#define __INT64_TYPE__ long long int
// I386:#define __INT8_C_SUFFIX__
// I386:#define __INT8_FMTd__ "hhd"
// I386:#define __INT8_FMTi__ "hhi"
// I386:#define __INT8_MAX__ 127
// I386:#define __INT8_TYPE__ signed char
// I386:#define __INTMAX_C_SUFFIX__ LL
// I386:#define __INTMAX_FMTd__ "lld"
// I386:#define __INTMAX_FMTi__ "lli"
// I386:#define __INTMAX_MAX__ 9223372036854775807LL
// I386:#define __INTMAX_TYPE__ long long int
// I386:#define __INTMAX_WIDTH__ 64
// I386:#define __INTPTR_FMTd__ "d"
// I386:#define __INTPTR_FMTi__ "i"
// I386:#define __INTPTR_MAX__ 2147483647
// I386:#define __INTPTR_TYPE__ int
// I386:#define __INTPTR_WIDTH__ 32
// I386:#define __INT_FAST16_FMTd__ "hd"
// I386:#define __INT_FAST16_FMTi__ "hi"
// I386:#define __INT_FAST16_MAX__ 32767
// I386:#define __INT_FAST16_TYPE__ short
// I386:#define __INT_FAST32_FMTd__ "d"
// I386:#define __INT_FAST32_FMTi__ "i"
// I386:#define __INT_FAST32_MAX__ 2147483647
// I386:#define __INT_FAST32_TYPE__ int
// I386:#define __INT_FAST64_FMTd__ "lld"
// I386:#define __INT_FAST64_FMTi__ "lli"
// I386:#define __INT_FAST64_MAX__ 9223372036854775807LL
// I386:#define __INT_FAST64_TYPE__ long long int
// I386:#define __INT_FAST8_FMTd__ "hhd"
// I386:#define __INT_FAST8_FMTi__ "hhi"
// I386:#define __INT_FAST8_MAX__ 127
// I386:#define __INT_FAST8_TYPE__ signed char
// I386:#define __INT_LEAST16_FMTd__ "hd"
// I386:#define __INT_LEAST16_FMTi__ "hi"
// I386:#define __INT_LEAST16_MAX__ 32767
// I386:#define __INT_LEAST16_TYPE__ short
// I386:#define __INT_LEAST32_FMTd__ "d"
// I386:#define __INT_LEAST32_FMTi__ "i"
// I386:#define __INT_LEAST32_MAX__ 2147483647
// I386:#define __INT_LEAST32_TYPE__ int
// I386:#define __INT_LEAST64_FMTd__ "lld"
// I386:#define __INT_LEAST64_FMTi__ "lli"
// I386:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// I386:#define __INT_LEAST64_TYPE__ long long int
// I386:#define __INT_LEAST8_FMTd__ "hhd"
// I386:#define __INT_LEAST8_FMTi__ "hhi"
// I386:#define __INT_LEAST8_MAX__ 127
// I386:#define __INT_LEAST8_TYPE__ signed char
// I386:#define __INT_MAX__ 2147483647
// I386:#define __LDBL_DENORM_MIN__ 3.64519953188247460253e-4951L
// I386:#define __LDBL_DIG__ 18
// I386:#define __LDBL_EPSILON__ 1.08420217248550443401e-19L
// I386:#define __LDBL_HAS_DENORM__ 1
// I386:#define __LDBL_HAS_INFINITY__ 1
// I386:#define __LDBL_HAS_QUIET_NAN__ 1
// I386:#define __LDBL_MANT_DIG__ 64
// I386:#define __LDBL_MAX_10_EXP__ 4932
// I386:#define __LDBL_MAX_EXP__ 16384
// I386:#define __LDBL_MAX__ 1.18973149535723176502e+4932L
// I386:#define __LDBL_MIN_10_EXP__ (-4931)
// I386:#define __LDBL_MIN_EXP__ (-16381)
// I386:#define __LDBL_MIN__ 3.36210314311209350626e-4932L
// I386:#define __LITTLE_ENDIAN__ 1
// I386:#define __LONG_LONG_MAX__ 9223372036854775807LL
// I386:#define __LONG_MAX__ 2147483647L
// I386-NOT:#define __LP64__
// I386:#define __NO_MATH_INLINES 1
// I386:#define __POINTER_WIDTH__ 32
// I386:#define __PTRDIFF_TYPE__ int
// I386:#define __PTRDIFF_WIDTH__ 32
// I386:#define __REGISTER_PREFIX__
// I386:#define __SCHAR_MAX__ 127
// I386:#define __SHRT_MAX__ 32767
// I386:#define __SIG_ATOMIC_MAX__ 2147483647
// I386:#define __SIG_ATOMIC_WIDTH__ 32
// I386:#define __SIZEOF_DOUBLE__ 8
// I386:#define __SIZEOF_FLOAT__ 4
// I386:#define __SIZEOF_INT__ 4
// I386:#define __SIZEOF_LONG_DOUBLE__ 12
// I386:#define __SIZEOF_LONG_LONG__ 8
// I386:#define __SIZEOF_LONG__ 4
// I386:#define __SIZEOF_POINTER__ 4
// I386:#define __SIZEOF_PTRDIFF_T__ 4
// I386:#define __SIZEOF_SHORT__ 2
// I386:#define __SIZEOF_SIZE_T__ 4
// I386:#define __SIZEOF_WCHAR_T__ 4
// I386:#define __SIZEOF_WINT_T__ 4
// I386:#define __SIZE_MAX__ 4294967295U
// I386:#define __SIZE_TYPE__ unsigned int
// I386:#define __SIZE_WIDTH__ 32
// I386:#define __UINT16_C_SUFFIX__
// I386:#define __UINT16_MAX__ 65535
// I386:#define __UINT16_TYPE__ unsigned short
// I386:#define __UINT32_C_SUFFIX__ U
// I386:#define __UINT32_MAX__ 4294967295U
// I386:#define __UINT32_TYPE__ unsigned int
// I386:#define __UINT64_C_SUFFIX__ ULL
// I386:#define __UINT64_MAX__ 18446744073709551615ULL
// I386:#define __UINT64_TYPE__ long long unsigned int
// I386:#define __UINT8_C_SUFFIX__
// I386:#define __UINT8_MAX__ 255
// I386:#define __UINT8_TYPE__ unsigned char
// I386:#define __UINTMAX_C_SUFFIX__ ULL
// I386:#define __UINTMAX_MAX__ 18446744073709551615ULL
// I386:#define __UINTMAX_TYPE__ long long unsigned int
// I386:#define __UINTMAX_WIDTH__ 64
// I386:#define __UINTPTR_MAX__ 4294967295U
// I386:#define __UINTPTR_TYPE__ unsigned int
// I386:#define __UINTPTR_WIDTH__ 32
// I386:#define __UINT_FAST16_MAX__ 65535
// I386:#define __UINT_FAST16_TYPE__ unsigned short
// I386:#define __UINT_FAST32_MAX__ 4294967295U
// I386:#define __UINT_FAST32_TYPE__ unsigned int
// I386:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// I386:#define __UINT_FAST64_TYPE__ long long unsigned int
// I386:#define __UINT_FAST8_MAX__ 255
// I386:#define __UINT_FAST8_TYPE__ unsigned char
// I386:#define __UINT_LEAST16_MAX__ 65535
// I386:#define __UINT_LEAST16_TYPE__ unsigned short
// I386:#define __UINT_LEAST32_MAX__ 4294967295U
// I386:#define __UINT_LEAST32_TYPE__ unsigned int
// I386:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// I386:#define __UINT_LEAST64_TYPE__ long long unsigned int
// I386:#define __UINT_LEAST8_MAX__ 255
// I386:#define __UINT_LEAST8_TYPE__ unsigned char
// I386:#define __USER_LABEL_PREFIX__
// I386:#define __WCHAR_MAX__ 2147483647
// I386:#define __WCHAR_TYPE__ int
// I386:#define __WCHAR_WIDTH__ 32
// I386:#define __WINT_TYPE__ int
// I386:#define __WINT_WIDTH__ 32
// I386:#define __i386 1
// I386:#define __i386__ 1
// I386:#define i386 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=i386-pc-linux-gnu -target-cpu pentium4 < /dev/null | FileCheck -match-full-lines -check-prefix I386-LINUX %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=i386-pc-linux-gnu -target-cpu pentium4 < /dev/null | FileCheck -match-full-lines -check-prefix I386-LINUX -check-prefix I386-LINUX-CXX %s
//
// I386-LINUX-NOT:#define _LP64
// I386-LINUX:#define __BIGGEST_ALIGNMENT__ 16
// I386-LINUX:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// I386-LINUX:#define __CHAR16_TYPE__ unsigned short
// I386-LINUX:#define __CHAR32_TYPE__ unsigned int
// I386-LINUX:#define __CHAR_BIT__ 8
// I386-LINUX:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// I386-LINUX:#define __DBL_DIG__ 15
// I386-LINUX:#define __DBL_EPSILON__ 2.2204460492503131e-16
// I386-LINUX:#define __DBL_HAS_DENORM__ 1
// I386-LINUX:#define __DBL_HAS_INFINITY__ 1
// I386-LINUX:#define __DBL_HAS_QUIET_NAN__ 1
// I386-LINUX:#define __DBL_MANT_DIG__ 53
// I386-LINUX:#define __DBL_MAX_10_EXP__ 308
// I386-LINUX:#define __DBL_MAX_EXP__ 1024
// I386-LINUX:#define __DBL_MAX__ 1.7976931348623157e+308
// I386-LINUX:#define __DBL_MIN_10_EXP__ (-307)
// I386-LINUX:#define __DBL_MIN_EXP__ (-1021)
// I386-LINUX:#define __DBL_MIN__ 2.2250738585072014e-308
// I386-LINUX:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// I386-LINUX:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// I386-LINUX:#define __FLT_DIG__ 6
// I386-LINUX:#define __FLT_EPSILON__ 1.19209290e-7F
// I386-LINUX:#define __FLT_EVAL_METHOD__ 0
// I386-LINUX:#define __FLT_HAS_DENORM__ 1
// I386-LINUX:#define __FLT_HAS_INFINITY__ 1
// I386-LINUX:#define __FLT_HAS_QUIET_NAN__ 1
// I386-LINUX:#define __FLT_MANT_DIG__ 24
// I386-LINUX:#define __FLT_MAX_10_EXP__ 38
// I386-LINUX:#define __FLT_MAX_EXP__ 128
// I386-LINUX:#define __FLT_MAX__ 3.40282347e+38F
// I386-LINUX:#define __FLT_MIN_10_EXP__ (-37)
// I386-LINUX:#define __FLT_MIN_EXP__ (-125)
// I386-LINUX:#define __FLT_MIN__ 1.17549435e-38F
// I386-LINUX:#define __FLT_RADIX__ 2
// I386-LINUX:#define __INT16_C_SUFFIX__
// I386-LINUX:#define __INT16_FMTd__ "hd"
// I386-LINUX:#define __INT16_FMTi__ "hi"
// I386-LINUX:#define __INT16_MAX__ 32767
// I386-LINUX:#define __INT16_TYPE__ short
// I386-LINUX:#define __INT32_C_SUFFIX__
// I386-LINUX:#define __INT32_FMTd__ "d"
// I386-LINUX:#define __INT32_FMTi__ "i"
// I386-LINUX:#define __INT32_MAX__ 2147483647
// I386-LINUX:#define __INT32_TYPE__ int
// I386-LINUX:#define __INT64_C_SUFFIX__ LL
// I386-LINUX:#define __INT64_FMTd__ "lld"
// I386-LINUX:#define __INT64_FMTi__ "lli"
// I386-LINUX:#define __INT64_MAX__ 9223372036854775807LL
// I386-LINUX:#define __INT64_TYPE__ long long int
// I386-LINUX:#define __INT8_C_SUFFIX__
// I386-LINUX:#define __INT8_FMTd__ "hhd"
// I386-LINUX:#define __INT8_FMTi__ "hhi"
// I386-LINUX:#define __INT8_MAX__ 127
// I386-LINUX:#define __INT8_TYPE__ signed char
// I386-LINUX:#define __INTMAX_C_SUFFIX__ LL
// I386-LINUX:#define __INTMAX_FMTd__ "lld"
// I386-LINUX:#define __INTMAX_FMTi__ "lli"
// I386-LINUX:#define __INTMAX_MAX__ 9223372036854775807LL
// I386-LINUX:#define __INTMAX_TYPE__ long long int
// I386-LINUX:#define __INTMAX_WIDTH__ 64
// I386-LINUX:#define __INTPTR_FMTd__ "d"
// I386-LINUX:#define __INTPTR_FMTi__ "i"
// I386-LINUX:#define __INTPTR_MAX__ 2147483647
// I386-LINUX:#define __INTPTR_TYPE__ int
// I386-LINUX:#define __INTPTR_WIDTH__ 32
// I386-LINUX:#define __INT_FAST16_FMTd__ "hd"
// I386-LINUX:#define __INT_FAST16_FMTi__ "hi"
// I386-LINUX:#define __INT_FAST16_MAX__ 32767
// I386-LINUX:#define __INT_FAST16_TYPE__ short
// I386-LINUX:#define __INT_FAST32_FMTd__ "d"
// I386-LINUX:#define __INT_FAST32_FMTi__ "i"
// I386-LINUX:#define __INT_FAST32_MAX__ 2147483647
// I386-LINUX:#define __INT_FAST32_TYPE__ int
// I386-LINUX:#define __INT_FAST64_FMTd__ "lld"
// I386-LINUX:#define __INT_FAST64_FMTi__ "lli"
// I386-LINUX:#define __INT_FAST64_MAX__ 9223372036854775807LL
// I386-LINUX:#define __INT_FAST64_TYPE__ long long int
// I386-LINUX:#define __INT_FAST8_FMTd__ "hhd"
// I386-LINUX:#define __INT_FAST8_FMTi__ "hhi"
// I386-LINUX:#define __INT_FAST8_MAX__ 127
// I386-LINUX:#define __INT_FAST8_TYPE__ signed char
// I386-LINUX:#define __INT_LEAST16_FMTd__ "hd"
// I386-LINUX:#define __INT_LEAST16_FMTi__ "hi"
// I386-LINUX:#define __INT_LEAST16_MAX__ 32767
// I386-LINUX:#define __INT_LEAST16_TYPE__ short
// I386-LINUX:#define __INT_LEAST32_FMTd__ "d"
// I386-LINUX:#define __INT_LEAST32_FMTi__ "i"
// I386-LINUX:#define __INT_LEAST32_MAX__ 2147483647
// I386-LINUX:#define __INT_LEAST32_TYPE__ int
// I386-LINUX:#define __INT_LEAST64_FMTd__ "lld"
// I386-LINUX:#define __INT_LEAST64_FMTi__ "lli"
// I386-LINUX:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// I386-LINUX:#define __INT_LEAST64_TYPE__ long long int
// I386-LINUX:#define __INT_LEAST8_FMTd__ "hhd"
// I386-LINUX:#define __INT_LEAST8_FMTi__ "hhi"
// I386-LINUX:#define __INT_LEAST8_MAX__ 127
// I386-LINUX:#define __INT_LEAST8_TYPE__ signed char
// I386-LINUX:#define __INT_MAX__ 2147483647
// I386-LINUX:#define __LDBL_DENORM_MIN__ 3.64519953188247460253e-4951L
// I386-LINUX:#define __LDBL_DIG__ 18
// I386-LINUX:#define __LDBL_EPSILON__ 1.08420217248550443401e-19L
// I386-LINUX:#define __LDBL_HAS_DENORM__ 1
// I386-LINUX:#define __LDBL_HAS_INFINITY__ 1
// I386-LINUX:#define __LDBL_HAS_QUIET_NAN__ 1
// I386-LINUX:#define __LDBL_MANT_DIG__ 64
// I386-LINUX:#define __LDBL_MAX_10_EXP__ 4932
// I386-LINUX:#define __LDBL_MAX_EXP__ 16384
// I386-LINUX:#define __LDBL_MAX__ 1.18973149535723176502e+4932L
// I386-LINUX:#define __LDBL_MIN_10_EXP__ (-4931)
// I386-LINUX:#define __LDBL_MIN_EXP__ (-16381)
// I386-LINUX:#define __LDBL_MIN__ 3.36210314311209350626e-4932L
// I386-LINUX:#define __LITTLE_ENDIAN__ 1
// I386-LINUX:#define __LONG_LONG_MAX__ 9223372036854775807LL
// I386-LINUX:#define __LONG_MAX__ 2147483647L
// I386-LINUX-NOT:#define __LP64__
// I386-LINUX:#define __NO_MATH_INLINES 1
// I386-LINUX:#define __POINTER_WIDTH__ 32
// I386-LINUX:#define __PTRDIFF_TYPE__ int
// I386-LINUX:#define __PTRDIFF_WIDTH__ 32
// I386-LINUX:#define __REGISTER_PREFIX__
// I386-LINUX:#define __SCHAR_MAX__ 127
// I386-LINUX:#define __SHRT_MAX__ 32767
// I386-LINUX:#define __SIG_ATOMIC_MAX__ 2147483647
// I386-LINUX:#define __SIG_ATOMIC_WIDTH__ 32
// I386-LINUX:#define __SIZEOF_DOUBLE__ 8
// I386-LINUX:#define __SIZEOF_FLOAT__ 4
// I386-LINUX:#define __SIZEOF_INT__ 4
// I386-LINUX:#define __SIZEOF_LONG_DOUBLE__ 12
// I386-LINUX:#define __SIZEOF_LONG_LONG__ 8
// I386-LINUX:#define __SIZEOF_LONG__ 4
// I386-LINUX:#define __SIZEOF_POINTER__ 4
// I386-LINUX:#define __SIZEOF_PTRDIFF_T__ 4
// I386-LINUX:#define __SIZEOF_SHORT__ 2
// I386-LINUX:#define __SIZEOF_SIZE_T__ 4
// I386-LINUX:#define __SIZEOF_WCHAR_T__ 4
// I386-LINUX:#define __SIZEOF_WINT_T__ 4
// I386-LINUX:#define __SIZE_MAX__ 4294967295U
// I386-LINUX:#define __SIZE_TYPE__ unsigned int
// I386-LINUX:#define __SIZE_WIDTH__ 32
// I386-LINUX-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 8U
// I386-LINUX:#define __UINT16_C_SUFFIX__
// I386-LINUX:#define __UINT16_MAX__ 65535
// I386-LINUX:#define __UINT16_TYPE__ unsigned short
// I386-LINUX:#define __UINT32_C_SUFFIX__ U
// I386-LINUX:#define __UINT32_MAX__ 4294967295U
// I386-LINUX:#define __UINT32_TYPE__ unsigned int
// I386-LINUX:#define __UINT64_C_SUFFIX__ ULL
// I386-LINUX:#define __UINT64_MAX__ 18446744073709551615ULL
// I386-LINUX:#define __UINT64_TYPE__ long long unsigned int
// I386-LINUX:#define __UINT8_C_SUFFIX__
// I386-LINUX:#define __UINT8_MAX__ 255
// I386-LINUX:#define __UINT8_TYPE__ unsigned char
// I386-LINUX:#define __UINTMAX_C_SUFFIX__ ULL
// I386-LINUX:#define __UINTMAX_MAX__ 18446744073709551615ULL
// I386-LINUX:#define __UINTMAX_TYPE__ long long unsigned int
// I386-LINUX:#define __UINTMAX_WIDTH__ 64
// I386-LINUX:#define __UINTPTR_MAX__ 4294967295U
// I386-LINUX:#define __UINTPTR_TYPE__ unsigned int
// I386-LINUX:#define __UINTPTR_WIDTH__ 32
// I386-LINUX:#define __UINT_FAST16_MAX__ 65535
// I386-LINUX:#define __UINT_FAST16_TYPE__ unsigned short
// I386-LINUX:#define __UINT_FAST32_MAX__ 4294967295U
// I386-LINUX:#define __UINT_FAST32_TYPE__ unsigned int
// I386-LINUX:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// I386-LINUX:#define __UINT_FAST64_TYPE__ long long unsigned int
// I386-LINUX:#define __UINT_FAST8_MAX__ 255
// I386-LINUX:#define __UINT_FAST8_TYPE__ unsigned char
// I386-LINUX:#define __UINT_LEAST16_MAX__ 65535
// I386-LINUX:#define __UINT_LEAST16_TYPE__ unsigned short
// I386-LINUX:#define __UINT_LEAST32_MAX__ 4294967295U
// I386-LINUX:#define __UINT_LEAST32_TYPE__ unsigned int
// I386-LINUX:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// I386-LINUX:#define __UINT_LEAST64_TYPE__ long long unsigned int
// I386-LINUX:#define __UINT_LEAST8_MAX__ 255
// I386-LINUX:#define __UINT_LEAST8_TYPE__ unsigned char
// I386-LINUX:#define __USER_LABEL_PREFIX__
// I386-LINUX:#define __WCHAR_MAX__ 2147483647
// I386-LINUX:#define __WCHAR_TYPE__ int
// I386-LINUX:#define __WCHAR_WIDTH__ 32
// I386-LINUX:#define __WINT_TYPE__ unsigned int
// I386-LINUX:#define __WINT_WIDTH__ 32
// I386-LINUX:#define __i386 1
// I386-LINUX:#define __i386__ 1
// I386-LINUX:#define i386 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=i386-netbsd < /dev/null | FileCheck -match-full-lines -check-prefix I386-NETBSD %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=i386-netbsd < /dev/null | FileCheck -match-full-lines -check-prefix I386-NETBSD -check-prefix I386-NETBSD-CXX %s
//
// I386-NETBSD-NOT:#define _LP64
// I386-NETBSD:#define __BIGGEST_ALIGNMENT__ 16
// I386-NETBSD:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// I386-NETBSD:#define __CHAR16_TYPE__ unsigned short
// I386-NETBSD:#define __CHAR32_TYPE__ unsigned int
// I386-NETBSD:#define __CHAR_BIT__ 8
// I386-NETBSD:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// I386-NETBSD:#define __DBL_DIG__ 15
// I386-NETBSD:#define __DBL_EPSILON__ 2.2204460492503131e-16
// I386-NETBSD:#define __DBL_HAS_DENORM__ 1
// I386-NETBSD:#define __DBL_HAS_INFINITY__ 1
// I386-NETBSD:#define __DBL_HAS_QUIET_NAN__ 1
// I386-NETBSD:#define __DBL_MANT_DIG__ 53
// I386-NETBSD:#define __DBL_MAX_10_EXP__ 308
// I386-NETBSD:#define __DBL_MAX_EXP__ 1024
// I386-NETBSD:#define __DBL_MAX__ 1.7976931348623157e+308
// I386-NETBSD:#define __DBL_MIN_10_EXP__ (-307)
// I386-NETBSD:#define __DBL_MIN_EXP__ (-1021)
// I386-NETBSD:#define __DBL_MIN__ 2.2250738585072014e-308
// I386-NETBSD:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// I386-NETBSD:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// I386-NETBSD:#define __FLT_DIG__ 6
// I386-NETBSD:#define __FLT_EPSILON__ 1.19209290e-7F
// I386-NETBSD:#define __FLT_EVAL_METHOD__ 2
// I386-NETBSD:#define __FLT_HAS_DENORM__ 1
// I386-NETBSD:#define __FLT_HAS_INFINITY__ 1
// I386-NETBSD:#define __FLT_HAS_QUIET_NAN__ 1
// I386-NETBSD:#define __FLT_MANT_DIG__ 24
// I386-NETBSD:#define __FLT_MAX_10_EXP__ 38
// I386-NETBSD:#define __FLT_MAX_EXP__ 128
// I386-NETBSD:#define __FLT_MAX__ 3.40282347e+38F
// I386-NETBSD:#define __FLT_MIN_10_EXP__ (-37)
// I386-NETBSD:#define __FLT_MIN_EXP__ (-125)
// I386-NETBSD:#define __FLT_MIN__ 1.17549435e-38F
// I386-NETBSD:#define __FLT_RADIX__ 2
// I386-NETBSD:#define __INT16_C_SUFFIX__
// I386-NETBSD:#define __INT16_FMTd__ "hd"
// I386-NETBSD:#define __INT16_FMTi__ "hi"
// I386-NETBSD:#define __INT16_MAX__ 32767
// I386-NETBSD:#define __INT16_TYPE__ short
// I386-NETBSD:#define __INT32_C_SUFFIX__
// I386-NETBSD:#define __INT32_FMTd__ "d"
// I386-NETBSD:#define __INT32_FMTi__ "i"
// I386-NETBSD:#define __INT32_MAX__ 2147483647
// I386-NETBSD:#define __INT32_TYPE__ int
// I386-NETBSD:#define __INT64_C_SUFFIX__ LL
// I386-NETBSD:#define __INT64_FMTd__ "lld"
// I386-NETBSD:#define __INT64_FMTi__ "lli"
// I386-NETBSD:#define __INT64_MAX__ 9223372036854775807LL
// I386-NETBSD:#define __INT64_TYPE__ long long int
// I386-NETBSD:#define __INT8_C_SUFFIX__
// I386-NETBSD:#define __INT8_FMTd__ "hhd"
// I386-NETBSD:#define __INT8_FMTi__ "hhi"
// I386-NETBSD:#define __INT8_MAX__ 127
// I386-NETBSD:#define __INT8_TYPE__ signed char
// I386-NETBSD:#define __INTMAX_C_SUFFIX__ LL
// I386-NETBSD:#define __INTMAX_FMTd__ "lld"
// I386-NETBSD:#define __INTMAX_FMTi__ "lli"
// I386-NETBSD:#define __INTMAX_MAX__ 9223372036854775807LL
// I386-NETBSD:#define __INTMAX_TYPE__ long long int
// I386-NETBSD:#define __INTMAX_WIDTH__ 64
// I386-NETBSD:#define __INTPTR_FMTd__ "d"
// I386-NETBSD:#define __INTPTR_FMTi__ "i"
// I386-NETBSD:#define __INTPTR_MAX__ 2147483647
// I386-NETBSD:#define __INTPTR_TYPE__ int
// I386-NETBSD:#define __INTPTR_WIDTH__ 32
// I386-NETBSD:#define __INT_FAST16_FMTd__ "hd"
// I386-NETBSD:#define __INT_FAST16_FMTi__ "hi"
// I386-NETBSD:#define __INT_FAST16_MAX__ 32767
// I386-NETBSD:#define __INT_FAST16_TYPE__ short
// I386-NETBSD:#define __INT_FAST32_FMTd__ "d"
// I386-NETBSD:#define __INT_FAST32_FMTi__ "i"
// I386-NETBSD:#define __INT_FAST32_MAX__ 2147483647
// I386-NETBSD:#define __INT_FAST32_TYPE__ int
// I386-NETBSD:#define __INT_FAST64_FMTd__ "lld"
// I386-NETBSD:#define __INT_FAST64_FMTi__ "lli"
// I386-NETBSD:#define __INT_FAST64_MAX__ 9223372036854775807LL
// I386-NETBSD:#define __INT_FAST64_TYPE__ long long int
// I386-NETBSD:#define __INT_FAST8_FMTd__ "hhd"
// I386-NETBSD:#define __INT_FAST8_FMTi__ "hhi"
// I386-NETBSD:#define __INT_FAST8_MAX__ 127
// I386-NETBSD:#define __INT_FAST8_TYPE__ signed char
// I386-NETBSD:#define __INT_LEAST16_FMTd__ "hd"
// I386-NETBSD:#define __INT_LEAST16_FMTi__ "hi"
// I386-NETBSD:#define __INT_LEAST16_MAX__ 32767
// I386-NETBSD:#define __INT_LEAST16_TYPE__ short
// I386-NETBSD:#define __INT_LEAST32_FMTd__ "d"
// I386-NETBSD:#define __INT_LEAST32_FMTi__ "i"
// I386-NETBSD:#define __INT_LEAST32_MAX__ 2147483647
// I386-NETBSD:#define __INT_LEAST32_TYPE__ int
// I386-NETBSD:#define __INT_LEAST64_FMTd__ "lld"
// I386-NETBSD:#define __INT_LEAST64_FMTi__ "lli"
// I386-NETBSD:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// I386-NETBSD:#define __INT_LEAST64_TYPE__ long long int
// I386-NETBSD:#define __INT_LEAST8_FMTd__ "hhd"
// I386-NETBSD:#define __INT_LEAST8_FMTi__ "hhi"
// I386-NETBSD:#define __INT_LEAST8_MAX__ 127
// I386-NETBSD:#define __INT_LEAST8_TYPE__ signed char
// I386-NETBSD:#define __INT_MAX__ 2147483647
// I386-NETBSD:#define __LDBL_DENORM_MIN__ 3.64519953188247460253e-4951L
// I386-NETBSD:#define __LDBL_DIG__ 18
// I386-NETBSD:#define __LDBL_EPSILON__ 1.08420217248550443401e-19L
// I386-NETBSD:#define __LDBL_HAS_DENORM__ 1
// I386-NETBSD:#define __LDBL_HAS_INFINITY__ 1
// I386-NETBSD:#define __LDBL_HAS_QUIET_NAN__ 1
// I386-NETBSD:#define __LDBL_MANT_DIG__ 64
// I386-NETBSD:#define __LDBL_MAX_10_EXP__ 4932
// I386-NETBSD:#define __LDBL_MAX_EXP__ 16384
// I386-NETBSD:#define __LDBL_MAX__ 1.18973149535723176502e+4932L
// I386-NETBSD:#define __LDBL_MIN_10_EXP__ (-4931)
// I386-NETBSD:#define __LDBL_MIN_EXP__ (-16381)
// I386-NETBSD:#define __LDBL_MIN__ 3.36210314311209350626e-4932L
// I386-NETBSD:#define __LITTLE_ENDIAN__ 1
// I386-NETBSD:#define __LONG_LONG_MAX__ 9223372036854775807LL
// I386-NETBSD:#define __LONG_MAX__ 2147483647L
// I386-NETBSD-NOT:#define __LP64__
// I386-NETBSD:#define __NO_MATH_INLINES 1
// I386-NETBSD:#define __POINTER_WIDTH__ 32
// I386-NETBSD:#define __PTRDIFF_TYPE__ int
// I386-NETBSD:#define __PTRDIFF_WIDTH__ 32
// I386-NETBSD:#define __REGISTER_PREFIX__
// I386-NETBSD:#define __SCHAR_MAX__ 127
// I386-NETBSD:#define __SHRT_MAX__ 32767
// I386-NETBSD:#define __SIG_ATOMIC_MAX__ 2147483647
// I386-NETBSD:#define __SIG_ATOMIC_WIDTH__ 32
// I386-NETBSD:#define __SIZEOF_DOUBLE__ 8
// I386-NETBSD:#define __SIZEOF_FLOAT__ 4
// I386-NETBSD:#define __SIZEOF_INT__ 4
// I386-NETBSD:#define __SIZEOF_LONG_DOUBLE__ 12
// I386-NETBSD:#define __SIZEOF_LONG_LONG__ 8
// I386-NETBSD:#define __SIZEOF_LONG__ 4
// I386-NETBSD:#define __SIZEOF_POINTER__ 4
// I386-NETBSD:#define __SIZEOF_PTRDIFF_T__ 4
// I386-NETBSD:#define __SIZEOF_SHORT__ 2
// I386-NETBSD:#define __SIZEOF_SIZE_T__ 4
// I386-NETBSD:#define __SIZEOF_WCHAR_T__ 4
// I386-NETBSD:#define __SIZEOF_WINT_T__ 4
// I386-NETBSD:#define __SIZE_MAX__ 4294967295U
// I386-NETBSD:#define __SIZE_TYPE__ unsigned int
// I386-NETBSD:#define __SIZE_WIDTH__ 32
// I386-NETBSD-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 4U
// I386-NETBSD:#define __UINT16_C_SUFFIX__
// I386-NETBSD:#define __UINT16_MAX__ 65535
// I386-NETBSD:#define __UINT16_TYPE__ unsigned short
// I386-NETBSD:#define __UINT32_C_SUFFIX__ U
// I386-NETBSD:#define __UINT32_MAX__ 4294967295U
// I386-NETBSD:#define __UINT32_TYPE__ unsigned int
// I386-NETBSD:#define __UINT64_C_SUFFIX__ ULL
// I386-NETBSD:#define __UINT64_MAX__ 18446744073709551615ULL
// I386-NETBSD:#define __UINT64_TYPE__ long long unsigned int
// I386-NETBSD:#define __UINT8_C_SUFFIX__
// I386-NETBSD:#define __UINT8_MAX__ 255
// I386-NETBSD:#define __UINT8_TYPE__ unsigned char
// I386-NETBSD:#define __UINTMAX_C_SUFFIX__ ULL
// I386-NETBSD:#define __UINTMAX_MAX__ 18446744073709551615ULL
// I386-NETBSD:#define __UINTMAX_TYPE__ long long unsigned int
// I386-NETBSD:#define __UINTMAX_WIDTH__ 64
// I386-NETBSD:#define __UINTPTR_MAX__ 4294967295U
// I386-NETBSD:#define __UINTPTR_TYPE__ unsigned int
// I386-NETBSD:#define __UINTPTR_WIDTH__ 32
// I386-NETBSD:#define __UINT_FAST16_MAX__ 65535
// I386-NETBSD:#define __UINT_FAST16_TYPE__ unsigned short
// I386-NETBSD:#define __UINT_FAST32_MAX__ 4294967295U
// I386-NETBSD:#define __UINT_FAST32_TYPE__ unsigned int
// I386-NETBSD:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// I386-NETBSD:#define __UINT_FAST64_TYPE__ long long unsigned int
// I386-NETBSD:#define __UINT_FAST8_MAX__ 255
// I386-NETBSD:#define __UINT_FAST8_TYPE__ unsigned char
// I386-NETBSD:#define __UINT_LEAST16_MAX__ 65535
// I386-NETBSD:#define __UINT_LEAST16_TYPE__ unsigned short
// I386-NETBSD:#define __UINT_LEAST32_MAX__ 4294967295U
// I386-NETBSD:#define __UINT_LEAST32_TYPE__ unsigned int
// I386-NETBSD:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// I386-NETBSD:#define __UINT_LEAST64_TYPE__ long long unsigned int
// I386-NETBSD:#define __UINT_LEAST8_MAX__ 255
// I386-NETBSD:#define __UINT_LEAST8_TYPE__ unsigned char
// I386-NETBSD:#define __USER_LABEL_PREFIX__
// I386-NETBSD:#define __WCHAR_MAX__ 2147483647
// I386-NETBSD:#define __WCHAR_TYPE__ int
// I386-NETBSD:#define __WCHAR_WIDTH__ 32
// I386-NETBSD:#define __WINT_TYPE__ int
// I386-NETBSD:#define __WINT_WIDTH__ 32
// I386-NETBSD:#define __i386 1
// I386-NETBSD:#define __i386__ 1
// I386-NETBSD:#define i386 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=i386-netbsd -target-feature +sse2 < /dev/null | FileCheck -match-full-lines -check-prefix I386-NETBSD-SSE %s
// I386-NETBSD-SSE:#define __FLT_EVAL_METHOD__ 0
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=i386-netbsd6  < /dev/null | FileCheck -match-full-lines -check-prefix I386-NETBSD6 %s
// I386-NETBSD6:#define __FLT_EVAL_METHOD__ 1
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=i386-netbsd6 -target-feature +sse2 < /dev/null | FileCheck -match-full-lines -check-prefix I386-NETBSD6-SSE %s
// I386-NETBSD6-SSE:#define __FLT_EVAL_METHOD__ 1

// RUN: %clang_cc1 -E -dM -triple=i686-pc-mingw32 < /dev/null | FileCheck -match-full-lines -check-prefix I386-DECLSPEC %s
// RUN: %clang_cc1 -E -dM -fms-extensions -triple=i686-pc-mingw32 < /dev/null | FileCheck -match-full-lines -check-prefix I386-DECLSPEC %s
// RUN: %clang_cc1 -E -dM -triple=i686-unknown-cygwin < /dev/null | FileCheck -match-full-lines -check-prefix I386-DECLSPEC %s
// RUN: %clang_cc1 -E -dM -fms-extensions -triple=i686-unknown-cygwin < /dev/null | FileCheck -match-full-lines -check-prefix I386-DECLSPEC %s
// I386-DECLSPEC: #define __declspec{{.*}}

//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips-none-none < /dev/null | FileCheck -match-full-lines -check-prefix MIPS32BE -check-prefix MIPS32BE-C %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=mips-none-none < /dev/null | FileCheck -match-full-lines -check-prefix MIPS32BE -check-prefix MIPS32BE-CXX %s
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
// MIPS32BE-C:#define __STDC_VERSION__ 201112L
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
// MIPS32BE:#define __mips_fpr 32
// MIPS32BE:#define __mips_hard_float 1
// MIPS32BE:#define __mips_o32 1
// MIPS32BE:#define _mips 1
// MIPS32BE:#define mips 1
//
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
// MIPS32EL:#define __mips_fpr 32
// MIPS32EL:#define __mips_hard_float 1
// MIPS32EL:#define __mips_o32 1
// MIPS32EL:#define _mips 1
// MIPS32EL:#define mips 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding \
// RUN:            -triple=mips64-none-none -target-abi n32 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPSN32BE -check-prefix MIPSN32BE-C %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding \
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
// MIPSN32BE-C: #define __STDC_VERSION__ 201112L
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
//
// RUN: %clang_cc1 -E -dM -ffreestanding \
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
// MIPSN32EL: #define __STDC_VERSION__ 201112L
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
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips64-none-none < /dev/null | FileCheck -match-full-lines -check-prefix MIPS64BE %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=mips64-none-none < /dev/null | FileCheck -match-full-lines -check-prefix MIPS64BE -check-prefix MIPS64BE-CXX %s
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
//
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
//
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
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips-none-nones \
// RUN:            -target-cpu mips32 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-32 %s
//
// MIPS-ARCH-32:#define _MIPS_ARCH "mips32"
// MIPS-ARCH-32:#define _MIPS_ARCH_MIPS32 1
// MIPS-ARCH-32:#define _MIPS_ISA _MIPS_ISA_MIPS32
// MIPS-ARCH-32:#define __mips_isa_rev 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips-none-none \
// RUN:            -target-cpu mips32r2 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-32R2 %s
//
// MIPS-ARCH-32R2:#define _MIPS_ARCH "mips32r2"
// MIPS-ARCH-32R2:#define _MIPS_ARCH_MIPS32R2 1
// MIPS-ARCH-32R2:#define _MIPS_ISA _MIPS_ISA_MIPS32
// MIPS-ARCH-32R2:#define __mips_isa_rev 2
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips-none-none \
// RUN:            -target-cpu mips32r3 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-32R3 %s
//
// MIPS-ARCH-32R3:#define _MIPS_ARCH "mips32r3"
// MIPS-ARCH-32R3:#define _MIPS_ARCH_MIPS32R3 1
// MIPS-ARCH-32R3:#define _MIPS_ISA _MIPS_ISA_MIPS32
// MIPS-ARCH-32R3:#define __mips_isa_rev 3
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips-none-none \
// RUN:            -target-cpu mips32r5 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-32R5 %s
//
// MIPS-ARCH-32R5:#define _MIPS_ARCH "mips32r5"
// MIPS-ARCH-32R5:#define _MIPS_ARCH_MIPS32R5 1
// MIPS-ARCH-32R5:#define _MIPS_ISA _MIPS_ISA_MIPS32
// MIPS-ARCH-32R5:#define __mips_isa_rev 5
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips-none-none \
// RUN:            -target-cpu mips32r6 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-32R6 %s
//
// MIPS-ARCH-32R6:#define _MIPS_ARCH "mips32r6"
// MIPS-ARCH-32R6:#define _MIPS_ARCH_MIPS32R6 1
// MIPS-ARCH-32R6:#define _MIPS_ISA _MIPS_ISA_MIPS32
// MIPS-ARCH-32R6:#define __mips_isa_rev 6
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips64-none-none \
// RUN:            < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-DEF64 %s
//
// MIPS-ARCH-DEF64:#define _MIPS_ARCH "mips64r2"
// MIPS-ARCH-DEF64:#define _MIPS_ARCH_MIPS64R2 1
// MIPS-ARCH-DEF64:#define _MIPS_ISA _MIPS_ISA_MIPS64
// MIPS-ARCH-DEF64:#define __mips_isa_rev 2
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips64-none-none \
// RUN:            -target-cpu mips64 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-64 %s
//
// MIPS-ARCH-64:#define _MIPS_ARCH "mips64"
// MIPS-ARCH-64:#define _MIPS_ARCH_MIPS64 1
// MIPS-ARCH-64:#define _MIPS_ISA _MIPS_ISA_MIPS64
// MIPS-ARCH-64:#define __mips_isa_rev 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips64-none-none \
// RUN:            -target-cpu mips64r2 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-64R2 %s
//
// MIPS-ARCH-64R2:#define _MIPS_ARCH "mips64r2"
// MIPS-ARCH-64R2:#define _MIPS_ARCH_MIPS64R2 1
// MIPS-ARCH-64R2:#define _MIPS_ISA _MIPS_ISA_MIPS64
// MIPS-ARCH-64R2:#define __mips_isa_rev 2
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips64-none-none \
// RUN:            -target-cpu mips64r3 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-64R3 %s
//
// MIPS-ARCH-64R3:#define _MIPS_ARCH "mips64r3"
// MIPS-ARCH-64R3:#define _MIPS_ARCH_MIPS64R3 1
// MIPS-ARCH-64R3:#define _MIPS_ISA _MIPS_ISA_MIPS64
// MIPS-ARCH-64R3:#define __mips_isa_rev 3
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips64-none-none \
// RUN:            -target-cpu mips64r5 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-64R5 %s
//
// MIPS-ARCH-64R5:#define _MIPS_ARCH "mips64r5"
// MIPS-ARCH-64R5:#define _MIPS_ARCH_MIPS64R5 1
// MIPS-ARCH-64R5:#define _MIPS_ISA _MIPS_ISA_MIPS64
// MIPS-ARCH-64R5:#define __mips_isa_rev 5
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips64-none-none \
// RUN:            -target-cpu mips64r6 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ARCH-64R6 %s
//
// MIPS-ARCH-64R6:#define _MIPS_ARCH "mips64r6"
// MIPS-ARCH-64R6:#define _MIPS_ARCH_MIPS64R6 1
// MIPS-ARCH-64R6:#define _MIPS_ISA _MIPS_ISA_MIPS64
// MIPS-ARCH-64R6:#define __mips_isa_rev 6
//
// Check MIPS float ABI macros
//
// RUN: %clang_cc1 -E -dM -ffreestanding \
// RUN:   -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-FABI-HARD %s
// MIPS-FABI-HARD:#define __mips_hard_float 1
//
// RUN: %clang_cc1 -target-feature +soft-float -E -dM -ffreestanding \
// RUN:   -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-FABI-SOFT %s
// MIPS-FABI-SOFT:#define __mips_soft_float 1
//
// RUN: %clang_cc1 -target-feature +single-float -E -dM -ffreestanding \
// RUN:   -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-FABI-SINGLE %s
// MIPS-FABI-SINGLE:#define __mips_hard_float 1
// MIPS-FABI-SINGLE:#define __mips_single_float 1
//
// RUN: %clang_cc1 -target-feature +soft-float -target-feature +single-float \
// RUN:   -E -dM -ffreestanding -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-FABI-SINGLE-SOFT %s
// MIPS-FABI-SINGLE-SOFT:#define __mips_single_float 1
// MIPS-FABI-SINGLE-SOFT:#define __mips_soft_float 1
//
// Check MIPS features macros
//
// RUN: %clang_cc1 -target-feature +mips16 \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS16 %s
// MIPS16:#define __mips16 1
//
// RUN: %clang_cc1 -target-feature -mips16 \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix NOMIPS16 %s
// NOMIPS16-NOT:#define __mips16 1
//
// RUN: %clang_cc1 -target-feature +micromips \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MICROMIPS %s
// MICROMIPS:#define __mips_micromips 1
//
// RUN: %clang_cc1 -target-feature -micromips \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix NOMICROMIPS %s
// NOMICROMIPS-NOT:#define __mips_micromips 1
//
// RUN: %clang_cc1 -target-feature +dsp \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-DSP %s
// MIPS-DSP:#define __mips_dsp 1
// MIPS-DSP:#define __mips_dsp_rev 1
// MIPS-DSP-NOT:#define __mips_dspr2 1
//
// RUN: %clang_cc1 -target-feature +dspr2 \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-DSPR2 %s
// MIPS-DSPR2:#define __mips_dsp 1
// MIPS-DSPR2:#define __mips_dsp_rev 2
// MIPS-DSPR2:#define __mips_dspr2 1
//
// RUN: %clang_cc1 -target-feature +msa \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-MSA %s
// MIPS-MSA:#define __mips_msa 1
//
// RUN: %clang_cc1 -target-cpu mips32r3 -target-feature +nan2008 \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-NAN2008 %s
// MIPS-NAN2008:#define __mips_nan2008 1
//
// RUN: %clang_cc1 -target-cpu mips32r3 -target-feature -nan2008 \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix NOMIPS-NAN2008 %s
// NOMIPS-NAN2008-NOT:#define __mips_nan2008 1
//
// RUN: %clang_cc1 -target-feature -fp64 \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS32-MFP32 %s
// MIPS32-MFP32:#define _MIPS_FPSET 16
// MIPS32-MFP32:#define __mips_fpr 32
//
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
// MIPS32-MFP32SF:#define __mips_fpr 32
//
// RUN: %clang_cc1 -target-feature +fp64 \
// RUN:   -E -dM -triple=mips64-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS64-MFP64 %s
// MIPS64-MFP64:#define _MIPS_FPSET 32
// MIPS64-MFP64:#define __mips_fpr 64
//
// RUN: %clang_cc1 -target-feature -fp64 -target-feature +single-float \
// RUN:   -E -dM -triple=mips64-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS64-NOMFP64 %s
// MIPS64-NOMFP64:#define _MIPS_FPSET 32
// MIPS64-NOMFP64:#define __mips_fpr 32
//
// RUN: %clang_cc1 -target-cpu mips32r6 \
// RUN:   -E -dM -triple=mips-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-XXR6 %s
// RUN: %clang_cc1 -target-cpu mips64r6 \
// RUN:   -E -dM -triple=mips64-none-none < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-XXR6 %s
// MIPS-XXR6:#define _MIPS_FPSET 32
// MIPS-XXR6:#define __mips_fpr 64
// MIPS-XXR6:#define __mips_nan2008 1
//
// RUN: %clang_cc1 -target-cpu mips32 \
// RUN:   -E -dM -triple=mips-unknown-netbsd -mrelocation-model pic < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ABICALLS-NETBSD %s
// MIPS-ABICALLS-NETBSD-NOT: #define __ABICALLS__ 1
// MIPS-ABICALLS-NETBSD: #define __mips_abicalls 1
//
// RUN: %clang_cc1 -target-cpu mips64 \
// RUN:   -E -dM -triple=mips64-unknown-netbsd -mrelocation-model pic < \
// RUN:   /dev/null | FileCheck -match-full-lines \
// RUN:   -check-prefix MIPS-ABICALLS-NETBSD64 %s
// MIPS-ABICALLS-NETBSD64-NOT: #define __ABICALLS__ 1
// MIPS-ABICALLS-NETBSD64: #define __mips_abicalls 1
//
// RUN: %clang_cc1 -target-cpu mips32 \
// RUN:   -E -dM -triple=mips-unknown-freebsd -mrelocation-model pic < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ABICALLS-FREEBSD %s
// MIPS-ABICALLS-FREEBSD: #define __ABICALLS__ 1
// MIPS-ABICALLS-FREEBSD: #define __mips_abicalls 1
//
// RUN: %clang_cc1 -target-cpu mips64 \
// RUN:   -E -dM -triple=mips64-unknown-freebsd -mrelocation-model pic < \
// RUN:   /dev/null | FileCheck -match-full-lines \
// RUN:   -check-prefix MIPS-ABICALLS-FREEBSD64 %s
// MIPS-ABICALLS-FREEBSD64: #define __ABICALLS__ 1
// MIPS-ABICALLS-FREEBSD64: #define __mips_abicalls 1
//
// RUN: %clang_cc1 -target-cpu mips32 \
// RUN:   -E -dM -triple=mips-unknown-openbsd -mrelocation-model pic < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix MIPS-ABICALLS-OPENBSD %s
// MIPS-ABICALLS-OPENBSD: #define __ABICALLS__ 1
// MIPS-ABICALLS-OPENBSD: #define __mips_abicalls 1
//
// RUN: %clang_cc1 -target-cpu mips64 \
// RUN:   -E -dM -triple=mips64-unknown-openbsd -mrelocation-model pic < \
// RUN:   /dev/null | FileCheck -match-full-lines \
// RUN:   -check-prefix MIPS-ABICALLS-OPENBSD64 %s
// MIPS-ABICALLS-OPENBSD64: #define __ABICALLS__ 1
// MIPS-ABICALLS-OPENBSD64: #define __mips_abicalls 1
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
// MSP430-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 8U
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
//
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
// PPC64:#define __FLT_EVAL_METHOD__ 0
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
// PPC64:#define __ppc64__ 1
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
// PPC64LE:#define _ARCH_PWR6X 1
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
// PPC64LE:#define __FLT_EVAL_METHOD__ 0
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
// PPC64LE:#define __ppc64__ 1
// PPC64LE:#define __ppc__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-cpu a2q -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCA2Q %s
//
// PPCA2Q:#define _ARCH_A2 1
// PPCA2Q:#define _ARCH_A2Q 1
// PPCA2Q:#define _ARCH_PPC 1
// PPCA2Q:#define _ARCH_PPC64 1
// PPCA2Q:#define _ARCH_QP 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-bgq-linux -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPCBGQ %s
//
// PPCBGQ:#define __THW_BLUEGENE__ 1
// PPCBGQ:#define __TOS_BGQ__ 1
// PPCBGQ:#define __bg__ 1
// PPCBGQ:#define __bgq__ 1
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
// PPCPWR7:#define _ARCH_PWR6X 1
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
// PPCPOWER7:#define _ARCH_PWR6X 1
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
// PPCPWR8:#define _ARCH_PWR6X 1
// PPCPWR8:#define _ARCH_PWR7 1
// PPCPWR8:#define _ARCH_PWR8 1
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
// PPCPOWER8:#define _ARCH_PWR6X 1
// PPCPOWER8:#define _ARCH_PWR7 1
// PPCPOWER8:#define _ARCH_PWR8 1
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
// PPCPWR9:#define _ARCH_PWR6X 1
// PPCPWR9:#define _ARCH_PWR7 1
// PPCPWR9:#define _ARCH_PWR9 1
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
// PPCPOWER9:#define _ARCH_PWR6X 1
// PPCPOWER9:#define _ARCH_PWR7 1
// PPCPOWER9:#define _ARCH_PWR9 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-none-none -target-feature +float128 -target-cpu power8 -fno-signed-char < /dev/null | FileCheck -check-prefix PPC-FLOAT128 %s
// PPC-FLOAT128:#define __FLOAT128__ 1
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
// PPC64-LINUX:#define __FLT_EVAL_METHOD__ 0
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
// PPC64-LINUX:#define __ppc64__ 1
// PPC64-LINUX:#define __ppc__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-unknown-linux-gnu < /dev/null | FileCheck -match-full-lines -check-prefix PPC64-ELFv1 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-unknown-linux-gnu -target-abi elfv1 < /dev/null | FileCheck -match-full-lines -check-prefix PPC64-ELFv1 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-unknown-linux-gnu -target-abi elfv1-qpx < /dev/null | FileCheck -match-full-lines -check-prefix PPC64-ELFv1 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-unknown-linux-gnu -target-abi elfv2 < /dev/null | FileCheck -match-full-lines -check-prefix PPC64-ELFv2 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64le-unknown-linux-gnu < /dev/null | FileCheck -match-full-lines -check-prefix PPC64-ELFv2 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64le-unknown-linux-gnu -target-abi elfv1 < /dev/null | FileCheck -match-full-lines -check-prefix PPC64-ELFv1 %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64le-unknown-linux-gnu -target-abi elfv2 < /dev/null | FileCheck -match-full-lines -check-prefix PPC64-ELFv2 %s
// PPC64-ELFv1:#define _CALL_ELF 1
// PPC64-ELFv2:#define _CALL_ELF 2
//
// Most of this is encompassed in other places.
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64le-unknown-linux-gnu -target-abi elfv2 < /dev/null | FileCheck -match-full-lines -check-prefix PPC64LE-LINUX %s
//
// PPC64LE-LINUX:#define _CALL_LINUX 1
//
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
//
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
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-unknown-linux-gnu -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix PPC32-LINUX %s
//
// PPC32-LINUX-NOT: _CALL_LINUX
//
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
// PPC-DARWIN:#define __STDC_VERSION__ 201112L
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

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-apple-darwin8 < /dev/null | FileCheck -match-full-lines -check-prefix PPC64-DARWIN %s
// PPC64-DARWIN:#define __STRUCT_PARM_ALIGN__ 16

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

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=s390x-none-none -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix S390X %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=s390x-none-none -fno-signed-char < /dev/null | FileCheck -match-full-lines -check-prefix S390X -check-prefix S390X-CXX %s
//
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
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=sparc-none-none < /dev/null | FileCheck -match-full-lines -check-prefix SPARC -check-prefix SPARC-DEFAULT %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=sparc-rtems-elf < /dev/null | FileCheck -match-full-lines -check-prefix SPARC -check-prefix SPARC-DEFAULT %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=sparc-none-netbsd < /dev/null | FileCheck -match-full-lines -check-prefix SPARC -check-prefix SPARC-NETOPENBSD %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=sparc-none-openbsd < /dev/null | FileCheck -match-full-lines -check-prefix SPARC -check-prefix SPARC-NETOPENBSD %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=sparc-none-none < /dev/null | FileCheck -match-full-lines -check-prefix SPARC -check-prefix SPARC-DEFAULT -check-prefix SPARC-DEFAULT-CXX %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=sparc-none-openbsd < /dev/null | FileCheck -match-full-lines -check-prefix SPARC -check-prefix SPARC-NETOPENBSD -check-prefix SPARC-NETOPENBSD-CXX %s
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
// SPARC:#define __VERSION__ "4.2.1 Compatible{{.*}}
// SPARC:#define __WCHAR_MAX__ 2147483647
// SPARC:#define __WCHAR_TYPE__ int
// SPARC:#define __WCHAR_WIDTH__ 32
// SPARC:#define __WINT_TYPE__ int
// SPARC:#define __WINT_WIDTH__ 32
// SPARC:#define __sparc 1
// SPARC:#define __sparc__ 1
// SPARC:#define __sparcv8 1
// SPARC:#define sparc 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=tce-none-none < /dev/null | FileCheck -match-full-lines -check-prefix TCE %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=tce-none-none < /dev/null | FileCheck -match-full-lines -check-prefix TCE -check-prefix TCE-CXX %s
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
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=x86_64-none-none < /dev/null | FileCheck -match-full-lines -check-prefix X86_64 %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=x86_64-none-none < /dev/null | FileCheck -match-full-lines -check-prefix X86_64 -check-prefix X86_64-CXX %s
//
// X86_64:#define _LP64 1
// X86_64-NOT:#define _LP32 1
// X86_64:#define __BIGGEST_ALIGNMENT__ 16
// X86_64:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// X86_64:#define __CHAR16_TYPE__ unsigned short
// X86_64:#define __CHAR32_TYPE__ unsigned int
// X86_64:#define __CHAR_BIT__ 8
// X86_64:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// X86_64:#define __DBL_DIG__ 15
// X86_64:#define __DBL_EPSILON__ 2.2204460492503131e-16
// X86_64:#define __DBL_HAS_DENORM__ 1
// X86_64:#define __DBL_HAS_INFINITY__ 1
// X86_64:#define __DBL_HAS_QUIET_NAN__ 1
// X86_64:#define __DBL_MANT_DIG__ 53
// X86_64:#define __DBL_MAX_10_EXP__ 308
// X86_64:#define __DBL_MAX_EXP__ 1024
// X86_64:#define __DBL_MAX__ 1.7976931348623157e+308
// X86_64:#define __DBL_MIN_10_EXP__ (-307)
// X86_64:#define __DBL_MIN_EXP__ (-1021)
// X86_64:#define __DBL_MIN__ 2.2250738585072014e-308
// X86_64:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// X86_64:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// X86_64:#define __FLT_DIG__ 6
// X86_64:#define __FLT_EPSILON__ 1.19209290e-7F
// X86_64:#define __FLT_EVAL_METHOD__ 0
// X86_64:#define __FLT_HAS_DENORM__ 1
// X86_64:#define __FLT_HAS_INFINITY__ 1
// X86_64:#define __FLT_HAS_QUIET_NAN__ 1
// X86_64:#define __FLT_MANT_DIG__ 24
// X86_64:#define __FLT_MAX_10_EXP__ 38
// X86_64:#define __FLT_MAX_EXP__ 128
// X86_64:#define __FLT_MAX__ 3.40282347e+38F
// X86_64:#define __FLT_MIN_10_EXP__ (-37)
// X86_64:#define __FLT_MIN_EXP__ (-125)
// X86_64:#define __FLT_MIN__ 1.17549435e-38F
// X86_64:#define __FLT_RADIX__ 2
// X86_64:#define __INT16_C_SUFFIX__
// X86_64:#define __INT16_FMTd__ "hd"
// X86_64:#define __INT16_FMTi__ "hi"
// X86_64:#define __INT16_MAX__ 32767
// X86_64:#define __INT16_TYPE__ short
// X86_64:#define __INT32_C_SUFFIX__
// X86_64:#define __INT32_FMTd__ "d"
// X86_64:#define __INT32_FMTi__ "i"
// X86_64:#define __INT32_MAX__ 2147483647
// X86_64:#define __INT32_TYPE__ int
// X86_64:#define __INT64_C_SUFFIX__ L
// X86_64:#define __INT64_FMTd__ "ld"
// X86_64:#define __INT64_FMTi__ "li"
// X86_64:#define __INT64_MAX__ 9223372036854775807L
// X86_64:#define __INT64_TYPE__ long int
// X86_64:#define __INT8_C_SUFFIX__
// X86_64:#define __INT8_FMTd__ "hhd"
// X86_64:#define __INT8_FMTi__ "hhi"
// X86_64:#define __INT8_MAX__ 127
// X86_64:#define __INT8_TYPE__ signed char
// X86_64:#define __INTMAX_C_SUFFIX__ L
// X86_64:#define __INTMAX_FMTd__ "ld"
// X86_64:#define __INTMAX_FMTi__ "li"
// X86_64:#define __INTMAX_MAX__ 9223372036854775807L
// X86_64:#define __INTMAX_TYPE__ long int
// X86_64:#define __INTMAX_WIDTH__ 64
// X86_64:#define __INTPTR_FMTd__ "ld"
// X86_64:#define __INTPTR_FMTi__ "li"
// X86_64:#define __INTPTR_MAX__ 9223372036854775807L
// X86_64:#define __INTPTR_TYPE__ long int
// X86_64:#define __INTPTR_WIDTH__ 64
// X86_64:#define __INT_FAST16_FMTd__ "hd"
// X86_64:#define __INT_FAST16_FMTi__ "hi"
// X86_64:#define __INT_FAST16_MAX__ 32767
// X86_64:#define __INT_FAST16_TYPE__ short
// X86_64:#define __INT_FAST32_FMTd__ "d"
// X86_64:#define __INT_FAST32_FMTi__ "i"
// X86_64:#define __INT_FAST32_MAX__ 2147483647
// X86_64:#define __INT_FAST32_TYPE__ int
// X86_64:#define __INT_FAST64_FMTd__ "ld"
// X86_64:#define __INT_FAST64_FMTi__ "li"
// X86_64:#define __INT_FAST64_MAX__ 9223372036854775807L
// X86_64:#define __INT_FAST64_TYPE__ long int
// X86_64:#define __INT_FAST8_FMTd__ "hhd"
// X86_64:#define __INT_FAST8_FMTi__ "hhi"
// X86_64:#define __INT_FAST8_MAX__ 127
// X86_64:#define __INT_FAST8_TYPE__ signed char
// X86_64:#define __INT_LEAST16_FMTd__ "hd"
// X86_64:#define __INT_LEAST16_FMTi__ "hi"
// X86_64:#define __INT_LEAST16_MAX__ 32767
// X86_64:#define __INT_LEAST16_TYPE__ short
// X86_64:#define __INT_LEAST32_FMTd__ "d"
// X86_64:#define __INT_LEAST32_FMTi__ "i"
// X86_64:#define __INT_LEAST32_MAX__ 2147483647
// X86_64:#define __INT_LEAST32_TYPE__ int
// X86_64:#define __INT_LEAST64_FMTd__ "ld"
// X86_64:#define __INT_LEAST64_FMTi__ "li"
// X86_64:#define __INT_LEAST64_MAX__ 9223372036854775807L
// X86_64:#define __INT_LEAST64_TYPE__ long int
// X86_64:#define __INT_LEAST8_FMTd__ "hhd"
// X86_64:#define __INT_LEAST8_FMTi__ "hhi"
// X86_64:#define __INT_LEAST8_MAX__ 127
// X86_64:#define __INT_LEAST8_TYPE__ signed char
// X86_64:#define __INT_MAX__ 2147483647
// X86_64:#define __LDBL_DENORM_MIN__ 3.64519953188247460253e-4951L
// X86_64:#define __LDBL_DIG__ 18
// X86_64:#define __LDBL_EPSILON__ 1.08420217248550443401e-19L
// X86_64:#define __LDBL_HAS_DENORM__ 1
// X86_64:#define __LDBL_HAS_INFINITY__ 1
// X86_64:#define __LDBL_HAS_QUIET_NAN__ 1
// X86_64:#define __LDBL_MANT_DIG__ 64
// X86_64:#define __LDBL_MAX_10_EXP__ 4932
// X86_64:#define __LDBL_MAX_EXP__ 16384
// X86_64:#define __LDBL_MAX__ 1.18973149535723176502e+4932L
// X86_64:#define __LDBL_MIN_10_EXP__ (-4931)
// X86_64:#define __LDBL_MIN_EXP__ (-16381)
// X86_64:#define __LDBL_MIN__ 3.36210314311209350626e-4932L
// X86_64:#define __LITTLE_ENDIAN__ 1
// X86_64:#define __LONG_LONG_MAX__ 9223372036854775807LL
// X86_64:#define __LONG_MAX__ 9223372036854775807L
// X86_64:#define __LP64__ 1
// X86_64-NOT:#define __ILP32__ 1
// X86_64:#define __MMX__ 1
// X86_64:#define __NO_MATH_INLINES 1
// X86_64:#define __POINTER_WIDTH__ 64
// X86_64:#define __PTRDIFF_TYPE__ long int
// X86_64:#define __PTRDIFF_WIDTH__ 64
// X86_64:#define __REGISTER_PREFIX__
// X86_64:#define __SCHAR_MAX__ 127
// X86_64:#define __SHRT_MAX__ 32767
// X86_64:#define __SIG_ATOMIC_MAX__ 2147483647
// X86_64:#define __SIG_ATOMIC_WIDTH__ 32
// X86_64:#define __SIZEOF_DOUBLE__ 8
// X86_64:#define __SIZEOF_FLOAT__ 4
// X86_64:#define __SIZEOF_INT__ 4
// X86_64:#define __SIZEOF_LONG_DOUBLE__ 16
// X86_64:#define __SIZEOF_LONG_LONG__ 8
// X86_64:#define __SIZEOF_LONG__ 8
// X86_64:#define __SIZEOF_POINTER__ 8
// X86_64:#define __SIZEOF_PTRDIFF_T__ 8
// X86_64:#define __SIZEOF_SHORT__ 2
// X86_64:#define __SIZEOF_SIZE_T__ 8
// X86_64:#define __SIZEOF_WCHAR_T__ 4
// X86_64:#define __SIZEOF_WINT_T__ 4
// X86_64:#define __SIZE_MAX__ 18446744073709551615UL
// X86_64:#define __SIZE_TYPE__ long unsigned int
// X86_64:#define __SIZE_WIDTH__ 64
// X86_64:#define __SSE2_MATH__ 1
// X86_64:#define __SSE2__ 1
// X86_64:#define __SSE_MATH__ 1
// X86_64:#define __SSE__ 1
// X86_64-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 16UL
// X86_64:#define __UINT16_C_SUFFIX__
// X86_64:#define __UINT16_MAX__ 65535
// X86_64:#define __UINT16_TYPE__ unsigned short
// X86_64:#define __UINT32_C_SUFFIX__ U
// X86_64:#define __UINT32_MAX__ 4294967295U
// X86_64:#define __UINT32_TYPE__ unsigned int
// X86_64:#define __UINT64_C_SUFFIX__ UL
// X86_64:#define __UINT64_MAX__ 18446744073709551615UL
// X86_64:#define __UINT64_TYPE__ long unsigned int
// X86_64:#define __UINT8_C_SUFFIX__
// X86_64:#define __UINT8_MAX__ 255
// X86_64:#define __UINT8_TYPE__ unsigned char
// X86_64:#define __UINTMAX_C_SUFFIX__ UL
// X86_64:#define __UINTMAX_MAX__ 18446744073709551615UL
// X86_64:#define __UINTMAX_TYPE__ long unsigned int
// X86_64:#define __UINTMAX_WIDTH__ 64
// X86_64:#define __UINTPTR_MAX__ 18446744073709551615UL
// X86_64:#define __UINTPTR_TYPE__ long unsigned int
// X86_64:#define __UINTPTR_WIDTH__ 64
// X86_64:#define __UINT_FAST16_MAX__ 65535
// X86_64:#define __UINT_FAST16_TYPE__ unsigned short
// X86_64:#define __UINT_FAST32_MAX__ 4294967295U
// X86_64:#define __UINT_FAST32_TYPE__ unsigned int
// X86_64:#define __UINT_FAST64_MAX__ 18446744073709551615UL
// X86_64:#define __UINT_FAST64_TYPE__ long unsigned int
// X86_64:#define __UINT_FAST8_MAX__ 255
// X86_64:#define __UINT_FAST8_TYPE__ unsigned char
// X86_64:#define __UINT_LEAST16_MAX__ 65535
// X86_64:#define __UINT_LEAST16_TYPE__ unsigned short
// X86_64:#define __UINT_LEAST32_MAX__ 4294967295U
// X86_64:#define __UINT_LEAST32_TYPE__ unsigned int
// X86_64:#define __UINT_LEAST64_MAX__ 18446744073709551615UL
// X86_64:#define __UINT_LEAST64_TYPE__ long unsigned int
// X86_64:#define __UINT_LEAST8_MAX__ 255
// X86_64:#define __UINT_LEAST8_TYPE__ unsigned char
// X86_64:#define __USER_LABEL_PREFIX__
// X86_64:#define __WCHAR_MAX__ 2147483647
// X86_64:#define __WCHAR_TYPE__ int
// X86_64:#define __WCHAR_WIDTH__ 32
// X86_64:#define __WINT_TYPE__ int
// X86_64:#define __WINT_WIDTH__ 32
// X86_64:#define __amd64 1
// X86_64:#define __amd64__ 1
// X86_64:#define __x86_64 1
// X86_64:#define __x86_64__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=x86_64h-none-none < /dev/null | FileCheck -match-full-lines -check-prefix X86_64H %s
//
// X86_64H:#define __x86_64 1
// X86_64H:#define __x86_64__ 1
// X86_64H:#define __x86_64h 1
// X86_64H:#define __x86_64h__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=x86_64-none-none-gnux32 < /dev/null | FileCheck -match-full-lines -check-prefix X32 %s
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=x86_64-none-none-gnux32 < /dev/null | FileCheck -match-full-lines -check-prefix X32 -check-prefix X32-CXX %s
//
// X32:#define _ILP32 1
// X32-NOT:#define _LP64 1
// X32:#define __BIGGEST_ALIGNMENT__ 16
// X32:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// X32:#define __CHAR16_TYPE__ unsigned short
// X32:#define __CHAR32_TYPE__ unsigned int
// X32:#define __CHAR_BIT__ 8
// X32:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// X32:#define __DBL_DIG__ 15
// X32:#define __DBL_EPSILON__ 2.2204460492503131e-16
// X32:#define __DBL_HAS_DENORM__ 1
// X32:#define __DBL_HAS_INFINITY__ 1
// X32:#define __DBL_HAS_QUIET_NAN__ 1
// X32:#define __DBL_MANT_DIG__ 53
// X32:#define __DBL_MAX_10_EXP__ 308
// X32:#define __DBL_MAX_EXP__ 1024
// X32:#define __DBL_MAX__ 1.7976931348623157e+308
// X32:#define __DBL_MIN_10_EXP__ (-307)
// X32:#define __DBL_MIN_EXP__ (-1021)
// X32:#define __DBL_MIN__ 2.2250738585072014e-308
// X32:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// X32:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// X32:#define __FLT_DIG__ 6
// X32:#define __FLT_EPSILON__ 1.19209290e-7F
// X32:#define __FLT_EVAL_METHOD__ 0
// X32:#define __FLT_HAS_DENORM__ 1
// X32:#define __FLT_HAS_INFINITY__ 1
// X32:#define __FLT_HAS_QUIET_NAN__ 1
// X32:#define __FLT_MANT_DIG__ 24
// X32:#define __FLT_MAX_10_EXP__ 38
// X32:#define __FLT_MAX_EXP__ 128
// X32:#define __FLT_MAX__ 3.40282347e+38F
// X32:#define __FLT_MIN_10_EXP__ (-37)
// X32:#define __FLT_MIN_EXP__ (-125)
// X32:#define __FLT_MIN__ 1.17549435e-38F
// X32:#define __FLT_RADIX__ 2
// X32:#define __ILP32__ 1
// X32-NOT:#define __LP64__ 1
// X32:#define __INT16_C_SUFFIX__
// X32:#define __INT16_FMTd__ "hd"
// X32:#define __INT16_FMTi__ "hi"
// X32:#define __INT16_MAX__ 32767
// X32:#define __INT16_TYPE__ short
// X32:#define __INT32_C_SUFFIX__
// X32:#define __INT32_FMTd__ "d"
// X32:#define __INT32_FMTi__ "i"
// X32:#define __INT32_MAX__ 2147483647
// X32:#define __INT32_TYPE__ int
// X32:#define __INT64_C_SUFFIX__ LL
// X32:#define __INT64_FMTd__ "lld"
// X32:#define __INT64_FMTi__ "lli"
// X32:#define __INT64_MAX__ 9223372036854775807LL
// X32:#define __INT64_TYPE__ long long int
// X32:#define __INT8_C_SUFFIX__
// X32:#define __INT8_FMTd__ "hhd"
// X32:#define __INT8_FMTi__ "hhi"
// X32:#define __INT8_MAX__ 127
// X32:#define __INT8_TYPE__ signed char
// X32:#define __INTMAX_C_SUFFIX__ LL
// X32:#define __INTMAX_FMTd__ "lld"
// X32:#define __INTMAX_FMTi__ "lli"
// X32:#define __INTMAX_MAX__ 9223372036854775807LL
// X32:#define __INTMAX_TYPE__ long long int
// X32:#define __INTMAX_WIDTH__ 64
// X32:#define __INTPTR_FMTd__ "d"
// X32:#define __INTPTR_FMTi__ "i"
// X32:#define __INTPTR_MAX__ 2147483647
// X32:#define __INTPTR_TYPE__ int
// X32:#define __INTPTR_WIDTH__ 32
// X32:#define __INT_FAST16_FMTd__ "hd"
// X32:#define __INT_FAST16_FMTi__ "hi"
// X32:#define __INT_FAST16_MAX__ 32767
// X32:#define __INT_FAST16_TYPE__ short
// X32:#define __INT_FAST32_FMTd__ "d"
// X32:#define __INT_FAST32_FMTi__ "i"
// X32:#define __INT_FAST32_MAX__ 2147483647
// X32:#define __INT_FAST32_TYPE__ int
// X32:#define __INT_FAST64_FMTd__ "lld"
// X32:#define __INT_FAST64_FMTi__ "lli"
// X32:#define __INT_FAST64_MAX__ 9223372036854775807LL
// X32:#define __INT_FAST64_TYPE__ long long int
// X32:#define __INT_FAST8_FMTd__ "hhd"
// X32:#define __INT_FAST8_FMTi__ "hhi"
// X32:#define __INT_FAST8_MAX__ 127
// X32:#define __INT_FAST8_TYPE__ signed char
// X32:#define __INT_LEAST16_FMTd__ "hd"
// X32:#define __INT_LEAST16_FMTi__ "hi"
// X32:#define __INT_LEAST16_MAX__ 32767
// X32:#define __INT_LEAST16_TYPE__ short
// X32:#define __INT_LEAST32_FMTd__ "d"
// X32:#define __INT_LEAST32_FMTi__ "i"
// X32:#define __INT_LEAST32_MAX__ 2147483647
// X32:#define __INT_LEAST32_TYPE__ int
// X32:#define __INT_LEAST64_FMTd__ "lld"
// X32:#define __INT_LEAST64_FMTi__ "lli"
// X32:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// X32:#define __INT_LEAST64_TYPE__ long long int
// X32:#define __INT_LEAST8_FMTd__ "hhd"
// X32:#define __INT_LEAST8_FMTi__ "hhi"
// X32:#define __INT_LEAST8_MAX__ 127
// X32:#define __INT_LEAST8_TYPE__ signed char
// X32:#define __INT_MAX__ 2147483647
// X32:#define __LDBL_DENORM_MIN__ 3.64519953188247460253e-4951L
// X32:#define __LDBL_DIG__ 18
// X32:#define __LDBL_EPSILON__ 1.08420217248550443401e-19L
// X32:#define __LDBL_HAS_DENORM__ 1
// X32:#define __LDBL_HAS_INFINITY__ 1
// X32:#define __LDBL_HAS_QUIET_NAN__ 1
// X32:#define __LDBL_MANT_DIG__ 64
// X32:#define __LDBL_MAX_10_EXP__ 4932
// X32:#define __LDBL_MAX_EXP__ 16384
// X32:#define __LDBL_MAX__ 1.18973149535723176502e+4932L
// X32:#define __LDBL_MIN_10_EXP__ (-4931)
// X32:#define __LDBL_MIN_EXP__ (-16381)
// X32:#define __LDBL_MIN__ 3.36210314311209350626e-4932L
// X32:#define __LITTLE_ENDIAN__ 1
// X32:#define __LONG_LONG_MAX__ 9223372036854775807LL
// X32:#define __LONG_MAX__ 2147483647L
// X32:#define __MMX__ 1
// X32:#define __NO_MATH_INLINES 1
// X32:#define __POINTER_WIDTH__ 32
// X32:#define __PTRDIFF_TYPE__ int
// X32:#define __PTRDIFF_WIDTH__ 32
// X32:#define __REGISTER_PREFIX__
// X32:#define __SCHAR_MAX__ 127
// X32:#define __SHRT_MAX__ 32767
// X32:#define __SIG_ATOMIC_MAX__ 2147483647
// X32:#define __SIG_ATOMIC_WIDTH__ 32
// X32:#define __SIZEOF_DOUBLE__ 8
// X32:#define __SIZEOF_FLOAT__ 4
// X32:#define __SIZEOF_INT__ 4
// X32:#define __SIZEOF_LONG_DOUBLE__ 16
// X32:#define __SIZEOF_LONG_LONG__ 8
// X32:#define __SIZEOF_LONG__ 4
// X32:#define __SIZEOF_POINTER__ 4
// X32:#define __SIZEOF_PTRDIFF_T__ 4
// X32:#define __SIZEOF_SHORT__ 2
// X32:#define __SIZEOF_SIZE_T__ 4
// X32:#define __SIZEOF_WCHAR_T__ 4
// X32:#define __SIZEOF_WINT_T__ 4
// X32:#define __SIZE_MAX__ 4294967295U
// X32:#define __SIZE_TYPE__ unsigned int
// X32:#define __SIZE_WIDTH__ 32
// X32:#define __SSE2_MATH__ 1
// X32:#define __SSE2__ 1
// X32:#define __SSE_MATH__ 1
// X32:#define __SSE__ 1
// X32-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 16U
// X32:#define __UINT16_C_SUFFIX__
// X32:#define __UINT16_MAX__ 65535
// X32:#define __UINT16_TYPE__ unsigned short
// X32:#define __UINT32_C_SUFFIX__ U
// X32:#define __UINT32_MAX__ 4294967295U
// X32:#define __UINT32_TYPE__ unsigned int
// X32:#define __UINT64_C_SUFFIX__ ULL
// X32:#define __UINT64_MAX__ 18446744073709551615ULL
// X32:#define __UINT64_TYPE__ long long unsigned int
// X32:#define __UINT8_C_SUFFIX__
// X32:#define __UINT8_MAX__ 255
// X32:#define __UINT8_TYPE__ unsigned char
// X32:#define __UINTMAX_C_SUFFIX__ ULL
// X32:#define __UINTMAX_MAX__ 18446744073709551615ULL
// X32:#define __UINTMAX_TYPE__ long long unsigned int
// X32:#define __UINTMAX_WIDTH__ 64
// X32:#define __UINTPTR_MAX__ 4294967295U
// X32:#define __UINTPTR_TYPE__ unsigned int
// X32:#define __UINTPTR_WIDTH__ 32
// X32:#define __UINT_FAST16_MAX__ 65535
// X32:#define __UINT_FAST16_TYPE__ unsigned short
// X32:#define __UINT_FAST32_MAX__ 4294967295U
// X32:#define __UINT_FAST32_TYPE__ unsigned int
// X32:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// X32:#define __UINT_FAST64_TYPE__ long long unsigned int
// X32:#define __UINT_FAST8_MAX__ 255
// X32:#define __UINT_FAST8_TYPE__ unsigned char
// X32:#define __UINT_LEAST16_MAX__ 65535
// X32:#define __UINT_LEAST16_TYPE__ unsigned short
// X32:#define __UINT_LEAST32_MAX__ 4294967295U
// X32:#define __UINT_LEAST32_TYPE__ unsigned int
// X32:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// X32:#define __UINT_LEAST64_TYPE__ long long unsigned int
// X32:#define __UINT_LEAST8_MAX__ 255
// X32:#define __UINT_LEAST8_TYPE__ unsigned char
// X32:#define __USER_LABEL_PREFIX__
// X32:#define __WCHAR_MAX__ 2147483647
// X32:#define __WCHAR_TYPE__ int
// X32:#define __WCHAR_WIDTH__ 32
// X32:#define __WINT_TYPE__ int
// X32:#define __WINT_WIDTH__ 32
// X32:#define __amd64 1
// X32:#define __amd64__ 1
// X32:#define __x86_64 1
// X32:#define __x86_64__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=x86_64-unknown-cloudabi < /dev/null | FileCheck -match-full-lines -check-prefix X86_64-CLOUDABI %s
//
// X86_64-CLOUDABI:#define _LP64 1
// X86_64-CLOUDABI:#define __ATOMIC_ACQUIRE 2
// X86_64-CLOUDABI:#define __ATOMIC_ACQ_REL 4
// X86_64-CLOUDABI:#define __ATOMIC_CONSUME 1
// X86_64-CLOUDABI:#define __ATOMIC_RELAXED 0
// X86_64-CLOUDABI:#define __ATOMIC_RELEASE 3
// X86_64-CLOUDABI:#define __ATOMIC_SEQ_CST 5
// X86_64-CLOUDABI:#define __BIGGEST_ALIGNMENT__ 16
// X86_64-CLOUDABI:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// X86_64-CLOUDABI:#define __CHAR16_TYPE__ unsigned short
// X86_64-CLOUDABI:#define __CHAR32_TYPE__ unsigned int
// X86_64-CLOUDABI:#define __CHAR_BIT__ 8
// X86_64-CLOUDABI:#define __CONSTANT_CFSTRINGS__ 1
// X86_64-CLOUDABI:#define __CloudABI__ 1
// X86_64-CLOUDABI:#define __DBL_DECIMAL_DIG__ 17
// X86_64-CLOUDABI:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// X86_64-CLOUDABI:#define __DBL_DIG__ 15
// X86_64-CLOUDABI:#define __DBL_EPSILON__ 2.2204460492503131e-16
// X86_64-CLOUDABI:#define __DBL_HAS_DENORM__ 1
// X86_64-CLOUDABI:#define __DBL_HAS_INFINITY__ 1
// X86_64-CLOUDABI:#define __DBL_HAS_QUIET_NAN__ 1
// X86_64-CLOUDABI:#define __DBL_MANT_DIG__ 53
// X86_64-CLOUDABI:#define __DBL_MAX_10_EXP__ 308
// X86_64-CLOUDABI:#define __DBL_MAX_EXP__ 1024
// X86_64-CLOUDABI:#define __DBL_MAX__ 1.7976931348623157e+308
// X86_64-CLOUDABI:#define __DBL_MIN_10_EXP__ (-307)
// X86_64-CLOUDABI:#define __DBL_MIN_EXP__ (-1021)
// X86_64-CLOUDABI:#define __DBL_MIN__ 2.2250738585072014e-308
// X86_64-CLOUDABI:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// X86_64-CLOUDABI:#define __ELF__ 1
// X86_64-CLOUDABI:#define __FINITE_MATH_ONLY__ 0
// X86_64-CLOUDABI:#define __FLT_DECIMAL_DIG__ 9
// X86_64-CLOUDABI:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// X86_64-CLOUDABI:#define __FLT_DIG__ 6
// X86_64-CLOUDABI:#define __FLT_EPSILON__ 1.19209290e-7F
// X86_64-CLOUDABI:#define __FLT_EVAL_METHOD__ 0
// X86_64-CLOUDABI:#define __FLT_HAS_DENORM__ 1
// X86_64-CLOUDABI:#define __FLT_HAS_INFINITY__ 1
// X86_64-CLOUDABI:#define __FLT_HAS_QUIET_NAN__ 1
// X86_64-CLOUDABI:#define __FLT_MANT_DIG__ 24
// X86_64-CLOUDABI:#define __FLT_MAX_10_EXP__ 38
// X86_64-CLOUDABI:#define __FLT_MAX_EXP__ 128
// X86_64-CLOUDABI:#define __FLT_MAX__ 3.40282347e+38F
// X86_64-CLOUDABI:#define __FLT_MIN_10_EXP__ (-37)
// X86_64-CLOUDABI:#define __FLT_MIN_EXP__ (-125)
// X86_64-CLOUDABI:#define __FLT_MIN__ 1.17549435e-38F
// X86_64-CLOUDABI:#define __FLT_RADIX__ 2
// X86_64-CLOUDABI:#define __GCC_ATOMIC_BOOL_LOCK_FREE 2
// X86_64-CLOUDABI:#define __GCC_ATOMIC_CHAR16_T_LOCK_FREE 2
// X86_64-CLOUDABI:#define __GCC_ATOMIC_CHAR32_T_LOCK_FREE 2
// X86_64-CLOUDABI:#define __GCC_ATOMIC_CHAR_LOCK_FREE 2
// X86_64-CLOUDABI:#define __GCC_ATOMIC_INT_LOCK_FREE 2
// X86_64-CLOUDABI:#define __GCC_ATOMIC_LLONG_LOCK_FREE 2
// X86_64-CLOUDABI:#define __GCC_ATOMIC_LONG_LOCK_FREE 2
// X86_64-CLOUDABI:#define __GCC_ATOMIC_POINTER_LOCK_FREE 2
// X86_64-CLOUDABI:#define __GCC_ATOMIC_SHORT_LOCK_FREE 2
// X86_64-CLOUDABI:#define __GCC_ATOMIC_TEST_AND_SET_TRUEVAL 1
// X86_64-CLOUDABI:#define __GCC_ATOMIC_WCHAR_T_LOCK_FREE 2
// X86_64-CLOUDABI:#define __GNUC_MINOR__ 2
// X86_64-CLOUDABI:#define __GNUC_PATCHLEVEL__ 1
// X86_64-CLOUDABI:#define __GNUC_STDC_INLINE__ 1
// X86_64-CLOUDABI:#define __GNUC__ 4
// X86_64-CLOUDABI:#define __GXX_ABI_VERSION 1002
// X86_64-CLOUDABI:#define __INT16_C_SUFFIX__
// X86_64-CLOUDABI:#define __INT16_FMTd__ "hd"
// X86_64-CLOUDABI:#define __INT16_FMTi__ "hi"
// X86_64-CLOUDABI:#define __INT16_MAX__ 32767
// X86_64-CLOUDABI:#define __INT16_TYPE__ short
// X86_64-CLOUDABI:#define __INT32_C_SUFFIX__
// X86_64-CLOUDABI:#define __INT32_FMTd__ "d"
// X86_64-CLOUDABI:#define __INT32_FMTi__ "i"
// X86_64-CLOUDABI:#define __INT32_MAX__ 2147483647
// X86_64-CLOUDABI:#define __INT32_TYPE__ int
// X86_64-CLOUDABI:#define __INT64_C_SUFFIX__ L
// X86_64-CLOUDABI:#define __INT64_FMTd__ "ld"
// X86_64-CLOUDABI:#define __INT64_FMTi__ "li"
// X86_64-CLOUDABI:#define __INT64_MAX__ 9223372036854775807L
// X86_64-CLOUDABI:#define __INT64_TYPE__ long int
// X86_64-CLOUDABI:#define __INT8_C_SUFFIX__
// X86_64-CLOUDABI:#define __INT8_FMTd__ "hhd"
// X86_64-CLOUDABI:#define __INT8_FMTi__ "hhi"
// X86_64-CLOUDABI:#define __INT8_MAX__ 127
// X86_64-CLOUDABI:#define __INT8_TYPE__ signed char
// X86_64-CLOUDABI:#define __INTMAX_C_SUFFIX__ L
// X86_64-CLOUDABI:#define __INTMAX_FMTd__ "ld"
// X86_64-CLOUDABI:#define __INTMAX_FMTi__ "li"
// X86_64-CLOUDABI:#define __INTMAX_MAX__ 9223372036854775807L
// X86_64-CLOUDABI:#define __INTMAX_TYPE__ long int
// X86_64-CLOUDABI:#define __INTMAX_WIDTH__ 64
// X86_64-CLOUDABI:#define __INTPTR_FMTd__ "ld"
// X86_64-CLOUDABI:#define __INTPTR_FMTi__ "li"
// X86_64-CLOUDABI:#define __INTPTR_MAX__ 9223372036854775807L
// X86_64-CLOUDABI:#define __INTPTR_TYPE__ long int
// X86_64-CLOUDABI:#define __INTPTR_WIDTH__ 64
// X86_64-CLOUDABI:#define __INT_FAST16_FMTd__ "hd"
// X86_64-CLOUDABI:#define __INT_FAST16_FMTi__ "hi"
// X86_64-CLOUDABI:#define __INT_FAST16_MAX__ 32767
// X86_64-CLOUDABI:#define __INT_FAST16_TYPE__ short
// X86_64-CLOUDABI:#define __INT_FAST32_FMTd__ "d"
// X86_64-CLOUDABI:#define __INT_FAST32_FMTi__ "i"
// X86_64-CLOUDABI:#define __INT_FAST32_MAX__ 2147483647
// X86_64-CLOUDABI:#define __INT_FAST32_TYPE__ int
// X86_64-CLOUDABI:#define __INT_FAST64_FMTd__ "ld"
// X86_64-CLOUDABI:#define __INT_FAST64_FMTi__ "li"
// X86_64-CLOUDABI:#define __INT_FAST64_MAX__ 9223372036854775807L
// X86_64-CLOUDABI:#define __INT_FAST64_TYPE__ long int
// X86_64-CLOUDABI:#define __INT_FAST8_FMTd__ "hhd"
// X86_64-CLOUDABI:#define __INT_FAST8_FMTi__ "hhi"
// X86_64-CLOUDABI:#define __INT_FAST8_MAX__ 127
// X86_64-CLOUDABI:#define __INT_FAST8_TYPE__ signed char
// X86_64-CLOUDABI:#define __INT_LEAST16_FMTd__ "hd"
// X86_64-CLOUDABI:#define __INT_LEAST16_FMTi__ "hi"
// X86_64-CLOUDABI:#define __INT_LEAST16_MAX__ 32767
// X86_64-CLOUDABI:#define __INT_LEAST16_TYPE__ short
// X86_64-CLOUDABI:#define __INT_LEAST32_FMTd__ "d"
// X86_64-CLOUDABI:#define __INT_LEAST32_FMTi__ "i"
// X86_64-CLOUDABI:#define __INT_LEAST32_MAX__ 2147483647
// X86_64-CLOUDABI:#define __INT_LEAST32_TYPE__ int
// X86_64-CLOUDABI:#define __INT_LEAST64_FMTd__ "ld"
// X86_64-CLOUDABI:#define __INT_LEAST64_FMTi__ "li"
// X86_64-CLOUDABI:#define __INT_LEAST64_MAX__ 9223372036854775807L
// X86_64-CLOUDABI:#define __INT_LEAST64_TYPE__ long int
// X86_64-CLOUDABI:#define __INT_LEAST8_FMTd__ "hhd"
// X86_64-CLOUDABI:#define __INT_LEAST8_FMTi__ "hhi"
// X86_64-CLOUDABI:#define __INT_LEAST8_MAX__ 127
// X86_64-CLOUDABI:#define __INT_LEAST8_TYPE__ signed char
// X86_64-CLOUDABI:#define __INT_MAX__ 2147483647
// X86_64-CLOUDABI:#define __LDBL_DECIMAL_DIG__ 21
// X86_64-CLOUDABI:#define __LDBL_DENORM_MIN__ 3.64519953188247460253e-4951L
// X86_64-CLOUDABI:#define __LDBL_DIG__ 18
// X86_64-CLOUDABI:#define __LDBL_EPSILON__ 1.08420217248550443401e-19L
// X86_64-CLOUDABI:#define __LDBL_HAS_DENORM__ 1
// X86_64-CLOUDABI:#define __LDBL_HAS_INFINITY__ 1
// X86_64-CLOUDABI:#define __LDBL_HAS_QUIET_NAN__ 1
// X86_64-CLOUDABI:#define __LDBL_MANT_DIG__ 64
// X86_64-CLOUDABI:#define __LDBL_MAX_10_EXP__ 4932
// X86_64-CLOUDABI:#define __LDBL_MAX_EXP__ 16384
// X86_64-CLOUDABI:#define __LDBL_MAX__ 1.18973149535723176502e+4932L
// X86_64-CLOUDABI:#define __LDBL_MIN_10_EXP__ (-4931)
// X86_64-CLOUDABI:#define __LDBL_MIN_EXP__ (-16381)
// X86_64-CLOUDABI:#define __LDBL_MIN__ 3.36210314311209350626e-4932L
// X86_64-CLOUDABI:#define __LITTLE_ENDIAN__ 1
// X86_64-CLOUDABI:#define __LONG_LONG_MAX__ 9223372036854775807LL
// X86_64-CLOUDABI:#define __LONG_MAX__ 9223372036854775807L
// X86_64-CLOUDABI:#define __LP64__ 1
// X86_64-CLOUDABI:#define __MMX__ 1
// X86_64-CLOUDABI:#define __NO_INLINE__ 1
// X86_64-CLOUDABI:#define __NO_MATH_INLINES 1
// X86_64-CLOUDABI:#define __ORDER_BIG_ENDIAN__ 4321
// X86_64-CLOUDABI:#define __ORDER_LITTLE_ENDIAN__ 1234
// X86_64-CLOUDABI:#define __ORDER_PDP_ENDIAN__ 3412
// X86_64-CLOUDABI:#define __POINTER_WIDTH__ 64
// X86_64-CLOUDABI:#define __PRAGMA_REDEFINE_EXTNAME 1
// X86_64-CLOUDABI:#define __PTRDIFF_FMTd__ "ld"
// X86_64-CLOUDABI:#define __PTRDIFF_FMTi__ "li"
// X86_64-CLOUDABI:#define __PTRDIFF_MAX__ 9223372036854775807L
// X86_64-CLOUDABI:#define __PTRDIFF_TYPE__ long int
// X86_64-CLOUDABI:#define __PTRDIFF_WIDTH__ 64
// X86_64-CLOUDABI:#define __REGISTER_PREFIX__
// X86_64-CLOUDABI:#define __SCHAR_MAX__ 127
// X86_64-CLOUDABI:#define __SHRT_MAX__ 32767
// X86_64-CLOUDABI:#define __SIG_ATOMIC_MAX__ 2147483647
// X86_64-CLOUDABI:#define __SIG_ATOMIC_WIDTH__ 32
// X86_64-CLOUDABI:#define __SIZEOF_DOUBLE__ 8
// X86_64-CLOUDABI:#define __SIZEOF_FLOAT__ 4
// X86_64-CLOUDABI:#define __SIZEOF_INT128__ 16
// X86_64-CLOUDABI:#define __SIZEOF_INT__ 4
// X86_64-CLOUDABI:#define __SIZEOF_LONG_DOUBLE__ 16
// X86_64-CLOUDABI:#define __SIZEOF_LONG_LONG__ 8
// X86_64-CLOUDABI:#define __SIZEOF_LONG__ 8
// X86_64-CLOUDABI:#define __SIZEOF_POINTER__ 8
// X86_64-CLOUDABI:#define __SIZEOF_PTRDIFF_T__ 8
// X86_64-CLOUDABI:#define __SIZEOF_SHORT__ 2
// X86_64-CLOUDABI:#define __SIZEOF_SIZE_T__ 8
// X86_64-CLOUDABI:#define __SIZEOF_WCHAR_T__ 4
// X86_64-CLOUDABI:#define __SIZEOF_WINT_T__ 4
// X86_64-CLOUDABI:#define __SIZE_FMTX__ "lX"
// X86_64-CLOUDABI:#define __SIZE_FMTo__ "lo"
// X86_64-CLOUDABI:#define __SIZE_FMTu__ "lu"
// X86_64-CLOUDABI:#define __SIZE_FMTx__ "lx"
// X86_64-CLOUDABI:#define __SIZE_MAX__ 18446744073709551615UL
// X86_64-CLOUDABI:#define __SIZE_TYPE__ long unsigned int
// X86_64-CLOUDABI:#define __SIZE_WIDTH__ 64
// X86_64-CLOUDABI:#define __SSE2_MATH__ 1
// X86_64-CLOUDABI:#define __SSE2__ 1
// X86_64-CLOUDABI:#define __SSE_MATH__ 1
// X86_64-CLOUDABI:#define __SSE__ 1
// X86_64-CLOUDABI:#define __STDC_HOSTED__ 0
// X86_64-CLOUDABI:#define __STDC_ISO_10646__ 201206L
// X86_64-CLOUDABI:#define __STDC_UTF_16__ 1
// X86_64-CLOUDABI:#define __STDC_UTF_32__ 1
// X86_64-CLOUDABI:#define __STDC_VERSION__ 201112L
// X86_64-CLOUDABI:#define __STDC__ 1
// X86_64-CLOUDABI:#define __UINT16_C_SUFFIX__
// X86_64-CLOUDABI:#define __UINT16_FMTX__ "hX"
// X86_64-CLOUDABI:#define __UINT16_FMTo__ "ho"
// X86_64-CLOUDABI:#define __UINT16_FMTu__ "hu"
// X86_64-CLOUDABI:#define __UINT16_FMTx__ "hx"
// X86_64-CLOUDABI:#define __UINT16_MAX__ 65535
// X86_64-CLOUDABI:#define __UINT16_TYPE__ unsigned short
// X86_64-CLOUDABI:#define __UINT32_C_SUFFIX__ U
// X86_64-CLOUDABI:#define __UINT32_FMTX__ "X"
// X86_64-CLOUDABI:#define __UINT32_FMTo__ "o"
// X86_64-CLOUDABI:#define __UINT32_FMTu__ "u"
// X86_64-CLOUDABI:#define __UINT32_FMTx__ "x"
// X86_64-CLOUDABI:#define __UINT32_MAX__ 4294967295U
// X86_64-CLOUDABI:#define __UINT32_TYPE__ unsigned int
// X86_64-CLOUDABI:#define __UINT64_C_SUFFIX__ UL
// X86_64-CLOUDABI:#define __UINT64_FMTX__ "lX"
// X86_64-CLOUDABI:#define __UINT64_FMTo__ "lo"
// X86_64-CLOUDABI:#define __UINT64_FMTu__ "lu"
// X86_64-CLOUDABI:#define __UINT64_FMTx__ "lx"
// X86_64-CLOUDABI:#define __UINT64_MAX__ 18446744073709551615UL
// X86_64-CLOUDABI:#define __UINT64_TYPE__ long unsigned int
// X86_64-CLOUDABI:#define __UINT8_C_SUFFIX__
// X86_64-CLOUDABI:#define __UINT8_FMTX__ "hhX"
// X86_64-CLOUDABI:#define __UINT8_FMTo__ "hho"
// X86_64-CLOUDABI:#define __UINT8_FMTu__ "hhu"
// X86_64-CLOUDABI:#define __UINT8_FMTx__ "hhx"
// X86_64-CLOUDABI:#define __UINT8_MAX__ 255
// X86_64-CLOUDABI:#define __UINT8_TYPE__ unsigned char
// X86_64-CLOUDABI:#define __UINTMAX_C_SUFFIX__ UL
// X86_64-CLOUDABI:#define __UINTMAX_FMTX__ "lX"
// X86_64-CLOUDABI:#define __UINTMAX_FMTo__ "lo"
// X86_64-CLOUDABI:#define __UINTMAX_FMTu__ "lu"
// X86_64-CLOUDABI:#define __UINTMAX_FMTx__ "lx"
// X86_64-CLOUDABI:#define __UINTMAX_MAX__ 18446744073709551615UL
// X86_64-CLOUDABI:#define __UINTMAX_TYPE__ long unsigned int
// X86_64-CLOUDABI:#define __UINTMAX_WIDTH__ 64
// X86_64-CLOUDABI:#define __UINTPTR_FMTX__ "lX"
// X86_64-CLOUDABI:#define __UINTPTR_FMTo__ "lo"
// X86_64-CLOUDABI:#define __UINTPTR_FMTu__ "lu"
// X86_64-CLOUDABI:#define __UINTPTR_FMTx__ "lx"
// X86_64-CLOUDABI:#define __UINTPTR_MAX__ 18446744073709551615UL
// X86_64-CLOUDABI:#define __UINTPTR_TYPE__ long unsigned int
// X86_64-CLOUDABI:#define __UINTPTR_WIDTH__ 64
// X86_64-CLOUDABI:#define __UINT_FAST16_FMTX__ "hX"
// X86_64-CLOUDABI:#define __UINT_FAST16_FMTo__ "ho"
// X86_64-CLOUDABI:#define __UINT_FAST16_FMTu__ "hu"
// X86_64-CLOUDABI:#define __UINT_FAST16_FMTx__ "hx"
// X86_64-CLOUDABI:#define __UINT_FAST16_MAX__ 65535
// X86_64-CLOUDABI:#define __UINT_FAST16_TYPE__ unsigned short
// X86_64-CLOUDABI:#define __UINT_FAST32_FMTX__ "X"
// X86_64-CLOUDABI:#define __UINT_FAST32_FMTo__ "o"
// X86_64-CLOUDABI:#define __UINT_FAST32_FMTu__ "u"
// X86_64-CLOUDABI:#define __UINT_FAST32_FMTx__ "x"
// X86_64-CLOUDABI:#define __UINT_FAST32_MAX__ 4294967295U
// X86_64-CLOUDABI:#define __UINT_FAST32_TYPE__ unsigned int
// X86_64-CLOUDABI:#define __UINT_FAST64_FMTX__ "lX"
// X86_64-CLOUDABI:#define __UINT_FAST64_FMTo__ "lo"
// X86_64-CLOUDABI:#define __UINT_FAST64_FMTu__ "lu"
// X86_64-CLOUDABI:#define __UINT_FAST64_FMTx__ "lx"
// X86_64-CLOUDABI:#define __UINT_FAST64_MAX__ 18446744073709551615UL
// X86_64-CLOUDABI:#define __UINT_FAST64_TYPE__ long unsigned int
// X86_64-CLOUDABI:#define __UINT_FAST8_FMTX__ "hhX"
// X86_64-CLOUDABI:#define __UINT_FAST8_FMTo__ "hho"
// X86_64-CLOUDABI:#define __UINT_FAST8_FMTu__ "hhu"
// X86_64-CLOUDABI:#define __UINT_FAST8_FMTx__ "hhx"
// X86_64-CLOUDABI:#define __UINT_FAST8_MAX__ 255
// X86_64-CLOUDABI:#define __UINT_FAST8_TYPE__ unsigned char
// X86_64-CLOUDABI:#define __UINT_LEAST16_FMTX__ "hX"
// X86_64-CLOUDABI:#define __UINT_LEAST16_FMTo__ "ho"
// X86_64-CLOUDABI:#define __UINT_LEAST16_FMTu__ "hu"
// X86_64-CLOUDABI:#define __UINT_LEAST16_FMTx__ "hx"
// X86_64-CLOUDABI:#define __UINT_LEAST16_MAX__ 65535
// X86_64-CLOUDABI:#define __UINT_LEAST16_TYPE__ unsigned short
// X86_64-CLOUDABI:#define __UINT_LEAST32_FMTX__ "X"
// X86_64-CLOUDABI:#define __UINT_LEAST32_FMTo__ "o"
// X86_64-CLOUDABI:#define __UINT_LEAST32_FMTu__ "u"
// X86_64-CLOUDABI:#define __UINT_LEAST32_FMTx__ "x"
// X86_64-CLOUDABI:#define __UINT_LEAST32_MAX__ 4294967295U
// X86_64-CLOUDABI:#define __UINT_LEAST32_TYPE__ unsigned int
// X86_64-CLOUDABI:#define __UINT_LEAST64_FMTX__ "lX"
// X86_64-CLOUDABI:#define __UINT_LEAST64_FMTo__ "lo"
// X86_64-CLOUDABI:#define __UINT_LEAST64_FMTu__ "lu"
// X86_64-CLOUDABI:#define __UINT_LEAST64_FMTx__ "lx"
// X86_64-CLOUDABI:#define __UINT_LEAST64_MAX__ 18446744073709551615UL
// X86_64-CLOUDABI:#define __UINT_LEAST64_TYPE__ long unsigned int
// X86_64-CLOUDABI:#define __UINT_LEAST8_FMTX__ "hhX"
// X86_64-CLOUDABI:#define __UINT_LEAST8_FMTo__ "hho"
// X86_64-CLOUDABI:#define __UINT_LEAST8_FMTu__ "hhu"
// X86_64-CLOUDABI:#define __UINT_LEAST8_FMTx__ "hhx"
// X86_64-CLOUDABI:#define __UINT_LEAST8_MAX__ 255
// X86_64-CLOUDABI:#define __UINT_LEAST8_TYPE__ unsigned char
// X86_64-CLOUDABI:#define __USER_LABEL_PREFIX__
// X86_64-CLOUDABI:#define __VERSION__ "4.2.1 Compatible{{.*}}
// X86_64-CLOUDABI:#define __WCHAR_MAX__ 2147483647
// X86_64-CLOUDABI:#define __WCHAR_TYPE__ int
// X86_64-CLOUDABI:#define __WCHAR_WIDTH__ 32
// X86_64-CLOUDABI:#define __WINT_TYPE__ int
// X86_64-CLOUDABI:#define __WINT_WIDTH__ 32
// X86_64-CLOUDABI:#define __amd64 1
// X86_64-CLOUDABI:#define __amd64__ 1
// X86_64-CLOUDABI:#define __clang__ 1
// X86_64-CLOUDABI:#define __clang_major__ {{.*}}
// X86_64-CLOUDABI:#define __clang_minor__ {{.*}}
// X86_64-CLOUDABI:#define __clang_patchlevel__ {{.*}}
// X86_64-CLOUDABI:#define __clang_version__ {{.*}}
// X86_64-CLOUDABI:#define __llvm__ 1
// X86_64-CLOUDABI:#define __x86_64 1
// X86_64-CLOUDABI:#define __x86_64__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=x86_64-pc-linux-gnu < /dev/null | FileCheck -match-full-lines -check-prefix X86_64-LINUX %s
//
// X86_64-LINUX:#define _LP64 1
// X86_64-LINUX:#define __BIGGEST_ALIGNMENT__ 16
// X86_64-LINUX:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// X86_64-LINUX:#define __CHAR16_TYPE__ unsigned short
// X86_64-LINUX:#define __CHAR32_TYPE__ unsigned int
// X86_64-LINUX:#define __CHAR_BIT__ 8
// X86_64-LINUX:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// X86_64-LINUX:#define __DBL_DIG__ 15
// X86_64-LINUX:#define __DBL_EPSILON__ 2.2204460492503131e-16
// X86_64-LINUX:#define __DBL_HAS_DENORM__ 1
// X86_64-LINUX:#define __DBL_HAS_INFINITY__ 1
// X86_64-LINUX:#define __DBL_HAS_QUIET_NAN__ 1
// X86_64-LINUX:#define __DBL_MANT_DIG__ 53
// X86_64-LINUX:#define __DBL_MAX_10_EXP__ 308
// X86_64-LINUX:#define __DBL_MAX_EXP__ 1024
// X86_64-LINUX:#define __DBL_MAX__ 1.7976931348623157e+308
// X86_64-LINUX:#define __DBL_MIN_10_EXP__ (-307)
// X86_64-LINUX:#define __DBL_MIN_EXP__ (-1021)
// X86_64-LINUX:#define __DBL_MIN__ 2.2250738585072014e-308
// X86_64-LINUX:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// X86_64-LINUX:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// X86_64-LINUX:#define __FLT_DIG__ 6
// X86_64-LINUX:#define __FLT_EPSILON__ 1.19209290e-7F
// X86_64-LINUX:#define __FLT_EVAL_METHOD__ 0
// X86_64-LINUX:#define __FLT_HAS_DENORM__ 1
// X86_64-LINUX:#define __FLT_HAS_INFINITY__ 1
// X86_64-LINUX:#define __FLT_HAS_QUIET_NAN__ 1
// X86_64-LINUX:#define __FLT_MANT_DIG__ 24
// X86_64-LINUX:#define __FLT_MAX_10_EXP__ 38
// X86_64-LINUX:#define __FLT_MAX_EXP__ 128
// X86_64-LINUX:#define __FLT_MAX__ 3.40282347e+38F
// X86_64-LINUX:#define __FLT_MIN_10_EXP__ (-37)
// X86_64-LINUX:#define __FLT_MIN_EXP__ (-125)
// X86_64-LINUX:#define __FLT_MIN__ 1.17549435e-38F
// X86_64-LINUX:#define __FLT_RADIX__ 2
// X86_64-LINUX:#define __INT16_C_SUFFIX__
// X86_64-LINUX:#define __INT16_FMTd__ "hd"
// X86_64-LINUX:#define __INT16_FMTi__ "hi"
// X86_64-LINUX:#define __INT16_MAX__ 32767
// X86_64-LINUX:#define __INT16_TYPE__ short
// X86_64-LINUX:#define __INT32_C_SUFFIX__
// X86_64-LINUX:#define __INT32_FMTd__ "d"
// X86_64-LINUX:#define __INT32_FMTi__ "i"
// X86_64-LINUX:#define __INT32_MAX__ 2147483647
// X86_64-LINUX:#define __INT32_TYPE__ int
// X86_64-LINUX:#define __INT64_C_SUFFIX__ L
// X86_64-LINUX:#define __INT64_FMTd__ "ld"
// X86_64-LINUX:#define __INT64_FMTi__ "li"
// X86_64-LINUX:#define __INT64_MAX__ 9223372036854775807L
// X86_64-LINUX:#define __INT64_TYPE__ long int
// X86_64-LINUX:#define __INT8_C_SUFFIX__
// X86_64-LINUX:#define __INT8_FMTd__ "hhd"
// X86_64-LINUX:#define __INT8_FMTi__ "hhi"
// X86_64-LINUX:#define __INT8_MAX__ 127
// X86_64-LINUX:#define __INT8_TYPE__ signed char
// X86_64-LINUX:#define __INTMAX_C_SUFFIX__ L
// X86_64-LINUX:#define __INTMAX_FMTd__ "ld"
// X86_64-LINUX:#define __INTMAX_FMTi__ "li"
// X86_64-LINUX:#define __INTMAX_MAX__ 9223372036854775807L
// X86_64-LINUX:#define __INTMAX_TYPE__ long int
// X86_64-LINUX:#define __INTMAX_WIDTH__ 64
// X86_64-LINUX:#define __INTPTR_FMTd__ "ld"
// X86_64-LINUX:#define __INTPTR_FMTi__ "li"
// X86_64-LINUX:#define __INTPTR_MAX__ 9223372036854775807L
// X86_64-LINUX:#define __INTPTR_TYPE__ long int
// X86_64-LINUX:#define __INTPTR_WIDTH__ 64
// X86_64-LINUX:#define __INT_FAST16_FMTd__ "hd"
// X86_64-LINUX:#define __INT_FAST16_FMTi__ "hi"
// X86_64-LINUX:#define __INT_FAST16_MAX__ 32767
// X86_64-LINUX:#define __INT_FAST16_TYPE__ short
// X86_64-LINUX:#define __INT_FAST32_FMTd__ "d"
// X86_64-LINUX:#define __INT_FAST32_FMTi__ "i"
// X86_64-LINUX:#define __INT_FAST32_MAX__ 2147483647
// X86_64-LINUX:#define __INT_FAST32_TYPE__ int
// X86_64-LINUX:#define __INT_FAST64_FMTd__ "ld"
// X86_64-LINUX:#define __INT_FAST64_FMTi__ "li"
// X86_64-LINUX:#define __INT_FAST64_MAX__ 9223372036854775807L
// X86_64-LINUX:#define __INT_FAST64_TYPE__ long int
// X86_64-LINUX:#define __INT_FAST8_FMTd__ "hhd"
// X86_64-LINUX:#define __INT_FAST8_FMTi__ "hhi"
// X86_64-LINUX:#define __INT_FAST8_MAX__ 127
// X86_64-LINUX:#define __INT_FAST8_TYPE__ signed char
// X86_64-LINUX:#define __INT_LEAST16_FMTd__ "hd"
// X86_64-LINUX:#define __INT_LEAST16_FMTi__ "hi"
// X86_64-LINUX:#define __INT_LEAST16_MAX__ 32767
// X86_64-LINUX:#define __INT_LEAST16_TYPE__ short
// X86_64-LINUX:#define __INT_LEAST32_FMTd__ "d"
// X86_64-LINUX:#define __INT_LEAST32_FMTi__ "i"
// X86_64-LINUX:#define __INT_LEAST32_MAX__ 2147483647
// X86_64-LINUX:#define __INT_LEAST32_TYPE__ int
// X86_64-LINUX:#define __INT_LEAST64_FMTd__ "ld"
// X86_64-LINUX:#define __INT_LEAST64_FMTi__ "li"
// X86_64-LINUX:#define __INT_LEAST64_MAX__ 9223372036854775807L
// X86_64-LINUX:#define __INT_LEAST64_TYPE__ long int
// X86_64-LINUX:#define __INT_LEAST8_FMTd__ "hhd"
// X86_64-LINUX:#define __INT_LEAST8_FMTi__ "hhi"
// X86_64-LINUX:#define __INT_LEAST8_MAX__ 127
// X86_64-LINUX:#define __INT_LEAST8_TYPE__ signed char
// X86_64-LINUX:#define __INT_MAX__ 2147483647
// X86_64-LINUX:#define __LDBL_DENORM_MIN__ 3.64519953188247460253e-4951L
// X86_64-LINUX:#define __LDBL_DIG__ 18
// X86_64-LINUX:#define __LDBL_EPSILON__ 1.08420217248550443401e-19L
// X86_64-LINUX:#define __LDBL_HAS_DENORM__ 1
// X86_64-LINUX:#define __LDBL_HAS_INFINITY__ 1
// X86_64-LINUX:#define __LDBL_HAS_QUIET_NAN__ 1
// X86_64-LINUX:#define __LDBL_MANT_DIG__ 64
// X86_64-LINUX:#define __LDBL_MAX_10_EXP__ 4932
// X86_64-LINUX:#define __LDBL_MAX_EXP__ 16384
// X86_64-LINUX:#define __LDBL_MAX__ 1.18973149535723176502e+4932L
// X86_64-LINUX:#define __LDBL_MIN_10_EXP__ (-4931)
// X86_64-LINUX:#define __LDBL_MIN_EXP__ (-16381)
// X86_64-LINUX:#define __LDBL_MIN__ 3.36210314311209350626e-4932L
// X86_64-LINUX:#define __LITTLE_ENDIAN__ 1
// X86_64-LINUX:#define __LONG_LONG_MAX__ 9223372036854775807LL
// X86_64-LINUX:#define __LONG_MAX__ 9223372036854775807L
// X86_64-LINUX:#define __LP64__ 1
// X86_64-LINUX:#define __MMX__ 1
// X86_64-LINUX:#define __NO_MATH_INLINES 1
// X86_64-LINUX:#define __POINTER_WIDTH__ 64
// X86_64-LINUX:#define __PTRDIFF_TYPE__ long int
// X86_64-LINUX:#define __PTRDIFF_WIDTH__ 64
// X86_64-LINUX:#define __REGISTER_PREFIX__
// X86_64-LINUX:#define __SCHAR_MAX__ 127
// X86_64-LINUX:#define __SHRT_MAX__ 32767
// X86_64-LINUX:#define __SIG_ATOMIC_MAX__ 2147483647
// X86_64-LINUX:#define __SIG_ATOMIC_WIDTH__ 32
// X86_64-LINUX:#define __SIZEOF_DOUBLE__ 8
// X86_64-LINUX:#define __SIZEOF_FLOAT__ 4
// X86_64-LINUX:#define __SIZEOF_INT__ 4
// X86_64-LINUX:#define __SIZEOF_LONG_DOUBLE__ 16
// X86_64-LINUX:#define __SIZEOF_LONG_LONG__ 8
// X86_64-LINUX:#define __SIZEOF_LONG__ 8
// X86_64-LINUX:#define __SIZEOF_POINTER__ 8
// X86_64-LINUX:#define __SIZEOF_PTRDIFF_T__ 8
// X86_64-LINUX:#define __SIZEOF_SHORT__ 2
// X86_64-LINUX:#define __SIZEOF_SIZE_T__ 8
// X86_64-LINUX:#define __SIZEOF_WCHAR_T__ 4
// X86_64-LINUX:#define __SIZEOF_WINT_T__ 4
// X86_64-LINUX:#define __SIZE_MAX__ 18446744073709551615UL
// X86_64-LINUX:#define __SIZE_TYPE__ long unsigned int
// X86_64-LINUX:#define __SIZE_WIDTH__ 64
// X86_64-LINUX:#define __SSE2_MATH__ 1
// X86_64-LINUX:#define __SSE2__ 1
// X86_64-LINUX:#define __SSE_MATH__ 1
// X86_64-LINUX:#define __SSE__ 1
// X86_64-LINUX:#define __UINT16_C_SUFFIX__
// X86_64-LINUX:#define __UINT16_MAX__ 65535
// X86_64-LINUX:#define __UINT16_TYPE__ unsigned short
// X86_64-LINUX:#define __UINT32_C_SUFFIX__ U
// X86_64-LINUX:#define __UINT32_MAX__ 4294967295U
// X86_64-LINUX:#define __UINT32_TYPE__ unsigned int
// X86_64-LINUX:#define __UINT64_C_SUFFIX__ UL
// X86_64-LINUX:#define __UINT64_MAX__ 18446744073709551615UL
// X86_64-LINUX:#define __UINT64_TYPE__ long unsigned int
// X86_64-LINUX:#define __UINT8_C_SUFFIX__
// X86_64-LINUX:#define __UINT8_MAX__ 255
// X86_64-LINUX:#define __UINT8_TYPE__ unsigned char
// X86_64-LINUX:#define __UINTMAX_C_SUFFIX__ UL
// X86_64-LINUX:#define __UINTMAX_MAX__ 18446744073709551615UL
// X86_64-LINUX:#define __UINTMAX_TYPE__ long unsigned int
// X86_64-LINUX:#define __UINTMAX_WIDTH__ 64
// X86_64-LINUX:#define __UINTPTR_MAX__ 18446744073709551615UL
// X86_64-LINUX:#define __UINTPTR_TYPE__ long unsigned int
// X86_64-LINUX:#define __UINTPTR_WIDTH__ 64
// X86_64-LINUX:#define __UINT_FAST16_MAX__ 65535
// X86_64-LINUX:#define __UINT_FAST16_TYPE__ unsigned short
// X86_64-LINUX:#define __UINT_FAST32_MAX__ 4294967295U
// X86_64-LINUX:#define __UINT_FAST32_TYPE__ unsigned int
// X86_64-LINUX:#define __UINT_FAST64_MAX__ 18446744073709551615UL
// X86_64-LINUX:#define __UINT_FAST64_TYPE__ long unsigned int
// X86_64-LINUX:#define __UINT_FAST8_MAX__ 255
// X86_64-LINUX:#define __UINT_FAST8_TYPE__ unsigned char
// X86_64-LINUX:#define __UINT_LEAST16_MAX__ 65535
// X86_64-LINUX:#define __UINT_LEAST16_TYPE__ unsigned short
// X86_64-LINUX:#define __UINT_LEAST32_MAX__ 4294967295U
// X86_64-LINUX:#define __UINT_LEAST32_TYPE__ unsigned int
// X86_64-LINUX:#define __UINT_LEAST64_MAX__ 18446744073709551615UL
// X86_64-LINUX:#define __UINT_LEAST64_TYPE__ long unsigned int
// X86_64-LINUX:#define __UINT_LEAST8_MAX__ 255
// X86_64-LINUX:#define __UINT_LEAST8_TYPE__ unsigned char
// X86_64-LINUX:#define __USER_LABEL_PREFIX__
// X86_64-LINUX:#define __WCHAR_MAX__ 2147483647
// X86_64-LINUX:#define __WCHAR_TYPE__ int
// X86_64-LINUX:#define __WCHAR_WIDTH__ 32
// X86_64-LINUX:#define __WINT_TYPE__ unsigned int
// X86_64-LINUX:#define __WINT_WIDTH__ 32
// X86_64-LINUX:#define __amd64 1
// X86_64-LINUX:#define __amd64__ 1
// X86_64-LINUX:#define __x86_64 1
// X86_64-LINUX:#define __x86_64__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=x86_64-unknown-freebsd9.1 < /dev/null | FileCheck -match-full-lines -check-prefix X86_64-FREEBSD %s
//
// X86_64-FREEBSD:#define __DBL_DECIMAL_DIG__ 17
// X86_64-FREEBSD:#define __FLT_DECIMAL_DIG__ 9
// X86_64-FREEBSD:#define __FreeBSD__ 9
// X86_64-FREEBSD:#define __FreeBSD_cc_version 900001
// X86_64-FREEBSD:#define __LDBL_DECIMAL_DIG__ 21
// X86_64-FREEBSD:#define __STDC_MB_MIGHT_NEQ_WC__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=x86_64-netbsd < /dev/null | FileCheck -match-full-lines -check-prefix X86_64-NETBSD %s
//
// X86_64-NETBSD:#define _LP64 1
// X86_64-NETBSD:#define __BIGGEST_ALIGNMENT__ 16
// X86_64-NETBSD:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// X86_64-NETBSD:#define __CHAR16_TYPE__ unsigned short
// X86_64-NETBSD:#define __CHAR32_TYPE__ unsigned int
// X86_64-NETBSD:#define __CHAR_BIT__ 8
// X86_64-NETBSD:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// X86_64-NETBSD:#define __DBL_DIG__ 15
// X86_64-NETBSD:#define __DBL_EPSILON__ 2.2204460492503131e-16
// X86_64-NETBSD:#define __DBL_HAS_DENORM__ 1
// X86_64-NETBSD:#define __DBL_HAS_INFINITY__ 1
// X86_64-NETBSD:#define __DBL_HAS_QUIET_NAN__ 1
// X86_64-NETBSD:#define __DBL_MANT_DIG__ 53
// X86_64-NETBSD:#define __DBL_MAX_10_EXP__ 308
// X86_64-NETBSD:#define __DBL_MAX_EXP__ 1024
// X86_64-NETBSD:#define __DBL_MAX__ 1.7976931348623157e+308
// X86_64-NETBSD:#define __DBL_MIN_10_EXP__ (-307)
// X86_64-NETBSD:#define __DBL_MIN_EXP__ (-1021)
// X86_64-NETBSD:#define __DBL_MIN__ 2.2250738585072014e-308
// X86_64-NETBSD:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// X86_64-NETBSD:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// X86_64-NETBSD:#define __FLT_DIG__ 6
// X86_64-NETBSD:#define __FLT_EPSILON__ 1.19209290e-7F
// X86_64-NETBSD:#define __FLT_EVAL_METHOD__ 0
// X86_64-NETBSD:#define __FLT_HAS_DENORM__ 1
// X86_64-NETBSD:#define __FLT_HAS_INFINITY__ 1
// X86_64-NETBSD:#define __FLT_HAS_QUIET_NAN__ 1
// X86_64-NETBSD:#define __FLT_MANT_DIG__ 24
// X86_64-NETBSD:#define __FLT_MAX_10_EXP__ 38
// X86_64-NETBSD:#define __FLT_MAX_EXP__ 128
// X86_64-NETBSD:#define __FLT_MAX__ 3.40282347e+38F
// X86_64-NETBSD:#define __FLT_MIN_10_EXP__ (-37)
// X86_64-NETBSD:#define __FLT_MIN_EXP__ (-125)
// X86_64-NETBSD:#define __FLT_MIN__ 1.17549435e-38F
// X86_64-NETBSD:#define __FLT_RADIX__ 2
// X86_64-NETBSD:#define __INT16_C_SUFFIX__
// X86_64-NETBSD:#define __INT16_FMTd__ "hd"
// X86_64-NETBSD:#define __INT16_FMTi__ "hi"
// X86_64-NETBSD:#define __INT16_MAX__ 32767
// X86_64-NETBSD:#define __INT16_TYPE__ short
// X86_64-NETBSD:#define __INT32_C_SUFFIX__
// X86_64-NETBSD:#define __INT32_FMTd__ "d"
// X86_64-NETBSD:#define __INT32_FMTi__ "i"
// X86_64-NETBSD:#define __INT32_MAX__ 2147483647
// X86_64-NETBSD:#define __INT32_TYPE__ int
// X86_64-NETBSD:#define __INT64_C_SUFFIX__ L
// X86_64-NETBSD:#define __INT64_FMTd__ "ld"
// X86_64-NETBSD:#define __INT64_FMTi__ "li"
// X86_64-NETBSD:#define __INT64_MAX__ 9223372036854775807L
// X86_64-NETBSD:#define __INT64_TYPE__ long int
// X86_64-NETBSD:#define __INT8_C_SUFFIX__
// X86_64-NETBSD:#define __INT8_FMTd__ "hhd"
// X86_64-NETBSD:#define __INT8_FMTi__ "hhi"
// X86_64-NETBSD:#define __INT8_MAX__ 127
// X86_64-NETBSD:#define __INT8_TYPE__ signed char
// X86_64-NETBSD:#define __INTMAX_C_SUFFIX__ L
// X86_64-NETBSD:#define __INTMAX_FMTd__ "ld"
// X86_64-NETBSD:#define __INTMAX_FMTi__ "li"
// X86_64-NETBSD:#define __INTMAX_MAX__ 9223372036854775807L
// X86_64-NETBSD:#define __INTMAX_TYPE__ long int
// X86_64-NETBSD:#define __INTMAX_WIDTH__ 64
// X86_64-NETBSD:#define __INTPTR_FMTd__ "ld"
// X86_64-NETBSD:#define __INTPTR_FMTi__ "li"
// X86_64-NETBSD:#define __INTPTR_MAX__ 9223372036854775807L
// X86_64-NETBSD:#define __INTPTR_TYPE__ long int
// X86_64-NETBSD:#define __INTPTR_WIDTH__ 64
// X86_64-NETBSD:#define __INT_FAST16_FMTd__ "hd"
// X86_64-NETBSD:#define __INT_FAST16_FMTi__ "hi"
// X86_64-NETBSD:#define __INT_FAST16_MAX__ 32767
// X86_64-NETBSD:#define __INT_FAST16_TYPE__ short
// X86_64-NETBSD:#define __INT_FAST32_FMTd__ "d"
// X86_64-NETBSD:#define __INT_FAST32_FMTi__ "i"
// X86_64-NETBSD:#define __INT_FAST32_MAX__ 2147483647
// X86_64-NETBSD:#define __INT_FAST32_TYPE__ int
// X86_64-NETBSD:#define __INT_FAST64_FMTd__ "ld"
// X86_64-NETBSD:#define __INT_FAST64_FMTi__ "li"
// X86_64-NETBSD:#define __INT_FAST64_MAX__ 9223372036854775807L
// X86_64-NETBSD:#define __INT_FAST64_TYPE__ long int
// X86_64-NETBSD:#define __INT_FAST8_FMTd__ "hhd"
// X86_64-NETBSD:#define __INT_FAST8_FMTi__ "hhi"
// X86_64-NETBSD:#define __INT_FAST8_MAX__ 127
// X86_64-NETBSD:#define __INT_FAST8_TYPE__ signed char
// X86_64-NETBSD:#define __INT_LEAST16_FMTd__ "hd"
// X86_64-NETBSD:#define __INT_LEAST16_FMTi__ "hi"
// X86_64-NETBSD:#define __INT_LEAST16_MAX__ 32767
// X86_64-NETBSD:#define __INT_LEAST16_TYPE__ short
// X86_64-NETBSD:#define __INT_LEAST32_FMTd__ "d"
// X86_64-NETBSD:#define __INT_LEAST32_FMTi__ "i"
// X86_64-NETBSD:#define __INT_LEAST32_MAX__ 2147483647
// X86_64-NETBSD:#define __INT_LEAST32_TYPE__ int
// X86_64-NETBSD:#define __INT_LEAST64_FMTd__ "ld"
// X86_64-NETBSD:#define __INT_LEAST64_FMTi__ "li"
// X86_64-NETBSD:#define __INT_LEAST64_MAX__ 9223372036854775807L
// X86_64-NETBSD:#define __INT_LEAST64_TYPE__ long int
// X86_64-NETBSD:#define __INT_LEAST8_FMTd__ "hhd"
// X86_64-NETBSD:#define __INT_LEAST8_FMTi__ "hhi"
// X86_64-NETBSD:#define __INT_LEAST8_MAX__ 127
// X86_64-NETBSD:#define __INT_LEAST8_TYPE__ signed char
// X86_64-NETBSD:#define __INT_MAX__ 2147483647
// X86_64-NETBSD:#define __LDBL_DENORM_MIN__ 3.64519953188247460253e-4951L
// X86_64-NETBSD:#define __LDBL_DIG__ 18
// X86_64-NETBSD:#define __LDBL_EPSILON__ 1.08420217248550443401e-19L
// X86_64-NETBSD:#define __LDBL_HAS_DENORM__ 1
// X86_64-NETBSD:#define __LDBL_HAS_INFINITY__ 1
// X86_64-NETBSD:#define __LDBL_HAS_QUIET_NAN__ 1
// X86_64-NETBSD:#define __LDBL_MANT_DIG__ 64
// X86_64-NETBSD:#define __LDBL_MAX_10_EXP__ 4932
// X86_64-NETBSD:#define __LDBL_MAX_EXP__ 16384
// X86_64-NETBSD:#define __LDBL_MAX__ 1.18973149535723176502e+4932L
// X86_64-NETBSD:#define __LDBL_MIN_10_EXP__ (-4931)
// X86_64-NETBSD:#define __LDBL_MIN_EXP__ (-16381)
// X86_64-NETBSD:#define __LDBL_MIN__ 3.36210314311209350626e-4932L
// X86_64-NETBSD:#define __LITTLE_ENDIAN__ 1
// X86_64-NETBSD:#define __LONG_LONG_MAX__ 9223372036854775807LL
// X86_64-NETBSD:#define __LONG_MAX__ 9223372036854775807L
// X86_64-NETBSD:#define __LP64__ 1
// X86_64-NETBSD:#define __MMX__ 1
// X86_64-NETBSD:#define __NO_MATH_INLINES 1
// X86_64-NETBSD:#define __POINTER_WIDTH__ 64
// X86_64-NETBSD:#define __PTRDIFF_TYPE__ long int
// X86_64-NETBSD:#define __PTRDIFF_WIDTH__ 64
// X86_64-NETBSD:#define __REGISTER_PREFIX__
// X86_64-NETBSD:#define __SCHAR_MAX__ 127
// X86_64-NETBSD:#define __SHRT_MAX__ 32767
// X86_64-NETBSD:#define __SIG_ATOMIC_MAX__ 2147483647
// X86_64-NETBSD:#define __SIG_ATOMIC_WIDTH__ 32
// X86_64-NETBSD:#define __SIZEOF_DOUBLE__ 8
// X86_64-NETBSD:#define __SIZEOF_FLOAT__ 4
// X86_64-NETBSD:#define __SIZEOF_INT__ 4
// X86_64-NETBSD:#define __SIZEOF_LONG_DOUBLE__ 16
// X86_64-NETBSD:#define __SIZEOF_LONG_LONG__ 8
// X86_64-NETBSD:#define __SIZEOF_LONG__ 8
// X86_64-NETBSD:#define __SIZEOF_POINTER__ 8
// X86_64-NETBSD:#define __SIZEOF_PTRDIFF_T__ 8
// X86_64-NETBSD:#define __SIZEOF_SHORT__ 2
// X86_64-NETBSD:#define __SIZEOF_SIZE_T__ 8
// X86_64-NETBSD:#define __SIZEOF_WCHAR_T__ 4
// X86_64-NETBSD:#define __SIZEOF_WINT_T__ 4
// X86_64-NETBSD:#define __SIZE_MAX__ 18446744073709551615UL
// X86_64-NETBSD:#define __SIZE_TYPE__ long unsigned int
// X86_64-NETBSD:#define __SIZE_WIDTH__ 64
// X86_64-NETBSD:#define __SSE2_MATH__ 1
// X86_64-NETBSD:#define __SSE2__ 1
// X86_64-NETBSD:#define __SSE_MATH__ 1
// X86_64-NETBSD:#define __SSE__ 1
// X86_64-NETBSD:#define __UINT16_C_SUFFIX__
// X86_64-NETBSD:#define __UINT16_MAX__ 65535
// X86_64-NETBSD:#define __UINT16_TYPE__ unsigned short
// X86_64-NETBSD:#define __UINT32_C_SUFFIX__ U
// X86_64-NETBSD:#define __UINT32_MAX__ 4294967295U
// X86_64-NETBSD:#define __UINT32_TYPE__ unsigned int
// X86_64-NETBSD:#define __UINT64_C_SUFFIX__ UL
// X86_64-NETBSD:#define __UINT64_MAX__ 18446744073709551615UL
// X86_64-NETBSD:#define __UINT64_TYPE__ long unsigned int
// X86_64-NETBSD:#define __UINT8_C_SUFFIX__
// X86_64-NETBSD:#define __UINT8_MAX__ 255
// X86_64-NETBSD:#define __UINT8_TYPE__ unsigned char
// X86_64-NETBSD:#define __UINTMAX_C_SUFFIX__ UL
// X86_64-NETBSD:#define __UINTMAX_MAX__ 18446744073709551615UL
// X86_64-NETBSD:#define __UINTMAX_TYPE__ long unsigned int
// X86_64-NETBSD:#define __UINTMAX_WIDTH__ 64
// X86_64-NETBSD:#define __UINTPTR_MAX__ 18446744073709551615UL
// X86_64-NETBSD:#define __UINTPTR_TYPE__ long unsigned int
// X86_64-NETBSD:#define __UINTPTR_WIDTH__ 64
// X86_64-NETBSD:#define __UINT_FAST16_MAX__ 65535
// X86_64-NETBSD:#define __UINT_FAST16_TYPE__ unsigned short
// X86_64-NETBSD:#define __UINT_FAST32_MAX__ 4294967295U
// X86_64-NETBSD:#define __UINT_FAST32_TYPE__ unsigned int
// X86_64-NETBSD:#define __UINT_FAST64_MAX__ 18446744073709551615UL
// X86_64-NETBSD:#define __UINT_FAST64_TYPE__ long unsigned int
// X86_64-NETBSD:#define __UINT_FAST8_MAX__ 255
// X86_64-NETBSD:#define __UINT_FAST8_TYPE__ unsigned char
// X86_64-NETBSD:#define __UINT_LEAST16_MAX__ 65535
// X86_64-NETBSD:#define __UINT_LEAST16_TYPE__ unsigned short
// X86_64-NETBSD:#define __UINT_LEAST32_MAX__ 4294967295U
// X86_64-NETBSD:#define __UINT_LEAST32_TYPE__ unsigned int
// X86_64-NETBSD:#define __UINT_LEAST64_MAX__ 18446744073709551615UL
// X86_64-NETBSD:#define __UINT_LEAST64_TYPE__ long unsigned int
// X86_64-NETBSD:#define __UINT_LEAST8_MAX__ 255
// X86_64-NETBSD:#define __UINT_LEAST8_TYPE__ unsigned char
// X86_64-NETBSD:#define __USER_LABEL_PREFIX__
// X86_64-NETBSD:#define __WCHAR_MAX__ 2147483647
// X86_64-NETBSD:#define __WCHAR_TYPE__ int
// X86_64-NETBSD:#define __WCHAR_WIDTH__ 32
// X86_64-NETBSD:#define __WINT_TYPE__ int
// X86_64-NETBSD:#define __WINT_WIDTH__ 32
// X86_64-NETBSD:#define __amd64 1
// X86_64-NETBSD:#define __amd64__ 1
// X86_64-NETBSD:#define __x86_64 1
// X86_64-NETBSD:#define __x86_64__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=x86_64-scei-ps4 < /dev/null | FileCheck -match-full-lines -check-prefix PS4 %s
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
// RUN: %clang_cc1 -E -dM -triple=x86_64-pc-mingw32 < /dev/null | FileCheck -match-full-lines -check-prefix X86-64-DECLSPEC %s
// RUN: %clang_cc1 -E -dM -fms-extensions -triple=x86_64-unknown-mingw32 < /dev/null | FileCheck -match-full-lines -check-prefix X86-64-DECLSPEC %s
// X86-64-DECLSPEC: #define __declspec{{.*}}
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=sparc64-none-none < /dev/null | FileCheck -match-full-lines -check-prefix SPARCV9 %s
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
// GNUSOURCE:#define _GNU_SOURCE 1
//
// RUN: %clang_cc1 -x c++ -std=c++98 -fno-rtti -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix NORTTI %s
// NORTTI: #define __GXX_ABI_VERSION {{.*}}
// NORTTI-NOT:#define __GXX_RTTI
// NORTTI:#define __STDC__ 1
//
// RUN: %clang_cc1 -triple arm-linux-androideabi -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix ANDROID %s
// ANDROID-NOT:#define __ANDROID_API__
// ANDROID:#define __ANDROID__ 1
//
// RUN: %clang_cc1 -x c++ -triple i686-linux-android -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix I386-ANDROID-CXX %s
// I386-ANDROID-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 4U
//
// RUN: %clang_cc1 -x c++ -triple x86_64-linux-android -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix X86_64-ANDROID-CXX %s
// X86_64-ANDROID-CXX:#define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 16UL
//
// RUN: %clang_cc1 -triple arm-linux-androideabi20 -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix ANDROID20 %s
// ANDROID20:#define __ANDROID_API__ 20
// ANDROID20:#define __ANDROID__ 1
//
// RUN: %clang_cc1 -triple lanai-unknown-unknown -E -dM < /dev/null | FileCheck -match-full-lines -check-prefix LANAI %s
// LANAI: #define __lanai__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=amd64-unknown-openbsd6.1 < /dev/null | FileCheck -match-full-lines -check-prefix OPENBSD %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=aarch64-unknown-openbsd6.1 < /dev/null | FileCheck -match-full-lines -check-prefix OPENBSD %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=arm-unknown-openbsd6.1-gnueabi < /dev/null | FileCheck -match-full-lines -check-prefix OPENBSD %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=i386-unknown-openbsd6.1 < /dev/null | FileCheck -match-full-lines -check-prefix OPENBSD %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc-unknown-openbsd6.1 < /dev/null | FileCheck -match-full-lines -check-prefix OPENBSD %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips64-unknown-openbsd6.1 < /dev/null | FileCheck -match-full-lines -check-prefix OPENBSD %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=mips64el-unknown-openbsd6.1 < /dev/null | FileCheck -match-full-lines -check-prefix OPENBSD %s
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=sparc64-unknown-openbsd6.1 < /dev/null | FileCheck -match-full-lines -check-prefix OPENBSD %s
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
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-unknown-freebsd < /dev/null | FileCheck -match-full-lines -check-prefix PPC64-FREEBSD %s
// PPC64-FREEBSD-NOT: #define __LONG_DOUBLE_128__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=xcore-none-none < /dev/null | FileCheck -match-full-lines -check-prefix XCORE %s
// XCORE:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// XCORE:#define __LITTLE_ENDIAN__ 1
// XCORE:#define __XS1B__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=wasm32-unknown-unknown \
// RUN:   < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=WEBASSEMBLY32 %s
//
// WEBASSEMBLY32:#define _ILP32 1
// WEBASSEMBLY32-NOT:#define _LP64
// WEBASSEMBLY32-NEXT:#define __ATOMIC_ACQUIRE 2
// WEBASSEMBLY32-NEXT:#define __ATOMIC_ACQ_REL 4
// WEBASSEMBLY32-NEXT:#define __ATOMIC_CONSUME 1
// WEBASSEMBLY32-NEXT:#define __ATOMIC_RELAXED 0
// WEBASSEMBLY32-NEXT:#define __ATOMIC_RELEASE 3
// WEBASSEMBLY32-NEXT:#define __ATOMIC_SEQ_CST 5
// WEBASSEMBLY32-NEXT:#define __BIGGEST_ALIGNMENT__ 16
// WEBASSEMBLY32-NEXT:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// WEBASSEMBLY32-NEXT:#define __CHAR16_TYPE__ unsigned short
// WEBASSEMBLY32-NEXT:#define __CHAR32_TYPE__ unsigned int
// WEBASSEMBLY32-NEXT:#define __CHAR_BIT__ 8
// WEBASSEMBLY32-NOT:#define __CHAR_UNSIGNED__
// WEBASSEMBLY32-NEXT:#define __CONSTANT_CFSTRINGS__ 1
// WEBASSEMBLY32-NEXT:#define __DBL_DECIMAL_DIG__ 17
// WEBASSEMBLY32-NEXT:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// WEBASSEMBLY32-NEXT:#define __DBL_DIG__ 15
// WEBASSEMBLY32-NEXT:#define __DBL_EPSILON__ 2.2204460492503131e-16
// WEBASSEMBLY32-NEXT:#define __DBL_HAS_DENORM__ 1
// WEBASSEMBLY32-NEXT:#define __DBL_HAS_INFINITY__ 1
// WEBASSEMBLY32-NEXT:#define __DBL_HAS_QUIET_NAN__ 1
// WEBASSEMBLY32-NEXT:#define __DBL_MANT_DIG__ 53
// WEBASSEMBLY32-NEXT:#define __DBL_MAX_10_EXP__ 308
// WEBASSEMBLY32-NEXT:#define __DBL_MAX_EXP__ 1024
// WEBASSEMBLY32-NEXT:#define __DBL_MAX__ 1.7976931348623157e+308
// WEBASSEMBLY32-NEXT:#define __DBL_MIN_10_EXP__ (-307)
// WEBASSEMBLY32-NEXT:#define __DBL_MIN_EXP__ (-1021)
// WEBASSEMBLY32-NEXT:#define __DBL_MIN__ 2.2250738585072014e-308
// WEBASSEMBLY32-NEXT:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// WEBASSEMBLY32-NOT:#define __ELF__
// WEBASSEMBLY32-NEXT:#define __FINITE_MATH_ONLY__ 0
// WEBASSEMBLY32-NEXT:#define __FLT_DECIMAL_DIG__ 9
// WEBASSEMBLY32-NEXT:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// WEBASSEMBLY32-NEXT:#define __FLT_DIG__ 6
// WEBASSEMBLY32-NEXT:#define __FLT_EPSILON__ 1.19209290e-7F
// WEBASSEMBLY32-NEXT:#define __FLT_EVAL_METHOD__ 0
// WEBASSEMBLY32-NEXT:#define __FLT_HAS_DENORM__ 1
// WEBASSEMBLY32-NEXT:#define __FLT_HAS_INFINITY__ 1
// WEBASSEMBLY32-NEXT:#define __FLT_HAS_QUIET_NAN__ 1
// WEBASSEMBLY32-NEXT:#define __FLT_MANT_DIG__ 24
// WEBASSEMBLY32-NEXT:#define __FLT_MAX_10_EXP__ 38
// WEBASSEMBLY32-NEXT:#define __FLT_MAX_EXP__ 128
// WEBASSEMBLY32-NEXT:#define __FLT_MAX__ 3.40282347e+38F
// WEBASSEMBLY32-NEXT:#define __FLT_MIN_10_EXP__ (-37)
// WEBASSEMBLY32-NEXT:#define __FLT_MIN_EXP__ (-125)
// WEBASSEMBLY32-NEXT:#define __FLT_MIN__ 1.17549435e-38F
// WEBASSEMBLY32-NEXT:#define __FLT_RADIX__ 2
// WEBASSEMBLY32-NEXT:#define __GCC_ATOMIC_BOOL_LOCK_FREE 2
// WEBASSEMBLY32-NEXT:#define __GCC_ATOMIC_CHAR16_T_LOCK_FREE 2
// WEBASSEMBLY32-NEXT:#define __GCC_ATOMIC_CHAR32_T_LOCK_FREE 2
// WEBASSEMBLY32-NEXT:#define __GCC_ATOMIC_CHAR_LOCK_FREE 2
// WEBASSEMBLY32-NEXT:#define __GCC_ATOMIC_INT_LOCK_FREE 2
// WEBASSEMBLY32-NEXT:#define __GCC_ATOMIC_LLONG_LOCK_FREE 1
// WEBASSEMBLY32-NEXT:#define __GCC_ATOMIC_LONG_LOCK_FREE 2
// WEBASSEMBLY32-NEXT:#define __GCC_ATOMIC_POINTER_LOCK_FREE 2
// WEBASSEMBLY32-NEXT:#define __GCC_ATOMIC_SHORT_LOCK_FREE 2
// WEBASSEMBLY32-NEXT:#define __GCC_ATOMIC_TEST_AND_SET_TRUEVAL 1
// WEBASSEMBLY32-NEXT:#define __GCC_ATOMIC_WCHAR_T_LOCK_FREE 2
// WEBASSEMBLY32-NEXT:#define __GNUC_MINOR__ {{.*}}
// WEBASSEMBLY32-NEXT:#define __GNUC_PATCHLEVEL__ {{.*}}
// WEBASSEMBLY32-NEXT:#define __GNUC_STDC_INLINE__ 1
// WEBASSEMBLY32-NEXT:#define __GNUC__ {{.*}}
// WEBASSEMBLY32-NEXT:#define __GXX_ABI_VERSION 1002
// WEBASSEMBLY32-NEXT:#define __ILP32__ 1
// WEBASSEMBLY32-NEXT:#define __INT16_C_SUFFIX__
// WEBASSEMBLY32-NEXT:#define __INT16_FMTd__ "hd"
// WEBASSEMBLY32-NEXT:#define __INT16_FMTi__ "hi"
// WEBASSEMBLY32-NEXT:#define __INT16_MAX__ 32767
// WEBASSEMBLY32-NEXT:#define __INT16_TYPE__ short
// WEBASSEMBLY32-NEXT:#define __INT32_C_SUFFIX__
// WEBASSEMBLY32-NEXT:#define __INT32_FMTd__ "d"
// WEBASSEMBLY32-NEXT:#define __INT32_FMTi__ "i"
// WEBASSEMBLY32-NEXT:#define __INT32_MAX__ 2147483647
// WEBASSEMBLY32-NEXT:#define __INT32_TYPE__ int
// WEBASSEMBLY32-NEXT:#define __INT64_C_SUFFIX__ LL
// WEBASSEMBLY32-NEXT:#define __INT64_FMTd__ "lld"
// WEBASSEMBLY32-NEXT:#define __INT64_FMTi__ "lli"
// WEBASSEMBLY32-NEXT:#define __INT64_MAX__ 9223372036854775807LL
// WEBASSEMBLY32-NEXT:#define __INT64_TYPE__ long long int
// WEBASSEMBLY32-NEXT:#define __INT8_C_SUFFIX__
// WEBASSEMBLY32-NEXT:#define __INT8_FMTd__ "hhd"
// WEBASSEMBLY32-NEXT:#define __INT8_FMTi__ "hhi"
// WEBASSEMBLY32-NEXT:#define __INT8_MAX__ 127
// WEBASSEMBLY32-NEXT:#define __INT8_TYPE__ signed char
// WEBASSEMBLY32-NEXT:#define __INTMAX_C_SUFFIX__ LL
// WEBASSEMBLY32-NEXT:#define __INTMAX_FMTd__ "lld"
// WEBASSEMBLY32-NEXT:#define __INTMAX_FMTi__ "lli"
// WEBASSEMBLY32-NEXT:#define __INTMAX_MAX__ 9223372036854775807LL
// WEBASSEMBLY32-NEXT:#define __INTMAX_TYPE__ long long int
// WEBASSEMBLY32-NEXT:#define __INTMAX_WIDTH__ 64
// WEBASSEMBLY32-NEXT:#define __INTPTR_FMTd__ "d"
// WEBASSEMBLY32-NEXT:#define __INTPTR_FMTi__ "i"
// WEBASSEMBLY32-NEXT:#define __INTPTR_MAX__ 2147483647
// WEBASSEMBLY32-NEXT:#define __INTPTR_TYPE__ int
// WEBASSEMBLY32-NEXT:#define __INTPTR_WIDTH__ 32
// WEBASSEMBLY32-NEXT:#define __INT_FAST16_FMTd__ "hd"
// WEBASSEMBLY32-NEXT:#define __INT_FAST16_FMTi__ "hi"
// WEBASSEMBLY32-NEXT:#define __INT_FAST16_MAX__ 32767
// WEBASSEMBLY32-NEXT:#define __INT_FAST16_TYPE__ short
// WEBASSEMBLY32-NEXT:#define __INT_FAST32_FMTd__ "d"
// WEBASSEMBLY32-NEXT:#define __INT_FAST32_FMTi__ "i"
// WEBASSEMBLY32-NEXT:#define __INT_FAST32_MAX__ 2147483647
// WEBASSEMBLY32-NEXT:#define __INT_FAST32_TYPE__ int
// WEBASSEMBLY32-NEXT:#define __INT_FAST64_FMTd__ "lld"
// WEBASSEMBLY32-NEXT:#define __INT_FAST64_FMTi__ "lli"
// WEBASSEMBLY32-NEXT:#define __INT_FAST64_MAX__ 9223372036854775807LL
// WEBASSEMBLY32-NEXT:#define __INT_FAST64_TYPE__ long long int
// WEBASSEMBLY32-NEXT:#define __INT_FAST8_FMTd__ "hhd"
// WEBASSEMBLY32-NEXT:#define __INT_FAST8_FMTi__ "hhi"
// WEBASSEMBLY32-NEXT:#define __INT_FAST8_MAX__ 127
// WEBASSEMBLY32-NEXT:#define __INT_FAST8_TYPE__ signed char
// WEBASSEMBLY32-NEXT:#define __INT_LEAST16_FMTd__ "hd"
// WEBASSEMBLY32-NEXT:#define __INT_LEAST16_FMTi__ "hi"
// WEBASSEMBLY32-NEXT:#define __INT_LEAST16_MAX__ 32767
// WEBASSEMBLY32-NEXT:#define __INT_LEAST16_TYPE__ short
// WEBASSEMBLY32-NEXT:#define __INT_LEAST32_FMTd__ "d"
// WEBASSEMBLY32-NEXT:#define __INT_LEAST32_FMTi__ "i"
// WEBASSEMBLY32-NEXT:#define __INT_LEAST32_MAX__ 2147483647
// WEBASSEMBLY32-NEXT:#define __INT_LEAST32_TYPE__ int
// WEBASSEMBLY32-NEXT:#define __INT_LEAST64_FMTd__ "lld"
// WEBASSEMBLY32-NEXT:#define __INT_LEAST64_FMTi__ "lli"
// WEBASSEMBLY32-NEXT:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// WEBASSEMBLY32-NEXT:#define __INT_LEAST64_TYPE__ long long int
// WEBASSEMBLY32-NEXT:#define __INT_LEAST8_FMTd__ "hhd"
// WEBASSEMBLY32-NEXT:#define __INT_LEAST8_FMTi__ "hhi"
// WEBASSEMBLY32-NEXT:#define __INT_LEAST8_MAX__ 127
// WEBASSEMBLY32-NEXT:#define __INT_LEAST8_TYPE__ signed char
// WEBASSEMBLY32-NEXT:#define __INT_MAX__ 2147483647
// WEBASSEMBLY32-NEXT:#define __LDBL_DECIMAL_DIG__ 36
// WEBASSEMBLY32-NEXT:#define __LDBL_DENORM_MIN__ 6.47517511943802511092443895822764655e-4966L
// WEBASSEMBLY32-NEXT:#define __LDBL_DIG__ 33
// WEBASSEMBLY32-NEXT:#define __LDBL_EPSILON__ 1.92592994438723585305597794258492732e-34L
// WEBASSEMBLY32-NEXT:#define __LDBL_HAS_DENORM__ 1
// WEBASSEMBLY32-NEXT:#define __LDBL_HAS_INFINITY__ 1
// WEBASSEMBLY32-NEXT:#define __LDBL_HAS_QUIET_NAN__ 1
// WEBASSEMBLY32-NEXT:#define __LDBL_MANT_DIG__ 113
// WEBASSEMBLY32-NEXT:#define __LDBL_MAX_10_EXP__ 4932
// WEBASSEMBLY32-NEXT:#define __LDBL_MAX_EXP__ 16384
// WEBASSEMBLY32-NEXT:#define __LDBL_MAX__ 1.18973149535723176508575932662800702e+4932L
// WEBASSEMBLY32-NEXT:#define __LDBL_MIN_10_EXP__ (-4931)
// WEBASSEMBLY32-NEXT:#define __LDBL_MIN_EXP__ (-16381)
// WEBASSEMBLY32-NEXT:#define __LDBL_MIN__ 3.36210314311209350626267781732175260e-4932L
// WEBASSEMBLY32-NEXT:#define __LITTLE_ENDIAN__ 1
// WEBASSEMBLY32-NEXT:#define __LONG_LONG_MAX__ 9223372036854775807LL
// WEBASSEMBLY32-NEXT:#define __LONG_MAX__ 2147483647L
// WEBASSEMBLY32-NOT:#define __LP64__
// WEBASSEMBLY32-NEXT:#define __NO_INLINE__ 1
// WEBASSEMBLY32-NEXT:#define __OBJC_BOOL_IS_BOOL 0
// WEBASSEMBLY32-NEXT:#define __ORDER_BIG_ENDIAN__ 4321
// WEBASSEMBLY32-NEXT:#define __ORDER_LITTLE_ENDIAN__ 1234
// WEBASSEMBLY32-NEXT:#define __ORDER_PDP_ENDIAN__ 3412
// WEBASSEMBLY32-NEXT:#define __POINTER_WIDTH__ 32
// WEBASSEMBLY32-NEXT:#define __PRAGMA_REDEFINE_EXTNAME 1
// WEBASSEMBLY32-NEXT:#define __PTRDIFF_FMTd__ "d"
// WEBASSEMBLY32-NEXT:#define __PTRDIFF_FMTi__ "i"
// WEBASSEMBLY32-NEXT:#define __PTRDIFF_MAX__ 2147483647
// WEBASSEMBLY32-NEXT:#define __PTRDIFF_TYPE__ int
// WEBASSEMBLY32-NEXT:#define __PTRDIFF_WIDTH__ 32
// WEBASSEMBLY32-NOT:#define __REGISTER_PREFIX__
// WEBASSEMBLY32-NEXT:#define __SCHAR_MAX__ 127
// WEBASSEMBLY32-NEXT:#define __SHRT_MAX__ 32767
// WEBASSEMBLY32-NEXT:#define __SIG_ATOMIC_MAX__ 2147483647L
// WEBASSEMBLY32-NEXT:#define __SIG_ATOMIC_WIDTH__ 32
// WEBASSEMBLY32-NEXT:#define __SIZEOF_DOUBLE__ 8
// WEBASSEMBLY32-NEXT:#define __SIZEOF_FLOAT__ 4
// WEBASSEMBLY32-NEXT:#define __SIZEOF_INT128__ 16
// WEBASSEMBLY32-NEXT:#define __SIZEOF_INT__ 4
// WEBASSEMBLY32-NEXT:#define __SIZEOF_LONG_DOUBLE__ 16
// WEBASSEMBLY32-NEXT:#define __SIZEOF_LONG_LONG__ 8
// WEBASSEMBLY32-NEXT:#define __SIZEOF_LONG__ 4
// WEBASSEMBLY32-NEXT:#define __SIZEOF_POINTER__ 4
// WEBASSEMBLY32-NEXT:#define __SIZEOF_PTRDIFF_T__ 4
// WEBASSEMBLY32-NEXT:#define __SIZEOF_SHORT__ 2
// WEBASSEMBLY32-NEXT:#define __SIZEOF_SIZE_T__ 4
// WEBASSEMBLY32-NEXT:#define __SIZEOF_WCHAR_T__ 4
// WEBASSEMBLY32-NEXT:#define __SIZEOF_WINT_T__ 4
// WEBASSEMBLY32-NEXT:#define __SIZE_FMTX__ "X"
// WEBASSEMBLY32-NEXT:#define __SIZE_FMTo__ "o"
// WEBASSEMBLY32-NEXT:#define __SIZE_FMTu__ "u"
// WEBASSEMBLY32-NEXT:#define __SIZE_FMTx__ "x"
// WEBASSEMBLY32-NEXT:#define __SIZE_MAX__ 4294967295U
// WEBASSEMBLY32-NEXT:#define __SIZE_TYPE__ unsigned int
// WEBASSEMBLY32-NEXT:#define __SIZE_WIDTH__ 32
// WEBASSEMBLY32-NEXT:#define __STDC_HOSTED__ 0
// WEBASSEMBLY32-NOT:#define __STDC_MB_MIGHT_NEQ_WC__
// WEBASSEMBLY32-NOT:#define __STDC_NO_ATOMICS__
// WEBASSEMBLY32-NOT:#define __STDC_NO_COMPLEX__
// WEBASSEMBLY32-NOT:#define __STDC_NO_VLA__
// WEBASSEMBLY32-NOT:#define __STDC_NO_THREADS__
// WEBASSEMBLY32-NEXT:#define __STDC_UTF_16__ 1
// WEBASSEMBLY32-NEXT:#define __STDC_UTF_32__ 1
// WEBASSEMBLY32-NEXT:#define __STDC_VERSION__ 201112L
// WEBASSEMBLY32-NEXT:#define __STDC__ 1
// WEBASSEMBLY32-NEXT:#define __UINT16_C_SUFFIX__
// WEBASSEMBLY32-NEXT:#define __UINT16_FMTX__ "hX"
// WEBASSEMBLY32-NEXT:#define __UINT16_FMTo__ "ho"
// WEBASSEMBLY32-NEXT:#define __UINT16_FMTu__ "hu"
// WEBASSEMBLY32-NEXT:#define __UINT16_FMTx__ "hx"
// WEBASSEMBLY32-NEXT:#define __UINT16_MAX__ 65535
// WEBASSEMBLY32-NEXT:#define __UINT16_TYPE__ unsigned short
// WEBASSEMBLY32-NEXT:#define __UINT32_C_SUFFIX__ U
// WEBASSEMBLY32-NEXT:#define __UINT32_FMTX__ "X"
// WEBASSEMBLY32-NEXT:#define __UINT32_FMTo__ "o"
// WEBASSEMBLY32-NEXT:#define __UINT32_FMTu__ "u"
// WEBASSEMBLY32-NEXT:#define __UINT32_FMTx__ "x"
// WEBASSEMBLY32-NEXT:#define __UINT32_MAX__ 4294967295U
// WEBASSEMBLY32-NEXT:#define __UINT32_TYPE__ unsigned int
// WEBASSEMBLY32-NEXT:#define __UINT64_C_SUFFIX__ ULL
// WEBASSEMBLY32-NEXT:#define __UINT64_FMTX__ "llX"
// WEBASSEMBLY32-NEXT:#define __UINT64_FMTo__ "llo"
// WEBASSEMBLY32-NEXT:#define __UINT64_FMTu__ "llu"
// WEBASSEMBLY32-NEXT:#define __UINT64_FMTx__ "llx"
// WEBASSEMBLY32-NEXT:#define __UINT64_MAX__ 18446744073709551615ULL
// WEBASSEMBLY32-NEXT:#define __UINT64_TYPE__ long long unsigned int
// WEBASSEMBLY32-NEXT:#define __UINT8_C_SUFFIX__
// WEBASSEMBLY32-NEXT:#define __UINT8_FMTX__ "hhX"
// WEBASSEMBLY32-NEXT:#define __UINT8_FMTo__ "hho"
// WEBASSEMBLY32-NEXT:#define __UINT8_FMTu__ "hhu"
// WEBASSEMBLY32-NEXT:#define __UINT8_FMTx__ "hhx"
// WEBASSEMBLY32-NEXT:#define __UINT8_MAX__ 255
// WEBASSEMBLY32-NEXT:#define __UINT8_TYPE__ unsigned char
// WEBASSEMBLY32-NEXT:#define __UINTMAX_C_SUFFIX__ ULL
// WEBASSEMBLY32-NEXT:#define __UINTMAX_FMTX__ "llX"
// WEBASSEMBLY32-NEXT:#define __UINTMAX_FMTo__ "llo"
// WEBASSEMBLY32-NEXT:#define __UINTMAX_FMTu__ "llu"
// WEBASSEMBLY32-NEXT:#define __UINTMAX_FMTx__ "llx"
// WEBASSEMBLY32-NEXT:#define __UINTMAX_MAX__ 18446744073709551615ULL
// WEBASSEMBLY32-NEXT:#define __UINTMAX_TYPE__ long long unsigned int
// WEBASSEMBLY32-NEXT:#define __UINTMAX_WIDTH__ 64
// WEBASSEMBLY32-NEXT:#define __UINTPTR_FMTX__ "X"
// WEBASSEMBLY32-NEXT:#define __UINTPTR_FMTo__ "o"
// WEBASSEMBLY32-NEXT:#define __UINTPTR_FMTu__ "u"
// WEBASSEMBLY32-NEXT:#define __UINTPTR_FMTx__ "x"
// WEBASSEMBLY32-NEXT:#define __UINTPTR_MAX__ 4294967295U
// WEBASSEMBLY32-NEXT:#define __UINTPTR_TYPE__ unsigned int
// WEBASSEMBLY32-NEXT:#define __UINTPTR_WIDTH__ 32
// WEBASSEMBLY32-NEXT:#define __UINT_FAST16_FMTX__ "hX"
// WEBASSEMBLY32-NEXT:#define __UINT_FAST16_FMTo__ "ho"
// WEBASSEMBLY32-NEXT:#define __UINT_FAST16_FMTu__ "hu"
// WEBASSEMBLY32-NEXT:#define __UINT_FAST16_FMTx__ "hx"
// WEBASSEMBLY32-NEXT:#define __UINT_FAST16_MAX__ 65535
// WEBASSEMBLY32-NEXT:#define __UINT_FAST16_TYPE__ unsigned short
// WEBASSEMBLY32-NEXT:#define __UINT_FAST32_FMTX__ "X"
// WEBASSEMBLY32-NEXT:#define __UINT_FAST32_FMTo__ "o"
// WEBASSEMBLY32-NEXT:#define __UINT_FAST32_FMTu__ "u"
// WEBASSEMBLY32-NEXT:#define __UINT_FAST32_FMTx__ "x"
// WEBASSEMBLY32-NEXT:#define __UINT_FAST32_MAX__ 4294967295U
// WEBASSEMBLY32-NEXT:#define __UINT_FAST32_TYPE__ unsigned int
// WEBASSEMBLY32-NEXT:#define __UINT_FAST64_FMTX__ "llX"
// WEBASSEMBLY32-NEXT:#define __UINT_FAST64_FMTo__ "llo"
// WEBASSEMBLY32-NEXT:#define __UINT_FAST64_FMTu__ "llu"
// WEBASSEMBLY32-NEXT:#define __UINT_FAST64_FMTx__ "llx"
// WEBASSEMBLY32-NEXT:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// WEBASSEMBLY32-NEXT:#define __UINT_FAST64_TYPE__ long long unsigned int
// WEBASSEMBLY32-NEXT:#define __UINT_FAST8_FMTX__ "hhX"
// WEBASSEMBLY32-NEXT:#define __UINT_FAST8_FMTo__ "hho"
// WEBASSEMBLY32-NEXT:#define __UINT_FAST8_FMTu__ "hhu"
// WEBASSEMBLY32-NEXT:#define __UINT_FAST8_FMTx__ "hhx"
// WEBASSEMBLY32-NEXT:#define __UINT_FAST8_MAX__ 255
// WEBASSEMBLY32-NEXT:#define __UINT_FAST8_TYPE__ unsigned char
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST16_FMTX__ "hX"
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST16_FMTo__ "ho"
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST16_FMTu__ "hu"
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST16_FMTx__ "hx"
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST16_MAX__ 65535
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST16_TYPE__ unsigned short
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST32_FMTX__ "X"
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST32_FMTo__ "o"
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST32_FMTu__ "u"
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST32_FMTx__ "x"
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST32_MAX__ 4294967295U
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST32_TYPE__ unsigned int
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST64_FMTX__ "llX"
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST64_FMTo__ "llo"
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST64_FMTu__ "llu"
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST64_FMTx__ "llx"
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST64_TYPE__ long long unsigned int
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST8_FMTX__ "hhX"
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST8_FMTo__ "hho"
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST8_FMTu__ "hhu"
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST8_FMTx__ "hhx"
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST8_MAX__ 255
// WEBASSEMBLY32-NEXT:#define __UINT_LEAST8_TYPE__ unsigned char
// WEBASSEMBLY32-NEXT:#define __USER_LABEL_PREFIX__
// WEBASSEMBLY32-NEXT:#define __VERSION__ "{{.*}}"
// WEBASSEMBLY32-NEXT:#define __WCHAR_MAX__ 2147483647
// WEBASSEMBLY32-NEXT:#define __WCHAR_TYPE__ int
// WEBASSEMBLY32-NOT:#define __WCHAR_UNSIGNED__
// WEBASSEMBLY32-NEXT:#define __WCHAR_WIDTH__ 32
// WEBASSEMBLY32-NEXT:#define __WINT_TYPE__ int
// WEBASSEMBLY32-NOT:#define __WINT_UNSIGNED__
// WEBASSEMBLY32-NEXT:#define __WINT_WIDTH__ 32
// WEBASSEMBLY32-NEXT:#define __clang__ 1
// WEBASSEMBLY32-NEXT:#define __clang_major__ {{.*}}
// WEBASSEMBLY32-NEXT:#define __clang_minor__ {{.*}}
// WEBASSEMBLY32-NEXT:#define __clang_patchlevel__ {{.*}}
// WEBASSEMBLY32-NEXT:#define __clang_version__ "{{.*}}"
// WEBASSEMBLY32-NEXT:#define __llvm__ 1
// WEBASSEMBLY32-NOT:#define __wasm_simd128__
// WEBASSEMBLY32-NOT:#define __wasm_simd256__
// WEBASSEMBLY32-NOT:#define __wasm_simd512__
// WEBASSEMBLY32-NOT:#define __unix
// WEBASSEMBLY32-NOT:#define __unix__
// WEBASSEMBLY32-NEXT:#define __wasm 1
// WEBASSEMBLY32-NEXT:#define __wasm32 1
// WEBASSEMBLY32-NEXT:#define __wasm32__ 1
// WEBASSEMBLY32-NOT:#define __wasm64
// WEBASSEMBLY32-NOT:#define __wasm64__
// WEBASSEMBLY32-NEXT:#define __wasm__ 1
//
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=wasm64-unknown-unknown \
// RUN:   < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=WEBASSEMBLY64 %s
//
// WEBASSEMBLY64-NOT:#define _ILP32
// WEBASSEMBLY64:#define _LP64 1
// WEBASSEMBLY64-NEXT:#define __ATOMIC_ACQUIRE 2
// WEBASSEMBLY64-NEXT:#define __ATOMIC_ACQ_REL 4
// WEBASSEMBLY64-NEXT:#define __ATOMIC_CONSUME 1
// WEBASSEMBLY64-NEXT:#define __ATOMIC_RELAXED 0
// WEBASSEMBLY64-NEXT:#define __ATOMIC_RELEASE 3
// WEBASSEMBLY64-NEXT:#define __ATOMIC_SEQ_CST 5
// WEBASSEMBLY64-NEXT:#define __BIGGEST_ALIGNMENT__ 16
// WEBASSEMBLY64-NEXT:#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// WEBASSEMBLY64-NEXT:#define __CHAR16_TYPE__ unsigned short
// WEBASSEMBLY64-NEXT:#define __CHAR32_TYPE__ unsigned int
// WEBASSEMBLY64-NEXT:#define __CHAR_BIT__ 8
// WEBASSEMBLY64-NOT:#define __CHAR_UNSIGNED__
// WEBASSEMBLY64-NEXT:#define __CONSTANT_CFSTRINGS__ 1
// WEBASSEMBLY64-NEXT:#define __DBL_DECIMAL_DIG__ 17
// WEBASSEMBLY64-NEXT:#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
// WEBASSEMBLY64-NEXT:#define __DBL_DIG__ 15
// WEBASSEMBLY64-NEXT:#define __DBL_EPSILON__ 2.2204460492503131e-16
// WEBASSEMBLY64-NEXT:#define __DBL_HAS_DENORM__ 1
// WEBASSEMBLY64-NEXT:#define __DBL_HAS_INFINITY__ 1
// WEBASSEMBLY64-NEXT:#define __DBL_HAS_QUIET_NAN__ 1
// WEBASSEMBLY64-NEXT:#define __DBL_MANT_DIG__ 53
// WEBASSEMBLY64-NEXT:#define __DBL_MAX_10_EXP__ 308
// WEBASSEMBLY64-NEXT:#define __DBL_MAX_EXP__ 1024
// WEBASSEMBLY64-NEXT:#define __DBL_MAX__ 1.7976931348623157e+308
// WEBASSEMBLY64-NEXT:#define __DBL_MIN_10_EXP__ (-307)
// WEBASSEMBLY64-NEXT:#define __DBL_MIN_EXP__ (-1021)
// WEBASSEMBLY64-NEXT:#define __DBL_MIN__ 2.2250738585072014e-308
// WEBASSEMBLY64-NEXT:#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
// WEBASSEMBLY64-NOT:#define __ELF__
// WEBASSEMBLY64-NEXT:#define __FINITE_MATH_ONLY__ 0
// WEBASSEMBLY64-NEXT:#define __FLT_DECIMAL_DIG__ 9
// WEBASSEMBLY64-NEXT:#define __FLT_DENORM_MIN__ 1.40129846e-45F
// WEBASSEMBLY64-NEXT:#define __FLT_DIG__ 6
// WEBASSEMBLY64-NEXT:#define __FLT_EPSILON__ 1.19209290e-7F
// WEBASSEMBLY64-NEXT:#define __FLT_EVAL_METHOD__ 0
// WEBASSEMBLY64-NEXT:#define __FLT_HAS_DENORM__ 1
// WEBASSEMBLY64-NEXT:#define __FLT_HAS_INFINITY__ 1
// WEBASSEMBLY64-NEXT:#define __FLT_HAS_QUIET_NAN__ 1
// WEBASSEMBLY64-NEXT:#define __FLT_MANT_DIG__ 24
// WEBASSEMBLY64-NEXT:#define __FLT_MAX_10_EXP__ 38
// WEBASSEMBLY64-NEXT:#define __FLT_MAX_EXP__ 128
// WEBASSEMBLY64-NEXT:#define __FLT_MAX__ 3.40282347e+38F
// WEBASSEMBLY64-NEXT:#define __FLT_MIN_10_EXP__ (-37)
// WEBASSEMBLY64-NEXT:#define __FLT_MIN_EXP__ (-125)
// WEBASSEMBLY64-NEXT:#define __FLT_MIN__ 1.17549435e-38F
// WEBASSEMBLY64-NEXT:#define __FLT_RADIX__ 2
// WEBASSEMBLY64-NEXT:#define __GCC_ATOMIC_BOOL_LOCK_FREE 2
// WEBASSEMBLY64-NEXT:#define __GCC_ATOMIC_CHAR16_T_LOCK_FREE 2
// WEBASSEMBLY64-NEXT:#define __GCC_ATOMIC_CHAR32_T_LOCK_FREE 2
// WEBASSEMBLY64-NEXT:#define __GCC_ATOMIC_CHAR_LOCK_FREE 2
// WEBASSEMBLY64-NEXT:#define __GCC_ATOMIC_INT_LOCK_FREE 2
// WEBASSEMBLY64-NEXT:#define __GCC_ATOMIC_LLONG_LOCK_FREE 2
// WEBASSEMBLY64-NEXT:#define __GCC_ATOMIC_LONG_LOCK_FREE 2
// WEBASSEMBLY64-NEXT:#define __GCC_ATOMIC_POINTER_LOCK_FREE 2
// WEBASSEMBLY64-NEXT:#define __GCC_ATOMIC_SHORT_LOCK_FREE 2
// WEBASSEMBLY64-NEXT:#define __GCC_ATOMIC_TEST_AND_SET_TRUEVAL 1
// WEBASSEMBLY64-NEXT:#define __GCC_ATOMIC_WCHAR_T_LOCK_FREE 2
// WEBASSEMBLY64-NEXT:#define __GNUC_MINOR__ {{.*}}
// WEBASSEMBLY64-NEXT:#define __GNUC_PATCHLEVEL__ {{.*}}
// WEBASSEMBLY64-NEXT:#define __GNUC_STDC_INLINE__ 1
// WEBASSEMBLY64-NEXT:#define __GNUC__ {{.}}
// WEBASSEMBLY64-NEXT:#define __GXX_ABI_VERSION 1002
// WEBASSEMBLY64-NOT:#define __ILP32__
// WEBASSEMBLY64-NEXT:#define __INT16_C_SUFFIX__
// WEBASSEMBLY64-NEXT:#define __INT16_FMTd__ "hd"
// WEBASSEMBLY64-NEXT:#define __INT16_FMTi__ "hi"
// WEBASSEMBLY64-NEXT:#define __INT16_MAX__ 32767
// WEBASSEMBLY64-NEXT:#define __INT16_TYPE__ short
// WEBASSEMBLY64-NEXT:#define __INT32_C_SUFFIX__
// WEBASSEMBLY64-NEXT:#define __INT32_FMTd__ "d"
// WEBASSEMBLY64-NEXT:#define __INT32_FMTi__ "i"
// WEBASSEMBLY64-NEXT:#define __INT32_MAX__ 2147483647
// WEBASSEMBLY64-NEXT:#define __INT32_TYPE__ int
// WEBASSEMBLY64-NEXT:#define __INT64_C_SUFFIX__ LL
// WEBASSEMBLY64-NEXT:#define __INT64_FMTd__ "lld"
// WEBASSEMBLY64-NEXT:#define __INT64_FMTi__ "lli"
// WEBASSEMBLY64-NEXT:#define __INT64_MAX__ 9223372036854775807LL
// WEBASSEMBLY64-NEXT:#define __INT64_TYPE__ long long int
// WEBASSEMBLY64-NEXT:#define __INT8_C_SUFFIX__
// WEBASSEMBLY64-NEXT:#define __INT8_FMTd__ "hhd"
// WEBASSEMBLY64-NEXT:#define __INT8_FMTi__ "hhi"
// WEBASSEMBLY64-NEXT:#define __INT8_MAX__ 127
// WEBASSEMBLY64-NEXT:#define __INT8_TYPE__ signed char
// WEBASSEMBLY64-NEXT:#define __INTMAX_C_SUFFIX__ LL
// WEBASSEMBLY64-NEXT:#define __INTMAX_FMTd__ "lld"
// WEBASSEMBLY64-NEXT:#define __INTMAX_FMTi__ "lli"
// WEBASSEMBLY64-NEXT:#define __INTMAX_MAX__ 9223372036854775807LL
// WEBASSEMBLY64-NEXT:#define __INTMAX_TYPE__ long long int
// WEBASSEMBLY64-NEXT:#define __INTMAX_WIDTH__ 64
// WEBASSEMBLY64-NEXT:#define __INTPTR_FMTd__ "ld"
// WEBASSEMBLY64-NEXT:#define __INTPTR_FMTi__ "li"
// WEBASSEMBLY64-NEXT:#define __INTPTR_MAX__ 9223372036854775807L
// WEBASSEMBLY64-NEXT:#define __INTPTR_TYPE__ long int
// WEBASSEMBLY64-NEXT:#define __INTPTR_WIDTH__ 64
// WEBASSEMBLY64-NEXT:#define __INT_FAST16_FMTd__ "hd"
// WEBASSEMBLY64-NEXT:#define __INT_FAST16_FMTi__ "hi"
// WEBASSEMBLY64-NEXT:#define __INT_FAST16_MAX__ 32767
// WEBASSEMBLY64-NEXT:#define __INT_FAST16_TYPE__ short
// WEBASSEMBLY64-NEXT:#define __INT_FAST32_FMTd__ "d"
// WEBASSEMBLY64-NEXT:#define __INT_FAST32_FMTi__ "i"
// WEBASSEMBLY64-NEXT:#define __INT_FAST32_MAX__ 2147483647
// WEBASSEMBLY64-NEXT:#define __INT_FAST32_TYPE__ int
// WEBASSEMBLY64-NEXT:#define __INT_FAST64_FMTd__ "lld"
// WEBASSEMBLY64-NEXT:#define __INT_FAST64_FMTi__ "lli"
// WEBASSEMBLY64-NEXT:#define __INT_FAST64_MAX__ 9223372036854775807LL
// WEBASSEMBLY64-NEXT:#define __INT_FAST64_TYPE__ long long int
// WEBASSEMBLY64-NEXT:#define __INT_FAST8_FMTd__ "hhd"
// WEBASSEMBLY64-NEXT:#define __INT_FAST8_FMTi__ "hhi"
// WEBASSEMBLY64-NEXT:#define __INT_FAST8_MAX__ 127
// WEBASSEMBLY64-NEXT:#define __INT_FAST8_TYPE__ signed char
// WEBASSEMBLY64-NEXT:#define __INT_LEAST16_FMTd__ "hd"
// WEBASSEMBLY64-NEXT:#define __INT_LEAST16_FMTi__ "hi"
// WEBASSEMBLY64-NEXT:#define __INT_LEAST16_MAX__ 32767
// WEBASSEMBLY64-NEXT:#define __INT_LEAST16_TYPE__ short
// WEBASSEMBLY64-NEXT:#define __INT_LEAST32_FMTd__ "d"
// WEBASSEMBLY64-NEXT:#define __INT_LEAST32_FMTi__ "i"
// WEBASSEMBLY64-NEXT:#define __INT_LEAST32_MAX__ 2147483647
// WEBASSEMBLY64-NEXT:#define __INT_LEAST32_TYPE__ int
// WEBASSEMBLY64-NEXT:#define __INT_LEAST64_FMTd__ "lld"
// WEBASSEMBLY64-NEXT:#define __INT_LEAST64_FMTi__ "lli"
// WEBASSEMBLY64-NEXT:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// WEBASSEMBLY64-NEXT:#define __INT_LEAST64_TYPE__ long long int
// WEBASSEMBLY64-NEXT:#define __INT_LEAST8_FMTd__ "hhd"
// WEBASSEMBLY64-NEXT:#define __INT_LEAST8_FMTi__ "hhi"
// WEBASSEMBLY64-NEXT:#define __INT_LEAST8_MAX__ 127
// WEBASSEMBLY64-NEXT:#define __INT_LEAST8_TYPE__ signed char
// WEBASSEMBLY64-NEXT:#define __INT_MAX__ 2147483647
// WEBASSEMBLY64-NEXT:#define __LDBL_DECIMAL_DIG__ 36
// WEBASSEMBLY64-NEXT:#define __LDBL_DENORM_MIN__ 6.47517511943802511092443895822764655e-4966L
// WEBASSEMBLY64-NEXT:#define __LDBL_DIG__ 33
// WEBASSEMBLY64-NEXT:#define __LDBL_EPSILON__ 1.92592994438723585305597794258492732e-34L
// WEBASSEMBLY64-NEXT:#define __LDBL_HAS_DENORM__ 1
// WEBASSEMBLY64-NEXT:#define __LDBL_HAS_INFINITY__ 1
// WEBASSEMBLY64-NEXT:#define __LDBL_HAS_QUIET_NAN__ 1
// WEBASSEMBLY64-NEXT:#define __LDBL_MANT_DIG__ 113
// WEBASSEMBLY64-NEXT:#define __LDBL_MAX_10_EXP__ 4932
// WEBASSEMBLY64-NEXT:#define __LDBL_MAX_EXP__ 16384
// WEBASSEMBLY64-NEXT:#define __LDBL_MAX__ 1.18973149535723176508575932662800702e+4932L
// WEBASSEMBLY64-NEXT:#define __LDBL_MIN_10_EXP__ (-4931)
// WEBASSEMBLY64-NEXT:#define __LDBL_MIN_EXP__ (-16381)
// WEBASSEMBLY64-NEXT:#define __LDBL_MIN__ 3.36210314311209350626267781732175260e-4932L
// WEBASSEMBLY64-NEXT:#define __LITTLE_ENDIAN__ 1
// WEBASSEMBLY64-NEXT:#define __LONG_LONG_MAX__ 9223372036854775807LL
// WEBASSEMBLY64-NEXT:#define __LONG_MAX__ 9223372036854775807L
// WEBASSEMBLY64-NEXT:#define __LP64__ 1
// WEBASSEMBLY64-NEXT:#define __NO_INLINE__ 1
// WEBASSEMBLY64-NEXT:#define __OBJC_BOOL_IS_BOOL 0
// WEBASSEMBLY64-NEXT:#define __ORDER_BIG_ENDIAN__ 4321
// WEBASSEMBLY64-NEXT:#define __ORDER_LITTLE_ENDIAN__ 1234
// WEBASSEMBLY64-NEXT:#define __ORDER_PDP_ENDIAN__ 3412
// WEBASSEMBLY64-NEXT:#define __POINTER_WIDTH__ 64
// WEBASSEMBLY64-NEXT:#define __PRAGMA_REDEFINE_EXTNAME 1
// WEBASSEMBLY64-NEXT:#define __PTRDIFF_FMTd__ "ld"
// WEBASSEMBLY64-NEXT:#define __PTRDIFF_FMTi__ "li"
// WEBASSEMBLY64-NEXT:#define __PTRDIFF_MAX__ 9223372036854775807L
// WEBASSEMBLY64-NEXT:#define __PTRDIFF_TYPE__ long int
// WEBASSEMBLY64-NEXT:#define __PTRDIFF_WIDTH__ 64
// WEBASSEMBLY64-NOT:#define __REGISTER_PREFIX__
// WEBASSEMBLY64-NEXT:#define __SCHAR_MAX__ 127
// WEBASSEMBLY64-NEXT:#define __SHRT_MAX__ 32767
// WEBASSEMBLY64-NEXT:#define __SIG_ATOMIC_MAX__ 9223372036854775807L
// WEBASSEMBLY64-NEXT:#define __SIG_ATOMIC_WIDTH__ 64
// WEBASSEMBLY64-NEXT:#define __SIZEOF_DOUBLE__ 8
// WEBASSEMBLY64-NEXT:#define __SIZEOF_FLOAT__ 4
// WEBASSEMBLY64-NEXT:#define __SIZEOF_INT128__ 16
// WEBASSEMBLY64-NEXT:#define __SIZEOF_INT__ 4
// WEBASSEMBLY64-NEXT:#define __SIZEOF_LONG_DOUBLE__ 16
// WEBASSEMBLY64-NEXT:#define __SIZEOF_LONG_LONG__ 8
// WEBASSEMBLY64-NEXT:#define __SIZEOF_LONG__ 8
// WEBASSEMBLY64-NEXT:#define __SIZEOF_POINTER__ 8
// WEBASSEMBLY64-NEXT:#define __SIZEOF_PTRDIFF_T__ 8
// WEBASSEMBLY64-NEXT:#define __SIZEOF_SHORT__ 2
// WEBASSEMBLY64-NEXT:#define __SIZEOF_SIZE_T__ 8
// WEBASSEMBLY64-NEXT:#define __SIZEOF_WCHAR_T__ 4
// WEBASSEMBLY64-NEXT:#define __SIZEOF_WINT_T__ 4
// WEBASSEMBLY64-NEXT:#define __SIZE_FMTX__ "lX"
// WEBASSEMBLY64-NEXT:#define __SIZE_FMTo__ "lo"
// WEBASSEMBLY64-NEXT:#define __SIZE_FMTu__ "lu"
// WEBASSEMBLY64-NEXT:#define __SIZE_FMTx__ "lx"
// WEBASSEMBLY64-NEXT:#define __SIZE_MAX__ 18446744073709551615UL
// WEBASSEMBLY64-NEXT:#define __SIZE_TYPE__ long unsigned int
// WEBASSEMBLY64-NEXT:#define __SIZE_WIDTH__ 64
// WEBASSEMBLY64-NEXT:#define __STDC_HOSTED__ 0
// WEBASSEMBLY64-NOT:#define __STDC_MB_MIGHT_NEQ_WC__
// WEBASSEMBLY64-NOT:#define __STDC_NO_ATOMICS__
// WEBASSEMBLY64-NOT:#define __STDC_NO_COMPLEX__
// WEBASSEMBLY64-NOT:#define __STDC_NO_VLA__
// WEBASSEMBLY64-NOT:#define __STDC_NO_THREADS__
// WEBASSEMBLY64-NEXT:#define __STDC_UTF_16__ 1
// WEBASSEMBLY64-NEXT:#define __STDC_UTF_32__ 1
// WEBASSEMBLY64-NEXT:#define __STDC_VERSION__ 201112L
// WEBASSEMBLY64-NEXT:#define __STDC__ 1
// WEBASSEMBLY64-NEXT:#define __UINT16_C_SUFFIX__
// WEBASSEMBLY64-NEXT:#define __UINT16_FMTX__ "hX"
// WEBASSEMBLY64-NEXT:#define __UINT16_FMTo__ "ho"
// WEBASSEMBLY64-NEXT:#define __UINT16_FMTu__ "hu"
// WEBASSEMBLY64-NEXT:#define __UINT16_FMTx__ "hx"
// WEBASSEMBLY64-NEXT:#define __UINT16_MAX__ 65535
// WEBASSEMBLY64-NEXT:#define __UINT16_TYPE__ unsigned short
// WEBASSEMBLY64-NEXT:#define __UINT32_C_SUFFIX__ U
// WEBASSEMBLY64-NEXT:#define __UINT32_FMTX__ "X"
// WEBASSEMBLY64-NEXT:#define __UINT32_FMTo__ "o"
// WEBASSEMBLY64-NEXT:#define __UINT32_FMTu__ "u"
// WEBASSEMBLY64-NEXT:#define __UINT32_FMTx__ "x"
// WEBASSEMBLY64-NEXT:#define __UINT32_MAX__ 4294967295U
// WEBASSEMBLY64-NEXT:#define __UINT32_TYPE__ unsigned int
// WEBASSEMBLY64-NEXT:#define __UINT64_C_SUFFIX__ ULL
// WEBASSEMBLY64-NEXT:#define __UINT64_FMTX__ "llX"
// WEBASSEMBLY64-NEXT:#define __UINT64_FMTo__ "llo"
// WEBASSEMBLY64-NEXT:#define __UINT64_FMTu__ "llu"
// WEBASSEMBLY64-NEXT:#define __UINT64_FMTx__ "llx"
// WEBASSEMBLY64-NEXT:#define __UINT64_MAX__ 18446744073709551615ULL
// WEBASSEMBLY64-NEXT:#define __UINT64_TYPE__ long long unsigned int
// WEBASSEMBLY64-NEXT:#define __UINT8_C_SUFFIX__
// WEBASSEMBLY64-NEXT:#define __UINT8_FMTX__ "hhX"
// WEBASSEMBLY64-NEXT:#define __UINT8_FMTo__ "hho"
// WEBASSEMBLY64-NEXT:#define __UINT8_FMTu__ "hhu"
// WEBASSEMBLY64-NEXT:#define __UINT8_FMTx__ "hhx"
// WEBASSEMBLY64-NEXT:#define __UINT8_MAX__ 255
// WEBASSEMBLY64-NEXT:#define __UINT8_TYPE__ unsigned char
// WEBASSEMBLY64-NEXT:#define __UINTMAX_C_SUFFIX__ ULL
// WEBASSEMBLY64-NEXT:#define __UINTMAX_FMTX__ "llX"
// WEBASSEMBLY64-NEXT:#define __UINTMAX_FMTo__ "llo"
// WEBASSEMBLY64-NEXT:#define __UINTMAX_FMTu__ "llu"
// WEBASSEMBLY64-NEXT:#define __UINTMAX_FMTx__ "llx"
// WEBASSEMBLY64-NEXT:#define __UINTMAX_MAX__ 18446744073709551615ULL
// WEBASSEMBLY64-NEXT:#define __UINTMAX_TYPE__ long long unsigned int
// WEBASSEMBLY64-NEXT:#define __UINTMAX_WIDTH__ 64
// WEBASSEMBLY64-NEXT:#define __UINTPTR_FMTX__ "lX"
// WEBASSEMBLY64-NEXT:#define __UINTPTR_FMTo__ "lo"
// WEBASSEMBLY64-NEXT:#define __UINTPTR_FMTu__ "lu"
// WEBASSEMBLY64-NEXT:#define __UINTPTR_FMTx__ "lx"
// WEBASSEMBLY64-NEXT:#define __UINTPTR_MAX__ 18446744073709551615UL
// WEBASSEMBLY64-NEXT:#define __UINTPTR_TYPE__ long unsigned int
// WEBASSEMBLY64-NEXT:#define __UINTPTR_WIDTH__ 64
// WEBASSEMBLY64-NEXT:#define __UINT_FAST16_FMTX__ "hX"
// WEBASSEMBLY64-NEXT:#define __UINT_FAST16_FMTo__ "ho"
// WEBASSEMBLY64-NEXT:#define __UINT_FAST16_FMTu__ "hu"
// WEBASSEMBLY64-NEXT:#define __UINT_FAST16_FMTx__ "hx"
// WEBASSEMBLY64-NEXT:#define __UINT_FAST16_MAX__ 65535
// WEBASSEMBLY64-NEXT:#define __UINT_FAST16_TYPE__ unsigned short
// WEBASSEMBLY64-NEXT:#define __UINT_FAST32_FMTX__ "X"
// WEBASSEMBLY64-NEXT:#define __UINT_FAST32_FMTo__ "o"
// WEBASSEMBLY64-NEXT:#define __UINT_FAST32_FMTu__ "u"
// WEBASSEMBLY64-NEXT:#define __UINT_FAST32_FMTx__ "x"
// WEBASSEMBLY64-NEXT:#define __UINT_FAST32_MAX__ 4294967295U
// WEBASSEMBLY64-NEXT:#define __UINT_FAST32_TYPE__ unsigned int
// WEBASSEMBLY64-NEXT:#define __UINT_FAST64_FMTX__ "llX"
// WEBASSEMBLY64-NEXT:#define __UINT_FAST64_FMTo__ "llo"
// WEBASSEMBLY64-NEXT:#define __UINT_FAST64_FMTu__ "llu"
// WEBASSEMBLY64-NEXT:#define __UINT_FAST64_FMTx__ "llx"
// WEBASSEMBLY64-NEXT:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// WEBASSEMBLY64-NEXT:#define __UINT_FAST64_TYPE__ long long unsigned int
// WEBASSEMBLY64-NEXT:#define __UINT_FAST8_FMTX__ "hhX"
// WEBASSEMBLY64-NEXT:#define __UINT_FAST8_FMTo__ "hho"
// WEBASSEMBLY64-NEXT:#define __UINT_FAST8_FMTu__ "hhu"
// WEBASSEMBLY64-NEXT:#define __UINT_FAST8_FMTx__ "hhx"
// WEBASSEMBLY64-NEXT:#define __UINT_FAST8_MAX__ 255
// WEBASSEMBLY64-NEXT:#define __UINT_FAST8_TYPE__ unsigned char
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST16_FMTX__ "hX"
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST16_FMTo__ "ho"
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST16_FMTu__ "hu"
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST16_FMTx__ "hx"
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST16_MAX__ 65535
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST16_TYPE__ unsigned short
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST32_FMTX__ "X"
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST32_FMTo__ "o"
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST32_FMTu__ "u"
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST32_FMTx__ "x"
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST32_MAX__ 4294967295U
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST32_TYPE__ unsigned int
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST64_FMTX__ "llX"
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST64_FMTo__ "llo"
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST64_FMTu__ "llu"
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST64_FMTx__ "llx"
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST64_TYPE__ long long unsigned int
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST8_FMTX__ "hhX"
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST8_FMTo__ "hho"
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST8_FMTu__ "hhu"
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST8_FMTx__ "hhx"
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST8_MAX__ 255
// WEBASSEMBLY64-NEXT:#define __UINT_LEAST8_TYPE__ unsigned char
// WEBASSEMBLY64-NEXT:#define __USER_LABEL_PREFIX__
// WEBASSEMBLY64-NEXT:#define __VERSION__ "{{.*}}"
// WEBASSEMBLY64-NEXT:#define __WCHAR_MAX__ 2147483647
// WEBASSEMBLY64-NEXT:#define __WCHAR_TYPE__ int
// WEBASSEMBLY64-NOT:#define __WCHAR_UNSIGNED__
// WEBASSEMBLY64-NEXT:#define __WCHAR_WIDTH__ 32
// WEBASSEMBLY64-NEXT:#define __WINT_TYPE__ int
// WEBASSEMBLY64-NOT:#define __WINT_UNSIGNED__
// WEBASSEMBLY64-NEXT:#define __WINT_WIDTH__ 32
// WEBASSEMBLY64-NEXT:#define __clang__ 1
// WEBASSEMBLY64-NEXT:#define __clang_major__ {{.*}}
// WEBASSEMBLY64-NEXT:#define __clang_minor__ {{.*}}
// WEBASSEMBLY64-NEXT:#define __clang_patchlevel__ {{.*}}
// WEBASSEMBLY64-NEXT:#define __clang_version__ "{{.*}}"
// WEBASSEMBLY64-NEXT:#define __llvm__ 1
// WEBASSEMBLY64-NOT:#define __wasm_simd128__
// WEBASSEMBLY64-NOT:#define __wasm_simd256__
// WEBASSEMBLY64-NOT:#define __wasm_simd512__
// WEBASSEMBLY64-NOT:#define __unix
// WEBASSEMBLY64-NOT:#define __unix__
// WEBASSEMBLY64-NEXT:#define __wasm 1
// WEBASSEMBLY64-NOT:#define __wasm32
// WEBASSEMBLY64-NOT:#define __wasm32__
// WEBASSEMBLY64-NEXT:#define __wasm64 1
// WEBASSEMBLY64-NEXT:#define __wasm64__ 1
// WEBASSEMBLY64-NEXT:#define __wasm__ 1

// RUN: %clang_cc1 -E -dM -ffreestanding -triple i686-windows-cygnus < /dev/null | FileCheck -match-full-lines -check-prefix CYGWIN-X32 %s
// CYGWIN-X32: #define __USER_LABEL_PREFIX__ _

// RUN: %clang_cc1 -E -dM -ffreestanding -triple x86_64-windows-cygnus < /dev/null | FileCheck -match-full-lines -check-prefix CYGWIN-X64 %s
// CYGWIN-X64: #define __USER_LABEL_PREFIX__

// RUN: %clang_cc1 -E -dM -ffreestanding -triple=avr \
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
