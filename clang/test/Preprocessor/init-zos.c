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
