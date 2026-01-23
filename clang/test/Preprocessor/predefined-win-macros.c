// This test verifies that the correct macros are predefined.
//
// RUN: %clang_cc1 %s -x c++ -E -dM -triple x86_64-pc-win32 -fms-extensions -fms-compatibility \
// RUN:     -fms-compatibility-version=19.00 -std=c++14 -o - | FileCheck -match-full-lines %s --check-prefix=CHECK-MS64
// RUN: %clang_cc1 %s -x c++ -E -dM -triple x86_64-pc-win32 -fms-extensions -fms-compatibility \
// RUN:     -fms-compatibility-version=19.00 -std=c++14 -o - | grep GCC | count 7
// CHECK-MS64: #define _INTEGRAL_MAX_BITS 64
// CHECK-MS64: #define _ISO_VOLATILE 1
// CHECK-MS64: #define _MSC_EXTENSIONS 1
// CHECK-MS64: #define _MSC_VER 1900
// CHECK-MS64: #define _MSVC_LANG 201402L
// CHECK-MS64: #define _M_AMD64 100
// CHECK-MS64: #define _M_X64 100
// CHECK-MS64: #define _WIN64 1
// CHECK-MS64-NOT: #define __STRICT_ANSI__
// CHECK-MS64-NOT: GNU
// CHECK-MS64-NOT: GXX
// CHECK-MS64: #define __GCC_ASM_FLAG_OUTPUTS__ 1
// CHECK-MS64: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1
// CHECK-MS64: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 1
// CHECK-MS64: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
// CHECK-MS64: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 1
// CHECK-MS64-NOT: GNU
// CHECK-MS64-NOT: GXX

// RUN: %clang_cc1 %s -x c++ -E -dM -triple i686-pc-win32 -fms-extensions -fms-compatibility \
// RUN:     -fms-compatibility-version=19.00 -std=c++17 -o - | FileCheck -match-full-lines %s --check-prefix=CHECK-MS
// RUN: %clang_cc1 %s -x c++ -E -dM -triple i686-pc-win32 -fms-extensions -fms-compatibility \
// RUN:     -fms-compatibility-version=19.00 -std=c++17 -o - | grep GCC | count 7
// CHECK-MS: #define _INTEGRAL_MAX_BITS 64
// CHECK-MS: #define _ISO_VOLATILE 1
// CHECK-MS: #define _MSC_EXTENSIONS 1
// CHECK-MS: #define _MSC_VER 1900
// CHECK-MS: #define _MSVC_LANG 201703L
// CHECK-MS: #define _M_IX86 600
// CHECK-MS: #define _M_IX86_FP 0
// CHECK-MS: #define _WIN32 1
// CHECK-MS-NOT: #define __STRICT_ANSI__
// CHECK-MS-NOT: GNU
// CHECK-MS-NOT: GXX
// CHECK-MS: #define __GCC_ASM_FLAG_OUTPUTS__ 1
// CHECK-MS: #define __GCC_CONSTRUCTIVE_SIZE {{.+}}
// CHECK-MS: #define __GCC_DESTRUCTIVE_SIZE {{.+}}
// CHECK-MS: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1
// CHECK-MS: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 1
// CHECK-MS: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
// CHECK-MS: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 1
// CHECK-MS-NOT: GNU
// CHECK-MS-NOT: GXX

// RUN: %clang_cc1 %s -x c++ -E -dM -triple i686-pc-win32 -fms-extensions -fms-compatibility \
// RUN:     -fms-compatibility-version=19.00 -std=c++20 -o - | FileCheck -match-full-lines %s --check-prefix=CHECK-MS-CPP20
// CHECK-MS-CPP20: #define _MSC_VER 1900
// CHECK-MS-CPP20: #define _MSVC_LANG 202002L

// RUN: %clang_cc1 %s -x c++ -E -dM -triple i686-pc-win32 -fms-extensions -fms-compatibility \
// RUN:     -fms-compatibility-version=19.00 -std=c++23 -o - | FileCheck -match-full-lines %s --check-prefix=CHECK-MS-CPP2B
// CHECK-MS-CPP2B: #define _MSC_VER 1900
// CHECK-MS-CPP2B: #define _MSVC_LANG 202302L

// RUN: %clang_cc1 %s -x c++ -E -dM -triple i686-pc-win32 -fms-extensions -fms-compatibility \
// RUN:     -fms-compatibility-version=19.00 -std=c++26 -o - | FileCheck -match-full-lines %s --check-prefix=CHECK-MS-CPP2C
// CHECK-MS-CPP2C: #define _MSC_VER 1900
// CHECK-MS-CPP2C: #define _MSVC_LANG 202400L

// RUN: %clang_cc1 -triple i386-windows %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-X86-WIN

// CHECK-X86-WIN-NOT: #define WIN32 1
// CHECK-X86-WIN-NOT: #define WIN64 1
// CHECK-X86-WIN-NOT: #define WINNT 1
// CHECK-X86-WIN: #define _WIN32 1
// CHECK-X86-WIN-NOT: #define _WIN64 1

// RUN: %clang_cc1 -triple thumbv7-windows %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-ARM-WIN

// CHECK-ARM-WIN-NOT: #define WIN32 1
// CHECK-ARM-WIN-NOT: #define WIN64 1
// CHECK-ARM-WIN-NOT: #define WINNT 1
// CHECK-ARM-WIN: #define _WIN32 1
// CHECK-ARM-WIN-NOT: #define _WIN64 1

// RUN: %clang_cc1 -triple x86_64-windows %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-AMD64-WIN

// CHECK-AMD64-WIN-NOT: #define WIN32 1
// CHECK-AMD64-WIN-NOT: #define WIN64 1
// CHECK-AMD64-WIN-NOT: #define WINNT 1
// CHECK-AMD64-WIN: #define _WIN32 1
// CHECK-AMD64-WIN: #define _WIN64 1

// RUN: %clang_cc1 -triple aarch64-windows %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-ARM64-WIN

// CHECK-ARM64-WIN-NOT: #define WIN32 1
// CHECK-ARM64-WIN-NOT: #define WIN64 1
// CHECK-ARM64-WIN-NOT: #define WINNT 1
// CHECK-ARM64-WIN: #define _M_ARM64 1
// CHECK-ARM64-WIN: #define _WIN32 1
// CHECK-ARM64-WIN: #define _WIN64 1

// RUN: %clang_cc1 -triple arm64ec-windows %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-ARM64EC-WIN

// CHECK-ARM64EC-WIN-NOT: #define WIN32 1
// CHECK-ARM64EC-WIN-NOT: #define WIN64 1
// CHECK-ARM64EC-WIN-NOT: #define WINNT 1
// CHECK-ARM64EC-WIN-NOT: #define _M_ARM64 1
// CHECK-ARM64EC-WIN: #define _M_AMD64 100
// CHECK-ARM64EC-WIN: #define _M_ARM64EC 1
// CHECK-ARM64EC-WIN: #define _M_X64 100
// CHECK-ARM64EC-WIN: #define _WIN32 1
// CHECK-ARM64EC-WIN: #define _WIN64 1

// RUN: %clang_cc1 -triple mipsel-windows %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-MIPSEL-WIN

// CHECK-MIPSEL-WIN: #define _M_MRX000 4000
// CHECK-MIPSEL-WIN: #define _WIN32 1
// CHECK-MIPSEL-WIN-NOT: #define _MIPS_ 1

// RUN: %clang_cc1 -triple i686-windows-gnu %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-X86-MINGW

// CHECK-X86-MINGW: #define WIN32 1
// CHECK-X86-MINGW-NOT: #define WIN64 1
// CHECK-X86-MINGW: #define WINNT 1
// CHECK-X86-MINGW: #define _WIN32 1
// CHECK-X86-MINGW-NOT: #define _WIN64 1
// CHECK-X86-MINGW: #define __GXX_TYPEINFO_EQUALITY_INLINE 0

// RUN: %clang_cc1 -triple thumbv7-windows-gnu %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-ARM-MINGW

// CHECK-ARM-MINGW: #define WIN32 1
// CHECK-ARM-MINGW-NOT: #define WIN64 1
// CHECK-ARM-MINGW: #define WINNT 1
// CHECK-ARM-MINGW: #define _WIN32 1
// CHECK-ARM-MINGW-NOT: #define _WIN64 1
// CHECK-ARM-MINGW: #define __GXX_TYPEINFO_EQUALITY_INLINE 0

// RUN: %clang_cc1 -triple x86_64-windows-gnu %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-AMD64-MINGW

// CHECK-AMD64-MINGW: #define WIN32 1
// CHECK-AMD64-MINGW: #define WIN64 1
// CHECK-AMD64-MINGW: #define WINNT 1
// CHECK-AMD64-MINGW: #define _WIN32 1
// CHECK-AMD64-MINGW: #define _WIN64 1
// CHECK-AMD64-MINGW: #define __GXX_TYPEINFO_EQUALITY_INLINE 0

// RUN: %clang_cc1 -triple aarch64-windows-gnu %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-ARM64-MINGW

// CHECK-ARM64-MINGW-NOT: #define _M_ARM64 1
// CHECK-ARM64-MINGW: #define WIN32 1
// CHECK-ARM64-MINGW: #define WIN64 1
// CHECK-ARM64-MINGW: #define WINNT 1
// CHECK-ARM64-MINGW: #define _WIN32 1
// CHECK-ARM64-MINGW: #define _WIN64 1
// CHECK-ARM64-MINGW: #define __GCC_ASM_FLAG_OUTPUTS__ 1
// CHECK-ARM64-MINGW: #define __GXX_TYPEINFO_EQUALITY_INLINE 0
// CHECK-ARM64-MINGW: #define __aarch64__ 1

// RUN: %clang_cc1 -triple arm64ec-windows-gnu %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-ARM64EC-MINGW

// CHECK-ARM64EC-MINGW-NOT: #define _M_ARM64EC 1
// CHECK-ARM64EC-MINGW: #define WIN32 1
// CHECK-ARM64EC-MINGW: #define WIN64 1
// CHECK-ARM64EC-MINGW: #define WINNT 1
// CHECK-ARM64EC-MINGW: #define _WIN32 1
// CHECK-ARM64EC-MINGW: #define _WIN64 1
// CHECK-ARM64EC-MINGW: #define __GCC_ASM_FLAG_OUTPUTS__ 1
// CHECK-ARM64EC-MINGW: #define __GXX_TYPEINFO_EQUALITY_INLINE 0
// CHECK-ARM64EC-MINGW-NOT: #define __aarch64__ 1
// CHECK-ARM64EC-MINGW: #define __amd64 1
// CHECK-ARM64EC-MINGW: #define __amd64__ 1
// CHECK-ARM64EC-MINGW: #define __arm64ec__ 1
// CHECK-ARM64EC-MINGW: #define __x86_64 1
// CHECK-ARM64EC-MINGW: #define __x86_64__ 1

// RUN: %clang_cc1 -triple mipsel-windows-gnu %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-MIPSEL-MINGW

// CHECK-MIPSEL-MINGW-NOT: #define _M_MRX000 4000
// CHECK-MIPSEL-MINGW: #define _MIPS_ 1
// CHECK-MIPSEL-MINGW: #define _WIN32 1
// CHECK-MIPSEL-MINGW: #define __mips 32
// CHECK-MIPSEL-MINGW: #define __mips__ 1
