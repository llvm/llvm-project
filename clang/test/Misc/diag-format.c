// RUN: %clang -fsyntax-only  %s 2>&1 | FileCheck %s -check-prefix=DEFAULT
// RUN: %clang -fsyntax-only -fdiagnostics-format=clang %s 2>&1 | FileCheck %s -check-prefix=DEFAULT
// RUN: %clang -fsyntax-only -fdiagnostics-format=clang -target x86_64-pc-win32 %s 2>&1 | FileCheck %s -check-prefix=DEFAULT
//
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fmsc-version=1300  %s 2>&1 | FileCheck %s -check-prefix=MSVC2010
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fms-compatibility-version=13.00  %s 2>&1 | FileCheck %s -check-prefix=MSVC2010
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fmsc-version=1300 -target x86_64-pc-win32 %s 2>&1 | FileCheck %s -check-prefix=MSVC2010
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fms-compatibility-version=13.00 -target x86_64-pc-win32 %s 2>&1 | FileCheck %s -check-prefix=MSVC2010
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fmsc-version=1300 -target x86_64-pc-win32 -fshow-column %s 2>&1 | FileCheck %s -check-prefix=MSVC2010
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fmsc-version=1800 -target x86_64-pc-win32 %s 2>&1 | FileCheck %s -check-prefix=MSVC2013
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -target x86_64-pc-win32 %s 2>&1 | FileCheck %s -check-prefix=MSVC
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fmsc-version=1900 -target x86_64-pc-win32 %s 2>&1 | FileCheck %s -check-prefix=MSVC2015
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fms-compatibility-version=13.00 -target x86_64-pc-win32 -fshow-column %s 2>&1 | FileCheck %s -check-prefix=MSVC2010
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fmsc-version=1800 -target x86_64-pc-win32 -fshow-column %s 2>&1 | FileCheck %s -check-prefix=MSVC2013
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -target x86_64-pc-win32 -fshow-column %s 2>&1 | FileCheck %s -check-prefix=MSVC
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fmsc-version=1900 -target x86_64-pc-win32 -fshow-column %s 2>&1 | FileCheck %s -check-prefix=MSVC2015
//
// RUN: %clang -fsyntax-only -fdiagnostics-format=vi    %s 2>&1 | FileCheck %s -check-prefix=VI
//
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fno-show-column -fmsc-version=1900 %s 2>&1 | FileCheck %s -check-prefix=MSVC2015_ORIG
//
// RUN: %clang -fsyntax-only -fno-show-column %s 2>&1 | FileCheck %s -check-prefix=NO_COLUMN
//
// RUN: not %clang -fsyntax-only -Werror -fdiagnostics-format=msvc-fallback -fmsc-version=1300 %s 2>&1 | FileCheck %s -check-prefix=MSVC2010-FALLBACK
// RUN: not %clang -fsyntax-only -Werror -fdiagnostics-format=msvc-fallback -fms-compatibility-version=13.00 %s 2>&1 | FileCheck %s -check-prefix=MSVC2010-FALLBACK
// RUN: not %clang -fsyntax-only -Werror -fdiagnostics-format=msvc-fallback -fmsc-version=1800 %s 2>&1 | FileCheck %s -check-prefix=MSVC2013-FALLBACK
// RUN: not %clang -fsyntax-only -Werror -fdiagnostics-format=msvc-fallback -fmsc-version=1900 %s 2>&1 | FileCheck %s -check-prefix=MSVC2015-FALLBACK







#ifdef foo
#endif bad // extension!
// DEFAULT: {{.*}}:36:8: warning: extra tokens at end of #endif directive [-Wextra-tokens]
// MSVC2010: {{.*}}(36,7) : warning: extra tokens at end of #endif directive [-Wextra-tokens]
// MSVC2013: {{.*}}(36,8) : warning: extra tokens at end of #endif directive [-Wextra-tokens]
// MSVC: {{.*\(36,[78]\) ?}}: warning: extra tokens at end of #endif directive [-Wextra-tokens]
// MSVC2015: {{.*}}(36,8): warning: extra tokens at end of #endif directive [-Wextra-tokens]
// VI: {{.*}} +36:8: warning: extra tokens at end of #endif directive [-Wextra-tokens]
// MSVC2015_ORIG: {{.*}}(36): warning: extra tokens at end of #endif directive [-Wextra-tokens]
// NO_COLUMN: {{.*}}:36: warning: extra tokens at end of #endif directive [-Wextra-tokens]
// MSVC2010-FALLBACK: {{.*}}(36,7) : error(clang): extra tokens at end of #endif directive
// MSVC2013-FALLBACK: {{.*}}(36,8) : error(clang): extra tokens at end of #endif directive
// MSVC2015-FALLBACK: {{.*}}(36,8): error(clang): extra tokens at end of #endif directive
int x;
