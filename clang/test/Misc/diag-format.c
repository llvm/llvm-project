// RUN: %clang -fsyntax-only  %s 2>&1 | FileCheck %s --strict-whitespace -check-prefix=DEFAULT
// RUN: %clang -fsyntax-only -fdiagnostics-format=clang %s 2>&1 | FileCheck %s --strict-whitespace -check-prefix=DEFAULT
// RUN: %clang -fsyntax-only -fdiagnostics-format=clang -target x86_64-pc-win32 %s 2>&1 | FileCheck %s --strict-whitespace -check-prefix=DEFAULT
//
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fmsc-version=1300  %s 2>&1 | FileCheck %s --strict-whitespace -check-prefix=MSVC2010
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fms-compatibility-version=13.00  %s 2>&1 | FileCheck %s --strict-whitespace -check-prefix=MSVC2010
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fmsc-version=1300 -target x86_64-pc-win32 %s 2>&1 | FileCheck %s --strict-whitespace -check-prefix=MSVC2010
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fms-compatibility-version=13.00 -target x86_64-pc-win32 %s 2>&1 | FileCheck %s --strict-whitespace -check-prefix=MSVC2010
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fmsc-version=1300 -target x86_64-pc-win32 -fshow-column %s 2>&1 | FileCheck %s --strict-whitespace -check-prefix=MSVC2010
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fmsc-version=1800 -target x86_64-pc-win32 %s 2>&1 | FileCheck %s --strict-whitespace -check-prefix=MSVC2013
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -target x86_64-pc-win32 %s 2>&1 | FileCheck %s --strict-whitespace -check-prefix=MSVC
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fmsc-version=1900 -target x86_64-pc-win32 %s 2>&1 | FileCheck %s --strict-whitespace -check-prefix=MSVC2015
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fms-compatibility-version=13.00 -target x86_64-pc-win32 -fshow-column %s 2>&1 | FileCheck %s --strict-whitespace -check-prefix=MSVC2010
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fmsc-version=1800 -target x86_64-pc-win32 -fshow-column %s 2>&1 | FileCheck %s --strict-whitespace -check-prefix=MSVC2013
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -target x86_64-pc-win32 -fshow-column %s 2>&1 | FileCheck %s --strict-whitespace -check-prefix=MSVC
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fmsc-version=1900 -target x86_64-pc-win32 -fshow-column %s 2>&1 | FileCheck %s --strict-whitespace -check-prefix=MSVC2015
//
// RUN: %clang -fsyntax-only -fdiagnostics-format=vi    %s 2>&1 | FileCheck %s --strict-whitespace -check-prefix=VI
//
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fno-show-column -fmsc-version=1900 %s 2>&1 | FileCheck %s --strict-whitespace -check-prefix=MSVC2015_ORIG
//
// RUN: %clang -fsyntax-only -fno-show-column %s 2>&1 | FileCheck %s --strict-whitespace -check-prefix=NO_COLUMN












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
int x;
