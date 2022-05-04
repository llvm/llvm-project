// RUN: not %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.6 %s 2>&1 | FileCheck %s --check-prefix NO-TLS
// RUN: %clang_cc1 -fsyntax-only -Wno-error=implicit-int -triple x86_64-apple-macosx10.7 %s 2>&1 | FileCheck %s --check-prefix TLS

// RUN: not %clang_cc1 -fsyntax-only -triple arm64-apple-ios7.1 %s 2>&1 | FileCheck %s --check-prefix NO-TLS
// RUN: %clang_cc1 -fsyntax-only -Wno-error=implicit-int -triple arm64-apple-ios8.0 %s 2>&1 | FileCheck %s --check-prefix TLS
// RUN: not %clang_cc1 -fsyntax-only -triple thumbv7s-apple-ios8.3 %s 2>&1 | FileCheck %s --check-prefix NO-TLS
// RUN: %clang_cc1 -fsyntax-only -Wno-error=implicit-int -triple thumbv7s-apple-ios9.0 %s 2>&1 | FileCheck %s --check-prefix TLS
// RUN: %clang_cc1 -fsyntax-only -Wno-error=implicit-int -triple armv7-apple-ios9.0 %s 2>&1 | FileCheck %s --check-prefix TLS
// RUN: not %clang_cc1 -fsyntax-only -triple i386-apple-ios9.0-simulator %s 2>&1 | FileCheck %s --check-prefix NO-TLS
// RUN: %clang_cc1 -fsyntax-only -Wno-error=implicit-int -triple i386-apple-ios10.0-simulator %s 2>&1 | FileCheck %s --check-prefix TLS

// RUN: not %clang_cc1 -fsyntax-only -triple thumbv7k-apple-watchos1.0 %s 2>&1 | FileCheck %s --check-prefix NO-TLS
// RUN: %clang_cc1 -fsyntax-only -Wno-error=implicit-int -triple thumbv7k-apple-watchos2.0 %s 2>&1 | FileCheck %s --check-prefix TLS
// RUN: not %clang_cc1 -fsyntax-only -triple i386-apple-watchos2.0-simulator %s 2>&1 | FileCheck %s --check-prefix NO-TLS
// RUN: %clang_cc1 -fsyntax-only -Wno-error=implicit-int -triple i386-apple-watchos3.0-simulator %s 2>&1 | FileCheck %s --check-prefix TLS


__thread int a;

// NO-TLS: thread-local storage is not supported for the current target
// TLS-NOT: thread-local storage is not supported for the current target

wibble;
