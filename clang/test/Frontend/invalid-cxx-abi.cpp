// REQUIRES: x86-registered-target

// These should succeed.
// RUN: %clang -c -fc++-abi=itanium -target x86_64-unknown-linux-gnu %s
// RUN: %clang -c -fc++-abi=fuchsia -target x86_64-unknown-fuchsia %s
// RUN: %clang -c -fc++-abi=microsoft -target x86_64-windows-msvc %s
// RUN: %clang_cc1 -fc++-abi=itanium -triple x86_64-unknown-linux-gnu %s
// RUN: %clang_cc1 -fc++-abi=fuchsia -triple x86_64-unknown-fuchsia %s
// RUN: %clang_cc1 -fc++-abi=microsoft -triple x86_64-windows-msvc %s

// RUN: not %clang -c -fc++-abi=InvalidABI %s 2>&1 | FileCheck %s -check-prefix=INVALID
// RUN: not %clang -c -fc++-abi=Fuchsia %s 2>&1 | FileCheck %s -check-prefix=CASE-SENSITIVE
// RUN: not %clang_cc1 -fc++-abi=InvalidABI %s 2>&1 | FileCheck %s -check-prefix=INVALID
// RUN: not %clang_cc1 -fc++-abi=Fuchsia %s 2>&1 | FileCheck %s -check-prefix=CASE-SENSITIVE
// INVALID: error: Invalid C++ ABI name 'InvalidABI'
// CASE-SENSITIVE: error: Invalid C++ ABI name 'Fuchsia'

// The flag is propgated from the driver to cc1.
// RUN: %clang -fc++-abi=InvalidABI %s -### 2>&1 | FileCheck %s -check-prefix=CC1-FLAG
// CC1-FLAG: -fc++-abi=InvalidABI

// Some C++ ABIs are not supported on some platforms.
// RUN: not %clang_cc1 -c -fc++-abi=fuchsia -triple i386 %s 2>&1 | FileCheck %s -check-prefix=UNSUPPORTED-FUCHSIA
// UNSUPPORTED-FUCHSIA: error: C++ ABI 'fuchsia' is not supported on target triple 'i386'
