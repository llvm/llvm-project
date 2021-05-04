// These shouldn't be valid -fc++-abi values.
// RUN: not %clang_cc1 -S -emit-llvm -o /dev/null -fc++-abi=InvalidABI %s 2>&1 | FileCheck %s -check-prefix=INVALID
// RUN: not %clang_cc1 -S -emit-llvm -o /dev/null -fc++-abi=Fuchsia %s 2>&1 | FileCheck %s -check-prefix=CASE-SENSITIVE
// INVALID: error: Invalid C++ ABI name 'InvalidABI'
// CASE-SENSITIVE: error: Invalid C++ ABI name 'Fuchsia'

// Some C++ ABIs are not supported on some platforms.
// RUN: not %clang_cc1 -S -emit-llvm -o /dev/null -fc++-abi=fuchsia -triple i386 %s 2>&1 | FileCheck %s -check-prefix=UNSUPPORTED-FUCHSIA
// UNSUPPORTED-FUCHSIA: error: C++ ABI 'fuchsia' is not supported on target triple 'i386'
