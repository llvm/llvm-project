// These should succeed.
// RUN: %clang_cc1 -fc++-abi=itanium %s
// RUN: %clang_cc1 -fc++-abi=arm %s
// RUN: %clang_cc1 -fc++-abi=ios %s
// RUN: %clang_cc1 -fc++-abi=ios64 %s
// RUN: %clang_cc1 -fc++-abi=aarch64 %s
// RUN: %clang_cc1 -fc++-abi=mips %s
// RUN: %clang_cc1 -fc++-abi=webassembly %s
// RUN: %clang_cc1 -fc++-abi=fuchsia %s
// RUN: %clang_cc1 -fc++-abi=xl %s
// RUN: %clang_cc1 -fc++-abi=microsoft %s

// RUN: not %clang_cc1 -fc++-abi=InvalidABI %s 2>&1 | FileCheck %s -check-prefix=INVALID
// RUN: not %clang_cc1 -fc++-abi=Fuchsia %s 2>&1 | FileCheck %s -check-prefix=CASE-SENSITIVE
// INVALID: error: Invalid C++ ABI name 'InvalidABI'
// CASE-SENSITIVE: error: Invalid C++ ABI name 'Fuchsia'
