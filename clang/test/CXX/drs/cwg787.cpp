// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++98 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++14 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++17 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++23 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++2c %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify

// expected-no-diagnostics

// This file intentionally does not end with a newline. CWG787 made this
// well-defined behavior.

// cwg787: 21