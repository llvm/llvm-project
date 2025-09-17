// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++17 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++23 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++2c %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// expected-no-diagnostics

// This file intentionally does not end with a newline. CWG787 made this
// well-defined behavior.

// cwg787: 21