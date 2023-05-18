// RUN: %clang_cc1  -fsyntax-only -verify -triple x86_64-apple-darwin %s
// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -verify %s
// expected-no-diagnostics

enum __attribute__((deprecated)) __attribute__((weak_import)) A {
    a0
};

