// RUN: %clang_cc1 -triple armv7-none-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -fsyntax-only -verify %s
// expected-no-diagnostics

void __clear_cache(void *a, void *b) {}
