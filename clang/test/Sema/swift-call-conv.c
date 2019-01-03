// RUN: %clang_cc1 -triple aarch64-unknown-windows-msvc -fsyntax-only %s -verify
// RUN: %clang_cc1 -triple thumbv7-unknown-windows-msvc -fsyntax-only %s -verify
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fsyntax-only %s -verify

// expected-no-diagnostics

void __attribute__((__swiftcall__)) f(void) {}
