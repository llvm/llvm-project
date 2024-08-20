// RUN: %clang_cc1 -triple aarch64-unknown-windows-msvc -fsyntax-only %s -verify
// RUN: %clang_cc1 -triple thumbv7-unknown-windows-msvc -fsyntax-only %s -verify
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fsyntax-only %s -verify
// RISC-V does not support swiftcall
// RUN: %clang_cc1 -triple riscv32-unknown-elf -fsyntax-only %s -verify

#if __has_extension(swiftcc)
// expected-no-diagnostics
#else
// expected-warning@+2 {{'__swiftcall__' calling convention is not supported for this target}}
#endif
void __attribute__((__swiftcall__)) f(void) {}

#if __has_extension(swiftasynccc)
// expected-no-diagnostics
#else
// expected-warning@+2 {{'__swiftasynccall__' calling convention is not supported for this target}}
#endif
void __attribute__((__swiftasynccall__)) g(void) {}

// Allow has_feature in addition to has_extension to let swift transition.
// rdar://133628186
#if __has_feature(swiftasynccc)
// expected-no-diagnostics
#else
// expected-warning@+2 {{'__swiftasynccall__' calling convention is not supported for this target}}
#endif
void __attribute__((__swiftasynccall__)) h(void) {}
