// RUN: %clang_cc1 -verify=x86 %s -triple x86_64-unknown-unknown
// RUN: %clang_cc1 -verify=x86 %s -triple i386-unknown-unknown
// RUN: %clang_cc1 -verify %s -triple riscv32-unknown-unknown
// RUN: %clang_cc1 -verify %s -triple riscv64-unknown-unknown
// RUN: %clang_cc1 -verify %s -triple aarch64-unknown-unknown

void *a() {
return __builtin_stack_address(); // expected-error {{builtin is not supported on this target}}
                                  // x86-no-diagnostics
}
