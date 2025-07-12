// RUN: %clang_cc1 -verify %s -triple x86_64-unknown-unknown -DTEST_x64
// RUN: %clang_cc1 -verify %s -triple i386-unknown-unknown -DTEST_x86
// RUN: %clang_cc1 -verify %s -triple riscv32-unknown-unknown -DTEST_riscv32
// RUN: %clang_cc1 -verify %s -triple riscv64-unknown-unknown -DTEST_riscv64
// RUN: %clang_cc1 -verify %s -triple aarch64-unknown-unknown -DTEST_aarch64

#if defined(TEST_x64) || defined(TEST_x86)
// expected-no-diagnostics
void *a() {
return __builtin_stack_address();
}
#else
void *a() {
return __builtin_stack_address(); // expected-error {{builtin is not supported on this target}}
}
#endif
