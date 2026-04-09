// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

extern int __attribute__((section(".shared"))) ext;
int getExt(void) {
  return ext;
}
// CIR: cir.global "private" external @ext : !s32i {{{.*}}section = ".shared"}
// LLVM: @ext = external global i32, section ".shared"

int __attribute__((section(".shared"))) glob = 42;
// CIR: cir.global external @glob = #cir.int<42> : !s32i {{{.*}}section = ".shared"}
// LLVM: @glob = global i32 42, section ".shared"

__attribute__((section(".custom_fn"))) void func_in_section(void) {}
// CIR: cir.func {{.*}}@func_in_section() {{.*}}section = ".custom_fn"
// LLVM: define {{.*}}@func_in_section(){{.*}}section ".custom_fn"
