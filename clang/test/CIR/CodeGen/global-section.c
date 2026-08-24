// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

extern int __attribute__((section(".shared"))) ext;
int getExt(void) {
  return ext;
}
// CIR-DAG: cir.global "private" external @ext : !s32i {{{.*}}section = ".shared"}
// LLVM-DAG: @ext = external global i32, section ".shared"

int __attribute__((section(".shared"))) glob = 42;
// CIR-DAG: cir.global external @glob = #cir.int<42> : !s32i {{{.*}}section = ".shared"}
// LLVM-DAG: @glob = global i32 42, section ".shared"

int getStaticLocal(void) {
  static int __attribute__((section(".static_local"))) sloc = 7;
  return ++sloc;
}
// CIR-DAG: cir.global "private" internal dso_local @getStaticLocal.sloc = #cir.int<7> : !s32i {{{.*}}section = ".static_local"}
// LLVM-DAG: @getStaticLocal.sloc = internal global i32 7, section ".static_local"

__attribute__((section(".custom_fn"))) void func_in_section(void) {}
// CIR: cir.func {{.*}}@func_in_section() {{.*}}section = ".custom_fn"
// LLVM: define {{.*}}@func_in_section(){{.*}}section ".custom_fn"
