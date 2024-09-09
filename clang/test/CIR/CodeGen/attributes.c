// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o -  | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM

extern int __attribute__((section(".shared"))) ext;
int getExt() {
  return ext;
}
// CIR:   cir.global "private" external @ext : !s32i {alignment = 4 : i64, section = ".shared"}
// LLVM:  @ext = external global i32, section ".shared"

int __attribute__((section(".shared"))) glob = 42;
// CIR:   cir.global external @glob = #cir.int<42> : !s32i {alignment = 4 : i64, section = ".shared"}
// LLVM:   @glob = global i32 42, section ".shared"


void __attribute__((__visibility__("hidden"))) foo();
// CIR: cir.func no_proto private hidden @foo(...) extra(#fn_attr)
int bah()
{
  foo();
  return 1;
}
