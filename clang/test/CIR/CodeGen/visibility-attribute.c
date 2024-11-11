// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s -input-file=%t.cir -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck %s -input-file=%t.ll -check-prefix=LLVM

extern int glob_default;
// CIR: cir.global "private" external @glob_default : !s32i
// LLVM: @glob_default = external global i32

extern int __attribute__((__visibility__("hidden"))) glob_hidden;
// CIR: cir.global "private" hidden external @glob_hidden : !s32i
// LLVM: @glob_hidden = external hidden global i32

extern int __attribute__((__visibility__("protected"))) glob_protected;
// CIR: cir.global "private" protected external @glob_protected : !s32i
// LLVM: @glob_protected = external protected global i32

int call_glob()
{
  return glob_default + glob_hidden + glob_protected;
}

void foo_default();
// CIR: cir.func no_proto private @foo_default(...)
// LLVM: declare {{.*}} void @foo_default(...)

void __attribute__((__visibility__("hidden"))) foo_hidden();
// CIR: cir.func no_proto private hidden @foo_hidden(...)
// LLVM: declare {{.*}} hidden void @foo_hidden(...)

void __attribute__((__visibility__("protected"))) foo_protected();
// CIR: cir.func no_proto private protected @foo_protected(...)
// LLVM: declare {{.*}} protected void @foo_protected(...)

static void static_foo_default() {}
// CIR: cir.func no_proto internal private @static_foo_default()
// LLVM: define internal void @static_foo_default()

static void __attribute__((__visibility__("hidden"))) static_foo_hidden() {}
// CIR: cir.func no_proto internal private @static_foo_hidden()
// LLVM: define internal void @static_foo_hidden()

static void __attribute__((__visibility__("protected"))) static_foo_protected() {}
// CIR: cir.func no_proto internal private @static_foo_protected()
// LLVM: define internal void @static_foo_protected()

void call_foo()
{
  foo_default();
  foo_hidden();
  foo_protected();
  static_foo_default();
  static_foo_hidden();
  static_foo_protected();
}
