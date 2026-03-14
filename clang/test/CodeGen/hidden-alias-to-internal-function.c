// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - -x c %s | FileCheck %s
// Reproduce the crash in PR19760.
static void foo(void) {}
void bar(void) __attribute__((alias("foo")))
__attribute__((visibility("hidden")));

// CHECK: @bar = hidden alias void (), ptr @foo
// CHECK: define internal void @foo()
