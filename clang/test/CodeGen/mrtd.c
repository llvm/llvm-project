// RUN: %clang_cc1 -mrtd -triple i386-unknown-unknown -std=c89 -emit-llvm -o - %s 2>&1 | FileCheck --check-prefixes=CHECK,X86 %s
// RUN: %clang_cc1 -mrtd -triple m68k-unknown-unknown -std=c89 -emit-llvm -o - %s 2>&1 | FileCheck --check-prefixes=CHECK,M68K %s

void baz(int arg);

// X86: define{{.*}} x86_stdcallcc void @foo(i32 noundef %arg) [[NUW:#[0-9]+]]
// M68K: define{{.*}} m68k_rtdcc void @foo(i32 noundef %arg)
void foo(int arg) {
// X86: call x86_stdcallcc i32 @bar(
#ifndef __mc68000__
  bar(arg);
#endif
// X86: call x86_stdcallcc void @baz(i32
// M68K: call m68k_rtdcc void @baz(i32
  baz(arg);
}

// X86: declare x86_stdcallcc i32 @bar(...)

// X86: declare x86_stdcallcc void @baz(i32 noundef)
// M68K: declare m68k_rtdcc void @baz(i32 noundef)

void qux(int arg, ...) { }
// CHECK: define{{.*}} void @qux(i32 noundef %arg, ...)

void quux(int a1, int a2, int a3) {
  qux(a1, a2, a3);
}
// X86-LABEL: define{{.*}} x86_stdcallcc void @quux
// M68K-LABEL: define{{.*}} m68k_rtdcc void @quux
// CHECK: call void (i32, ...) @qux

// X86: attributes [[NUW]] = { noinline nounwind{{.*}} }
