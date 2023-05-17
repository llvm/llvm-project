// RUN: %clang_cc1 -mrtd -triple i386-unknown-unknown -std=c89 -emit-llvm -o - %s 2>&1 | FileCheck %s

// CHECK: mrtd.c:10:3: warning: function with no prototype cannot use the stdcall calling convention

void baz(int arg);

// CHECK: define{{.*}} x86_stdcallcc void @foo(i32 noundef %arg) [[NUW:#[0-9]+]]
void foo(int arg) {
// CHECK: call x86_stdcallcc i32 @bar(
  bar(arg);
// CHECK: call x86_stdcallcc void @baz(i32
  baz(arg);
}

// CHECK: declare x86_stdcallcc i32 @bar(...)

// CHECK: declare x86_stdcallcc void @baz(i32 noundef)

void qux(int arg, ...) { }
// CHECK: define{{.*}} void @qux(i32 noundef %arg, ...)

void quux(int a1, int a2, int a3) {
  qux(a1, a2, a3);
}
// CHECK-LABEL: define{{.*}} x86_stdcallcc void @quux
// CHECK: call void (i32, ...) @qux

// CHECK: attributes [[NUW]] = { noinline nounwind{{.*}} }
