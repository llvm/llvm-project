// RUN: %clang_cc1 -mrtd -triple i386-unknown-freebsd9.0 -emit-llvm -o - %s | FileCheck %s

void baz(int arg);

// CHECK: define x86_stdcallcc void @foo(i32 %arg) nounwind
void foo(int arg) {
// CHECK: call x86_stdcallcc i32 bitcast (i32 (...)* @bar to i32 (i32)*)(
  bar(arg);
// CHECK: call x86_stdcallcc void @baz(i32
  baz(arg);
}

// CHECK: declare x86_stdcallcc i32 @bar(...)

// CHECK: declare x86_stdcallcc void @baz(i32)
