// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
int __attribute__((target("sse4.2"))) foo(int i, ...) { return 0; }
int __attribute__((target("arch=sandybridge"))) foo(int i, ...);
int __attribute__((target("arch=ivybridge"))) foo(int i, ...) {return 1;}
int __attribute__((target("default"))) foo(int i, ...) { return 2; }

int bar() {
  return foo(1, 'a', 1.1) + foo(2, 2.2, "asdf");
}

// CHECK: @foo.ifunc = ifunc i32 (i32, ...), i32 (i32, ...)* ()* @foo.resolver
// CHECK: define i32 @foo.sse4.2(i32 %i, ...)
// CHECK: ret i32 0
// CHECK: define i32 @foo.arch_ivybridge(i32 %i, ...)
// CHECK: ret i32 1
// CHECK: define i32 @foo(i32 %i, ...)
// CHECK: ret i32 2
// CHECK: define i32 @bar()
// CHECK: call i32 (i32, ...) @foo.ifunc(i32 1, i32 97, double
// CHECK: call i32 (i32, ...) @foo.ifunc(i32 2, double 2.2{{[0-9Ee+]+}}, i8* getelementptr inbounds 
// CHECK: define i32 (i32, ...)* @foo.resolver()
// CHECK: ret i32 (i32, ...)* @foo.arch_sandybridge
// CHECK: ret i32 (i32, ...)* @foo.arch_ivybridge
// CHECK: ret i32 (i32, ...)* @foo.sse4.2
// CHECK: ret i32 (i32, ...)* @foo
// CHECK: declare i32 @foo.arch_sandybridge(i32, ...)
