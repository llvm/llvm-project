// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
__attribute__((error("oh no"))) void foo(void);

void bar(void) {
  foo();
}

// CHECK: call void @foo(), !srcloc [[SRCLOC:![0-9]+]]
// CHECK: declare{{.*}} void @foo() [[ATTR:#[0-9]+]]
// CHECK: attributes [[ATTR]] = {{{.*}}"dontcall-error"="oh no"
// CHECK: [[SRCLOC]] = !{i32 {{[0-9]+}}}
