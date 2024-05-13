// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
__attribute__((warning("oh no"))) void foo(void);

void bar(void) {
  foo();
}

// CHECK: call void @foo(), !srcloc [[SRCLOC:![0-9]+]]
// CHECK: declare{{.*}} void @foo() [[ATTR:#[0-9]+]]
// CHECK: attributes [[ATTR]] = {{{.*}}"dontcall-warn"="oh no"
// CHECK: [[SRCLOC]] = !{i32 {{[0-9]+}}}
