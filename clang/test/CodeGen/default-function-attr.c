// RUN: %clang_cc1 -triple x86_64-unknown-linux -default-function-attr foo=bar -emit-llvm %s -o - | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple x86_64-unknown-linux -default-function-attr foo=bar -emit-llvm %s -o - | FileCheck %s %}

// CHECK: define{{.*}} void @foo() #[[X:[0-9]+]]
void foo(void) {}

// CHECK: attributes #[[X]] = {{.*}} "foo"="bar"
