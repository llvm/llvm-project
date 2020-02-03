// RUN: %clang_cc1 -triple x86_64-unknown-linux -default-function-attr foo=bar -emit-llvm %s -o - | FileCheck %s

// CHECK: define dso_local void @foo() #[[X:[0-9]+]]
void foo() {}

// CHECK: attributes #[[X]] = {{.*}} "foo"="bar"
