// RUN: %clang -cc1 -triple x86_64-unknown-unknown %s -debug-info-kind=limited -fdynamic-debugging -o - \
// RUN:   -emit-llvm --discard-dynamic-debugging-debug-module \
// RUN: | FileCheck %s

// CHECK: define dso_local i32 @f() #0
// CHECK: attributes #0 =
// CHECK-SAME: noipa
// CHECK-SAME: nooutline
int f() { return 0; }
