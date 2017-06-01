// RUN: %clang_cc1 -triple x86_64-unknown-linux -O1 -fblocks \
// RUN:   -fsanitize=cfi-icall -fsanitize-cfi-cross-dso \
// RUN:   -emit-llvm -o - %s | FileCheck %s

// CHECK: define void @f() {{.*}} !type !{{.*}} !type !{{.*}}
void f(void);
void (*pf)(void) = f;
void f(void) { }

// Check that we do not crash on non-FunctionDecl definitions.
void (^g)(void) = ^{};
