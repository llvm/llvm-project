// RUN: %clang -mbackchain --target=s390x-linux -S -emit-llvm -o - %s | FileCheck %s

// CHECK: define dso_local void @foo() [[NUW:#[0-9]+]]
void foo(void) {
}

// CHECK: attributes [[NUW]] = { {{.*}} "backchain" {{.*}} }
