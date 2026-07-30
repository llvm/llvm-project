// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -Oz -o - %s | FileCheck %s

// CHECK: define{{.*}} void @f() local_unnamed_addr [[ATTRS:#[0-9]+]] {
void f(void) __attribute__((leaf));

void f(void)
{
}

// CHECK: attributes [[ATTRS]] = { {{.*}}nocallback{{.*}} }
