// RUN: %clang_cc1 -x c %s -triple x86_64-linux-gnu -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -x c %s -triple x86_64-linux-gnu -emit-llvm -fno-builtin -o - | FileCheck %s

// CHECK: ; Function Attrs: returns_twice
// CHECK-NEXT: declare {{.*}} @vfork(
extern int vfork(void);
void test() {
  vfork();
}
