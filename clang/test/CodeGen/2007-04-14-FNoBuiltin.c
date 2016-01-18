// RUN: %clang_cc1 -emit-llvm %s -O2 -fno-builtin -o - | FileCheck %s
// Check that -fno-builtin is honored.

extern int printf(const char*, ...);

// CHECK: define {{.*}}void {{.*}}foo(
void foo(const char *msg) {
  // CHECK: call {{.*}}printf
  printf("%s\n",msg);
}
