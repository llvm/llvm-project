// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -o - -triple i386-apple-darwin %s | FileCheck %s
// PR9571

struct t {
  int x;
};

extern struct t *cfun;

int f(void) {
  if (!(cfun + 0))
    // CHECK: icmp ne %struct.t*
    return 0;
  return cfun->x;
}
