// RUN: clang-tidy -checks='-*,bugprone-casting-through-void' %s -- -std=c99 2>&1 | FileCheck %s --allow-empty
// CHECK-NOT: warning:

double d = 100;

void normal_test() {
  (int *)(void *)&d;
  int x = 1;
  char *y = (char*)(void*)&x;
  char *z = (char*)&x;
}
