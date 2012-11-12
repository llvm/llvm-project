// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

struct p {
  char a;
  int b;
} __attribute__ ((packed));

struct p t = { 1, 10 };
struct p u;

int main () {
  // CHECK: align 1
  // CHECK: align 1
  int tmp = t.b;
  u.b = tmp;
  return tmp;

}
