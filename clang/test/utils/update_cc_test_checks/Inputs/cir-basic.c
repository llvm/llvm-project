// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -o - %s | FileCheck %s --check-prefix=CIR

int add(int a, int b) { return a + b; }

int element(int i) {
  int a[4] = {1, 2, 3, 4};
  return a[i];
}
