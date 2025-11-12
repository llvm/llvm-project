// RUN: %clang_min_runtime -fsanitize=alignment %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK

void f(int &n) {}

int *t;

int main() {
  int r;
  t = (int *)(((char *)&r) + 1);
  // CHECK: ubsan: type-mismatch by 0x{{[[:xdigit:]]+}} address 0x{{[[:xdigit:]]+$}}
  // CHECK-NOT: type-mismatch
  f(*t);
}
