// RUN: %clang_asan -O0 -fsanitize-address-use-after-scope %s -o %t && %run %t

// Function jumps over variable initialization making lifetime analysis
// ambiguous. Asan should ignore such variable and program must not fail.

int *ptr;

void f(int cond) {
  if (cond)
    goto label;
  int tmp = 1;

label:
  ptr = &tmp;
  *ptr = 5;
}

int main() {
  f(1);
  return 0;
}
