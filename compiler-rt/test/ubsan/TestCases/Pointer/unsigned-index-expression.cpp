// RUN: %clangxx -fsanitize=pointer-overflow %s -o %t
// RUN: %t 2>&1 | FileCheck %s

int main(int argc, char *argv[]) {
  char c;
  char *p = &c;
  unsigned long long offset = -1;

  // CHECK: unsigned-index-expression.cpp:[[@LINE+1]]:15: runtime error: unsigned pointer index expression result is 0x{{.*}}, preceding its base 0x{{.*}}
  char *q = p + offset;

  return 0;
}
