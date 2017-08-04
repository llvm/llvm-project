// RUN: %clangxx -std=c++11 -fsanitize=pointer-overflow %s -o %t
// RUN: %t 2>&1 | FileCheck %s

int main(int argc, char *argv[]) {
  char c;
  char *p = &c;
  unsigned long long neg_1 = -1;

  // CHECK: unsigned-index-expression.cpp:[[@LINE+1]]:15: runtime error: addition of unsigned offset to 0x{{.*}} overflowed to 0x{{.*}}
  char *q = p + neg_1;

  // CHECK: unsigned-index-expression.cpp:[[@LINE+1]]:16: runtime error: subtraction of unsigned offset from 0x{{.*}} overflowed to 0x{{.*}}
  char *q1 = p - neg_1;

  // CHECK: unsigned-index-expression.cpp:[[@LINE+2]]:16: runtime error: pointer index expression with base 0x{{0*}} overflowed to 0x{{.*}}
  char *n = nullptr;
  char *q2 = n - 1ULL;

  return 0;
}
