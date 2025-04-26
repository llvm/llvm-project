// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// https://github.com/llvm/llvm-project/issues/62544

int printf(const char *, ...);
int a, b, c;
long d;
int main() {
  short *e = &a;
  int *f = &a;
  *f = 0;
  for (; b <= 9; b++) {
    int **g = &f;
    *f = d;
    *g = &c;
  }

  // CHECK:      TypeSanitizer: type-aliasing-violation on address
  // CHECK-NEXT: WRITE of size 2 at {{.+}} with type short accesses an existing object of type int
  // CHECK-NEXT:   in main {{.*/?}}violation-pr62544.c:22
  *e = 3;
  printf("%d\n", a);
}
