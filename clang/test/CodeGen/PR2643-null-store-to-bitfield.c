// RUN: %clang_cc1 -emit-llvm -o - %s
// PR2643

void foo(void) {
  struct {
    int a : 1;
    int b : 1;
  } entry = {0};
}

