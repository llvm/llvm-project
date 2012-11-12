// RUN: %clang_cc1 %s -o - -emit-llvm | FileCheck %s
// PR1513

struct s{
long a;
long b;
};

void f(struct s a, char *b, signed char C) {
  // CHECK: i8 signext

}
