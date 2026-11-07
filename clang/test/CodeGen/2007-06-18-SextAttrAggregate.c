// RUN: %clang_cc1 -triple=sparc64-unknown-linux-gnu -no-enable-noundef-analysis %s -o - -emit-llvm | FileCheck %s

// PR1513

// The triple here is chosen so that the struct is passed in two arguments,
// and the signed char requires sign extension.

struct s{
long a;
long b;
};

void f(struct s a, char *b, signed char C) {
  // CHECK: i8 signext

}
